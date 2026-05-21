import argparse
import json
import torch
import hydra
from torch import nn
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Callable, List, Dict, Tuple, Literal
from tqdm.auto import tqdm

from ego4d_goalstep.utils.clusters import clusterize, compress
from ego4d_goalstep.utils.evaluate import display_results, evaluate_nlq_performance
from ego4d_goalstep.utils.utils import load_features
from utils.random import seed_everything

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_grad_enabled(False)


def build_hiero_fe(model: nn.Module, task: nn.Module, stride: int = 16, fps: int = 30, device: torch.device = "cuda") -> Callable[[torch.Tensor], torch.Tensor]:
    node_length = stride / fps

    def visual_fe(features: torch.Tensor):
        pos = torch.arange(0, features.shape[0], device=device) * node_length
        indices = torch.arange(0, features.shape[0], device=device)
        batch = torch.zeros_like(pos, dtype=torch.long)
        mask = torch.ones_like(pos, dtype=torch.int).bool()
        data = Data(x=features.unsqueeze(1), pos=pos, indices=indices, batch=batch, mask=mask)

        graphs = model(data.to(device=device))
        features = task(graphs, data)

        output = {}
        for d in range(model.depth):
            mask = graphs.depth == d
            output[d] = {
                "features": features[mask],
                "assignments": graphs.assignments[mask] if hasattr(graphs, "assignments") else None
            }
        return output

    def text_fe(text: List[str]):
        queries_features = task.encode_text(text)
        return nn.functional.normalize(queries_features, p=2, dim=-1)

    return visual_fe, text_fe


def build_encoders(
    features: Literal["omnivore_video_swinl", "egovlp"], ckpt: Optional[str], device: torch.device = "cuda"
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[List[str]], torch.Tensor]]:
    if features == "egovlp" and ckpt is None:
        print("Building vanilla EgoVLP video and text encoders...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", TOKENIZERS_PARALLELISM=False)
        text_model = AutoModel.from_pretrained("distilbert-base-uncased", cache_dir="pretrained/distilbert-base-uncased")
        text_model = text_model.to(device)
        text_proj = nn.Sequential(nn.ReLU(), nn.Linear(768, 256)).to(device)
        text_model.load_state_dict(torch.load("pretrained/egovlp_text.pth", weights_only=True), strict=True)
        text_proj.load_state_dict(torch.load("pretrained/egovlp_txt_proj.pth", weights_only=True), strict=True)
        text_model = text_model.eval()
        text_proj = text_proj.eval()

        def text_encoder_fe(text: List[str]):
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
            queries = text_proj(text_model(**tokens).last_hidden_state[:, 0, :])
            return nn.functional.normalize(queries, p=2, dim=-1)

        return None, text_encoder_fe

    print(f"Building HiERO with {features} features using checkpoint {ckpt}...")
    state = torch.load(ckpt, weights_only=False)
    input_size = 1536 if "omnivore" in features else 256
    model = hydra.utils.instantiate(state["config"]["model"], clustering_at_inference=True, input_size=input_size, _recursive_=False).to(device)
    task = hydra.utils.instantiate(state["config"]["task"], _recursive_=False).to(device).eval()
    
    model_state = state["model"]
    # Backward compatibility for legacy single DMoN projector checkpoints
    if "dmon_projector.0.weight" in model_state and "dmon_projector.1.0.weight" not in model_state:
        print("Migrating legacy DMoN projector weights to depth-specific format...")
        old_w0 = model_state.pop("dmon_projector.0.weight")
        old_b0 = model_state.pop("dmon_projector.0.bias")
        old_w2 = model_state.pop("dmon_projector.2.weight")
        old_b2 = model_state.pop("dmon_projector.2.bias")
        
        for d in range(1, model.depth):
            model_state[f"dmon_projector.{d}.0.weight"] = old_w0.clone()
            model_state[f"dmon_projector.{d}.0.bias"] = old_b0.clone()
            model_state[f"dmon_projector.{d}.2.weight"] = old_w2.clone()
            model_state[f"dmon_projector.{d}.2.bias"] = old_b2.clone()

    model.load_state_dict(state["model"], strict=True)
    task.load_state_dict(state["task"], strict=True)
    return build_hiero_fe(model, task)


def build_level_cache(
    raw_video_features: torch.Tensor,
    visual_encoder: Optional[Callable],
    target_hierarchical_level: int,
    temporal_compression_factor: int,
    num_clusters: int,
    seconds_per_node: float
) -> Dict[int, dict]:
    """
    Builds a cache containing processed video features, clusters, and temporal parameters
    for the base level (0) and the specified target hierarchical level.
    """
    # Define the required levels: the leaf level (0) and the desired abstraction level
    required_levels = [0, target_hierarchical_level]
    # If a visual encoder (HiERO) is available, pass the raw features to get the hierarchical output
    hierarchical_output = visual_encoder(raw_video_features) if visual_encoder is not None else None
    # Initialize the dictionary that will cache the extracted data for each level
    level_data_cache = {}

    # Iterate over the required levels to populate the cache
    for current_level in required_levels:
        # Calculate the expected temporal compression at this level (halves for each depth level)
        current_level_compression = max(1, int(temporal_compression_factor / (2 ** current_level)))

        # Initialize features for this level as the raw ones and initialize cluster assignments to None
        level_specific_features = raw_video_features
        cluster_assignments = None

        # If the encoder returned hierarchical output, extract data for the current level
        if hierarchical_output is not None:
            level_data = hierarchical_output[current_level]
            level_specific_features = level_data["features"]
            cluster_assignments = level_data["assignments"]

        if current_level == 0:
            # At level 0 (base), every node/frame is a cluster of length 1, and 
            # the cluster features coincide with the node features themselves
            temporal_clusters = [(i, 1) for i in range(len(level_specific_features))]
            cluster_features = level_specific_features
        else:
            # Use default spectral clustering on the level features
            temporal_clusters = compress(clusterize(level_specific_features, num_clusters))

        if current_level > 0:
            # Filter out clusters that are shorter than the required compression
            temporal_clusters = [(start_idx, length) for (start_idx, length) in temporal_clusters if length > current_level_compression]
            # Calculate the average feature for each cluster by averaging the features of its nodes
            cluster_features = [level_specific_features[start_idx : start_idx + length].mean(0) for (start_idx, length) in temporal_clusters]
            # If we found valid clusters, stack them into a tensor and apply L2 normalization
            if cluster_features:
                cluster_features = torch.nn.functional.normalize(torch.stack(cluster_features), p=2, dim=-1)
            else:
                # Otherwise, create an empty tensor of the correct shape
                cluster_features = torch.empty((0, level_specific_features.shape[-1]), device=level_specific_features.device)

        # Determine the actual number of nodes present at this level
        actual_node_count = len(cluster_assignments) if (current_level > 0 and cluster_assignments is not None and len(cluster_assignments) > 0 and cluster_assignments[0].item() != -1) else len(level_specific_features)
        # Scale the length in seconds of a node at this level relative to the raw level
        dynamic_seconds_per_node = (len(raw_video_features) / max(1, actual_node_count)) * seconds_per_node
        # Save all computed information for the current level in the cache
        level_data_cache[current_level] = {
            "clusters": temporal_clusters,               # List of tuples (start_idx, length)
            "feature_clusters": cluster_features,        # Tensor with the mean features of the clusters
            "features": level_specific_features,         # Features for each individual node of the level
            "node_length": dynamic_seconds_per_node,     # Duration in seconds of each node at this level
            "actual_nodes": actual_node_count            # Total number of nodes
        }

    return level_data_cache


def compute_hybrid_similarities(
    query_feature: torch.Tensor,
    target_level_data: dict,
    base_level_data: dict,
    alpha_weight: float = 0.7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the hybrid similarity between the textual query feature and the video features,
    fusing fine-grained similarity (level 0) with coarse-grained similarity (target level).
    """
    base_normalized_features = torch.nn.functional.normalize(base_level_data["features"], p=2, dim=-1)
    # Compute the cosine similarity between the query and all base features
    base_similarity_scores = (query_feature @ base_normalized_features.T)
    # Extract the aggregated cluster features from the target level
    target_cluster_features = target_level_data["feature_clusters"]

    # If there are valid clusters at the target level, fuse the similarities
    if len(target_cluster_features) > 0:
        # Initialize a tensor of mapped similarities with a minimum value (-1.0)
        target_mapped_similarities = torch.zeros_like(base_similarity_scores) - 1.0
        # Scaling factor to map indices from the target level back to the base level
        level_scaling_factor = len(base_normalized_features) / max(1, target_level_data["actual_nodes"])

        # For each cluster at the target level, calculate the similarity and spread it over corresponding base nodes
        for cluster_idx, (cluster_start, cluster_length) in enumerate(target_level_data["clusters"]):
            # Calculate start and end indices projected onto the base level
            mapped_start_idx = int(cluster_start * level_scaling_factor)
            mapped_end_idx = int((cluster_start + cluster_length) * level_scaling_factor)
            # If the mapped start is valid, assign the similarity between the query and the cluster
            if mapped_start_idx < len(target_mapped_similarities):
                target_mapped_similarities[mapped_start_idx:min(mapped_end_idx, len(target_mapped_similarities))] = (query_feature @ target_cluster_features[cluster_idx]).item()

        final_similarities = alpha_weight * base_similarity_scores + (1 - alpha_weight) * target_mapped_similarities
    else:
        # If there are no clusters at the target level, only use the base similarity
        final_similarities = base_similarity_scores

    # Calculate the start times (in seconds) for each node of the base level
    node_start_times_seconds = torch.arange(len(base_similarity_scores), device=final_similarities.device) * base_level_data["node_length"]
    
    return final_similarities, node_start_times_seconds


def run_viterbi_decoding(
    query_similarities_list: List[torch.Tensor], 
    query_start_times_list: List[torch.Tensor]
) -> Dict[int, int]:
    """
    Applies the Viterbi algorithm to find the optimal temporal sequence of procedural steps.
    Ensures that the predicted order strictly respects the real temporal order of the queries.
    """
    num_queries = len(query_similarities_list)
    if num_queries == 0: return {}

    # Initialize the dynamic programming (DP) table and backpointers to reconstruct the path
    dp_table = [torch.zeros_like(sims) for sims in query_similarities_list]
    backpointer_table = [torch.zeros_like(sims, dtype=torch.long) for sims in query_similarities_list]

    # The first step directly takes the similarities of the first query
    dp_table[0] = query_similarities_list[0]

    # Iterate over all subsequent queries to compute the best paths
    for query_idx in range(1, num_queries):
        # Create a temporal mask to enforce monotonicity: current step must happen after (or at most 1 sec before) the previous step
        temporal_mask = query_start_times_list[query_idx-1].unsqueeze(0) <= query_start_times_list[query_idx].unsqueeze(1) + 1.0
        # Apply the mask to previous DP scores; invalid paths receive a huge penalty (-1e9)
        masked_previous_dp = dp_table[query_idx-1].unsqueeze(0) * temporal_mask + (-1e9) * (~temporal_mask)
        # Find the maximum score and the origin index (backpointer) for each current node
        max_previous_scores, best_previous_indices = masked_previous_dp.max(dim=1)
        # Update the DP table and store the backpointers
        dp_table[query_idx] = query_similarities_list[query_idx] + max_previous_scores
        backpointer_table[query_idx] = best_previous_indices

    # If the alignment failed (score too low), fallback to individual maximum predictions (argmax) ignoring order
    if dp_table[-1].max() <= -1e8:
        optimal_path = [sims.argmax().item() for sims in query_similarities_list]
    else:
        # Reconstruct the optimal path starting from the last query and going backwards (backtracking)
        optimal_path = []
        current_best_node = dp_table[-1].argmax().item()
        optimal_path.append(current_best_node)
        
        for query_idx in range(num_queries-1, 0, -1):
            current_best_node = backpointer_table[query_idx][current_best_node].item()
            optimal_path.append(current_best_node)
        
        optimal_path.reverse()

    # Return a dictionary mapping the query index to the optimal node index
    return {q_idx: optimal_path[q_idx] for q_idx in range(num_queries)}


def get_top_peaks(
    similarities: torch.Tensor, 
    top1_idx: int, 
    prev_idx: Optional[int] = None, 
    next_idx: Optional[int] = None, 
    top_k: int = 15, 
    nms_window: int = 5
) -> List[int]:
    """
    Extracts the top K local similarity peaks, applying a Non-Maximum Suppression (NMS) technique.
    Generates a mix of peaks:
    - Bounded peaks (respecting Viterbi context)
    - Unbounded peaks (global context)
    """
    final_top_peaks = [top1_idx]
    
    # 1. Extract bounded peaks
    temp_sims_bounded = similarities.clone()
    if prev_idx is not None:
        temp_sims_bounded[:max(0, prev_idx - 2)] = -1e9
    if next_idx is not None:
        temp_sims_bounded[next_idx + 3:] = -1e9
        
    bounded_indices = []
    for _ in range(top_k):
        if temp_sims_bounded.max() <= -1e8: break
        curr_max = temp_sims_bounded.argmax().item()
        bounded_indices.append(curr_max)
        
        suppression_start = max(0, curr_max - nms_window)
        suppression_end = min(len(temp_sims_bounded), curr_max + nms_window + 1)
        temp_sims_bounded[suppression_start:suppression_end] = -1e9

    # Add bounded peaks ensuring they don't overlap with the Viterbi top-1
    for peak_idx in bounded_indices:
        if peak_idx != top1_idx and abs(peak_idx - top1_idx) > nms_window:
            final_top_peaks.append(peak_idx)

    # 2. Extract unbounded peaks
    temp_sims_unbounded = similarities.clone()
    unbounded_indices = []
    for _ in range(top_k):
        if temp_sims_unbounded.max() <= -1e8: break
        curr_max = temp_sims_unbounded.argmax().item()
        unbounded_indices.append(curr_max)
        
        suppression_start = max(0, curr_max - nms_window)
        suppression_end = min(len(temp_sims_unbounded), curr_max + nms_window + 1)
        temp_sims_unbounded[suppression_start:suppression_end] = -1e9

    for peak_idx in unbounded_indices:
        is_novel = all(abs(peak_idx - existing_peak) > nms_window for existing_peak in final_top_peaks)
        if is_novel:
            final_top_peaks.append(peak_idx)
            
    return final_top_peaks


def expand_and_pad_segments(
    query_feature: torch.Tensor,
    all_base_features: torch.Tensor,
    peak_indices: List[int],
    seconds_per_node: float,
    clip_start_video_seconds: float,
    expansion_threshold_ratios: List[float] = [0.5, 0.6, 0.4],
    absolute_minimum_similarity: float = 0.4,
    expansion_patience: int = 3,
    minimum_duration_seconds: float = 2,
    local_window_size: int = 150
) -> List[List[float]]:
    """
    For each identified peak, expands the boundaries to the left and right by dynamically calculating
    local similarities to find the actual start and end of the segment.
    Also adds padding if the predicted duration is below a minimum threshold.
    """
    expanded_predictions = []

    # Iterate over all identified peak indices
    for global_peak_idx in peak_indices:
        # Crop a local temporal window around the peak for expansion
        window_start_idx = max(0, global_peak_idx - local_window_size)
        window_end_idx = min(len(all_base_features), global_peak_idx + local_window_size + 1)
        local_features_normalized = torch.nn.functional.normalize(all_base_features[window_start_idx:window_end_idx], p=2, dim=-1)
        local_similarities = (query_feature @ local_features_normalized.T)
        local_peak_idx = max(0, min(global_peak_idx - window_start_idx, len(local_similarities) - 1))
        
        # Expand the same peak with different strictness thresholds
        for expansion_ratio in expansion_threshold_ratios:
            # Calculate the stopping threshold for expansion (a percentage of peak similarity, but not less than the absolute minimum)
            stopping_threshold = max(min(local_similarities[local_peak_idx].item() * expansion_ratio, absolute_minimum_similarity), 0.0)

            def expand_direction(step_direction: int) -> int:
                current_idx = local_peak_idx
                tolerance_strikes = 0
                last_valid_idx = local_peak_idx
                
                while 0 <= current_idx + step_direction < len(local_similarities):
                    current_idx += step_direction
                    
                    if local_similarities[current_idx].item() >= stopping_threshold:
                        tolerance_strikes = 0
                        last_valid_idx = current_idx
                    else:
                        tolerance_strikes += 1
                        if tolerance_strikes > expansion_patience: break
                
                return last_valid_idx

            refined_start_idx = window_start_idx + expand_direction(-1)
            refined_end_idx = window_start_idx + expand_direction(1) + 1

            # Convert node indices to continuous time (seconds) by subtracting the clip start relative to the original video
            predicted_start_sec = refined_start_idx * seconds_per_node - clip_start_video_seconds
            predicted_end_sec = refined_end_idx * seconds_per_node - clip_start_video_seconds
            # Apply padding ensuring a minimum segment duration to avoid overly short fragments
            current_duration = predicted_end_sec - predicted_start_sec
            if current_duration < minimum_duration_seconds:
                padding_needed = (minimum_duration_seconds - current_duration) / 2
                # Prevent the start time from becoming negative
                predicted_start_sec = max(0.0, predicted_start_sec - padding_needed)
                predicted_end_sec += padding_needed

            expanded_predictions.append([predicted_start_sec, predicted_end_sec])

    return expanded_predictions


def evaluate_grounding(args, dataset_annotations: dict, visual_encoder: Optional[Callable], text_encoder: Optional[Callable]):
    """
    Main function that iterates over all videos and clips in the dataset to perform temporal predictions.
    """
    all_predictions = []
    
    for video_data in tqdm(dataset_annotations["videos"], leave=False):
        video_uid = video_data["video_uid"].replace("grp-", "")

        try:
            raw_video_features = load_features(video_uid, root=f"data/ego4d/raw/features/{args.features}").float()
        except FileNotFoundError:
            continue

        level_data_cache = build_level_cache(
            raw_video_features=raw_video_features, 
            visual_encoder=visual_encoder, 
            target_hierarchical_level=args.target_level, 
            temporal_compression_factor=args.threshold, 
            num_clusters=args.n_clusters, 
            seconds_per_node=16/30
        )

        for clip_data in video_data["clips"]:
            for annotation_data in clip_data["annotations"]:
                if text_encoder is None: continue

                # Format text queries by prepending '#C C ' to respect the standard Ego4D prompt
                text_queries = [f"#C C {q['query']}" for q in annotation_data["language_queries"]]
                if not text_queries: continue
                
                query_embeddings = text_encoder(text_queries)
                
                query_similarities_list = []
                query_starts_list = []
                
                # For each query embedding, calculate the similarity map
                for q_emb in query_embeddings:
                    similarity_map, start_times_map = compute_hybrid_similarities(
                        query_feature=q_emb, 
                        target_level_data=level_data_cache[args.target_level], 
                        base_level_data=level_data_cache[0]
                    )
                    query_similarities_list.append(similarity_map)
                    query_starts_list.append(start_times_map)

                # Apply Viterbi to find the optimal temporal sequence respecting step order
                viterbi_optimal_paths = run_viterbi_decoding(query_similarities_list, query_starts_list)

                # For each processed query, refine the prediction
                for query_idx, q_emb in enumerate(query_embeddings):
                    prev_viterbi = viterbi_optimal_paths[query_idx - 1] if query_idx > 0 else None
                    next_viterbi = viterbi_optimal_paths[query_idx + 1] if query_idx < len(query_embeddings) - 1 else None

                    top_local_peaks = get_top_peaks(
                        similarities=query_similarities_list[query_idx], 
                        top1_idx=viterbi_optimal_paths[query_idx],
                        prev_idx=prev_viterbi,
                        next_idx=next_viterbi
                    )
                    
                    # Expand segment boundaries starting from the found peaks
                    expanded_segment_predictions = expand_and_pad_segments(
                        query_feature=q_emb,
                        all_base_features=level_data_cache[0]["features"],
                        peak_indices=top_local_peaks,
                        seconds_per_node=level_data_cache[0]["node_length"],
                        clip_start_video_seconds=clip_data["video_start_sec"]
                    )

                    # Apply IoU-based NMS on the expanded segments to ensure Top-K diversity
                    filtered_predictions = []
                    for pred in expanded_segment_predictions:
                        is_redundant = False
                        for fp in filtered_predictions:
                            # Calculate IoU between the new prediction and already accepted predictions
                            inter = max(0, min(pred[1], fp[1]) - max(pred[0], fp[0]))
                            union = max(pred[1], fp[1]) - min(pred[0], fp[0])
                            iou = inter / union if union > 0 else 0
                            
                            # If the segment overlaps significantly with a higher-ranked one, discard it
                            if iou > 0.65:
                                is_redundant = True
                                break
                        
                        if not is_redundant:
                            filtered_predictions.append(pred)
                        
                        if len(filtered_predictions) == 5:
                            break

                    expanded_segment_predictions = filtered_predictions

                    # Ensure exactly 5 predictions for test submissions
                    while len(expanded_segment_predictions) < 5:
                        if len(expanded_segment_predictions) > 0:
                            expanded_segment_predictions.append(expanded_segment_predictions[-1])
                        else:
                            expanded_segment_predictions.append([0.0, 1.0])
                    expanded_segment_predictions = expanded_segment_predictions[:5]

                    # For test set, clip_uid and annotation_uid must both be the video_uid
                    c_uid = video_data["video_uid"] if args.split == "test" else clip_data["clip_uid"]
                    a_uid = video_data["video_uid"] if args.split == "test" else annotation_data["annotation_uid"]

                    all_predictions.append({
                        "clip_uid": c_uid,
                        "annotation_uid": a_uid,
                        "query_idx": query_idx,
                        "predicted_times": expanded_segment_predictions,
                    })

    if args.split == "test":
        print("\n--- Generating Submission JSON for Test Split ---")
        submission = {
            "version": "1.0",
            "challenge": "ego4d_goalstep_challenge",
            "results": all_predictions
        }
        output_file = f"test_submission_{args.features}.json"
        with open(output_file, "w") as f:
            json.dump(submission, f, indent=4)
        print(f"Submission saved to {output_file}")
    else:
        print("\n--- Standard Step Grounding Metrics (mIoU) ---")
        evaluation_thresholds = [0.3, 0.5, 0.01]
        top_k_metrics = [1, 3, 5]
        metrics_results, mean_iou = evaluate_nlq_performance(all_predictions, dataset_annotations, evaluation_thresholds, top_k_metrics, per_instance=False)
        print(display_results(metrics_results, mean_iou, evaluation_thresholds, top_k_metrics))
        return metrics_results, mean_iou


def main(arg):
    seed_everything(arg.seed)

    # Load visual and text encoders 
    visual_encoder, text_encoder = build_encoders(arg.features, arg.ckpt)

    # Load validation set annotations
    with open(f"ego4d_goalstep/annotations/{arg.split}.json", "r", encoding="utf-8") as file:
        dataset_annotations = json.load(file)

    print(f"Starting Step Grounding Evaluation on Ego4D ({arg.split} split)...")
    evaluate_grounding(arg, dataset_annotations, visual_encoder, text_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Step Grounding Evaluation (Blind Inference).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--features", type=str, choices=["omnivore_video_swinl", "egovlp", "LaViLa-L"], required=True, help="Type of pre-extracted video features to use.")
    parser.add_argument("--ckpt", type=str, help="Optional path to a checkpoint for the visual (and/or text) encoder.")
    parser.add_argument("--n-clusters", type=int, default=16, help="Number of clusters to use if no assignments are provided.")
    parser.add_argument("--threshold", type=int, default=2, help="Base compression factor.")
    parser.add_argument("--target-level", type=int, default=2, help="Hierarchical abstraction target level to use.")

    main(parser.parse_args())
