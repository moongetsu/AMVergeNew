import os
import time
from typing import Any
import numpy as np

from utils.scene_utils import run_stage, run_ffmpeg_segment, collect_scenes, generate_thumbnails
from utils.video_utils import merge_short_scenes, emit_progress

def run_hybrid_split(video_path: str, output_dir: str, log_fn, method="hybrid") -> list[dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    total_start = time.perf_counter()
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    # Stage 1: Detection
    cut_points = run_stage(
        10,
        f"Running {method} detection...",
        lambda: _get_consensus_cuts(video_path, method, log_fn),
        log_fn
    )

    if not cut_points:
        log_fn("No cuts detected. Returning single scene.")
        cut_points = []

    # Finalize cut points
    cut_points = sorted(list(set(cut_points)))
    cut_points = merge_short_scenes([0.0] + cut_points, min_duration=0.5)[1:]

    output_pattern = os.path.join(output_dir, f"{file_name}_%04d.mp4")

    run_stage(
        50,
        f"Cutting {len(cut_points)} scenes...",
        lambda: run_ffmpeg_segment(video_path, output_pattern, cut_points, log_fn),
        log_fn
    )

    final_scenes = run_stage(
        75,
        "Building scenes...",
        lambda: collect_scenes(output_dir, file_name, cut_points),
        log_fn
    )

    run_stage(
        90,
        "Generating thumbnails...",
        lambda: generate_thumbnails(output_dir, final_scenes, file_name, log_fn),
        log_fn
    )

    emit_progress(100, "Done")
    total_end = time.perf_counter()
    log_fn(f"TIMING|total_end_to_end|seconds={total_end - total_start:.3f}")

    return final_scenes

def _get_consensus_cuts(video_path, method, log_fn):
    all_detectors_cuts = []
    
    # Try PySceneDetect first (Content/Adaptive)
    try:
        from scenedetect import detect, ContentDetector, AdaptiveDetector
        
        if method in ["content", "hybrid"]:
            log_fn("Running Content detector...")
            scene_list = detect(video_path, ContentDetector())
            cuts = [s[0].get_seconds() for s in scene_list if s[0].get_seconds() > 0]
            all_detectors_cuts.append(cuts)
            log_fn(f"Content cuts: {len(cuts)}")
            
        if method in ["adaptive", "hybrid"]:
            log_fn("Running Adaptive detector...")
            scene_list = detect(video_path, AdaptiveDetector())
            cuts = [s[0].get_seconds() for s in scene_list if s[0].get_seconds() > 0]
            all_detectors_cuts.append(cuts)
            log_fn(f"Adaptive cuts: {len(cuts)}")
            
    except ImportError:
        log_fn("scenedetect not installed. Skipping pixel-based detection.")
        if method != "hybrid": return []

    # Try TransNet V2
    if method in ["transnetv2", "hybrid"]:
        try:
            from methods.transnet_split import _run_transnet_inference
            log_fn("Running TransNet V2...")
            # We need a way to get raw timestamps from transnet_split
            # For now let's assume we can import its helper or we copy logic
            # To keep it simple, let's just use the existing transnet logic
            from methods import trim_scenes_transnetv2
            # But we want the raw cut points... 
            # I'll just use a simplified version of TransNet inference here
            tn_cuts = _get_transnet_raw_cuts(video_path, log_fn)
            all_detectors_cuts.append(tn_cuts)
            log_fn(f"TransNet cuts: {len(tn_cuts)}")
        except Exception as e:
            log_fn(f"TransNet error: {e}")

    if not all_detectors_cuts:
        return []

    if method != "hybrid":
        return all_detectors_cuts[0]

    # Hybrid Consensus Logic (Voting)
    # If 2 out of 3 detectors agree within a small window (e.g. 0.2s), we keep the cut.
    # We take the average timestamp of the agreeing detectors.
    
    log_fn("Computing consensus (Hybrid)...")
    consensus_cuts = []
    threshold = 0.25 # seconds
    
    # Flatten all points for checking
    all_points = sorted([p for sublist in all_detectors_cuts for p in sublist])
    if not all_points: return []
    
    # Simple grouping
    groups = []
    if all_points:
        current_group = [all_points[0]]
        for p in all_points[1:]:
            if p - current_group[-1] <= threshold:
                current_group.append(p)
            else:
                groups.append(current_group)
                current_group = [p]
        groups.append(current_group)
    
    # A cut is valid if it appears in at least 2 different detector results
    # or if we only have 1 detector available (fallback)
    min_votes = min(2, len(all_detectors_cuts))
    
    for group in groups:
        # Count how many unique detector lists contributed to this group
        contributing_detectors = 0
        for detector_cuts in all_detectors_cuts:
            if any(any(abs(p - dc) <= threshold for dc in detector_cuts) for p in group):
                contributing_detectors += 1
        
        if contributing_detectors >= min_votes:
            consensus_cuts.append(sum(group) / len(group))
            
    log_fn(f"Consensus reached: {len(consensus_cuts)} cuts.")
    return consensus_cuts

def _get_transnet_raw_cuts(video_path, log_fn):
    # This is a bit redundant but ensures we get raw timestamps
    try:
        import torch
        from transnetv2_pytorch import TransNetV2
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            device_str = "cuda"
            device = torch.device("cuda")
        else:
            device_str = "cpu"
            device = torch.device("cpu")
            
        # FIX: Ensure device is explicitly passed during initialization
        model = TransNetV2(device=device_str)
        model = model.to(device)
        model.eval()
            
        log_fn(f"Running TransNet raw inference on {device}...")
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
        
        # FIX: Convert tensor to numpy before calling library method
        if torch.is_tensor(single_frame_predictions):
            single_frame_predictions = single_frame_predictions.cpu().numpy()
            
        scenes = model.predictions_to_scenes(single_frame_predictions)
        
        # Convert frame indices to seconds (approximate, needs FPS)
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0: fps = 23.976
        
        return [s[0] / fps for s in scenes if s[0] > 0]
    except Exception as e:
        log_fn(f"TransNet raw error: {e}")
        return []
