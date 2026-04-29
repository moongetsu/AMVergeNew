import os
import time
from typing import Any
import torch

from utils.scene_utils import run_stage, run_ffmpeg_segment, collect_scenes, generate_thumbnails
from utils.video_utils import emit_progress, merge_short_scenes

def trim_scenes_transnetv2(video_path: str, output_dir: str, log_fn) -> list[dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    total_start = time.perf_counter()
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    def run_inference():
        from transnetv2_pytorch import TransNetV2
        import cv2
        
        # Performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            device = torch.device("cuda")
            log_fn("CUDA available! Moving TransNetV2 to GPU...")
        else:
            device = torch.device("cpu")
            log_fn("CUDA not found. Falling back to CPU...")

        model = TransNetV2()
        model = model.to(device)
        model.eval()
        
        # PROOF: Verify model is actually on GPU
        is_on_gpu = next(model.parameters()).is_cuda
        log_fn(f"DEBUG: Model is_cuda = {is_on_gpu}")
        
        if is_on_gpu:
            log_fn(f"SUCCESS: TransNetV2 is running on {torch.cuda.get_device_name(0)}")
        else:
            log_fn("WARNING: Model failed to move to GPU. Processing will be slow.")

        log_fn(f"Running optimized TransNetV2 inference...")
        # predict_video is generally more optimized for GPU than detect_scenes
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
        
        # FIX: Convert tensor to numpy before calling library method
        if torch.is_tensor(single_frame_predictions):
            single_frame_predictions = single_frame_predictions.cpu().numpy()
            
        scenes = model.predictions_to_scenes(single_frame_predictions)
        
        # Convert frame indices to seconds
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0: fps = 23.976
        
        # Format like the previous detect_scenes output if needed, 
        # or just return raw timestamps for the next step.
        return [{"start_time": s[0] / fps} for s in scenes]

    scenes_data = run_stage(
        10,
        "Running TransNetV2 inference...",
        run_inference,
        log_fn
    )

    log_fn(f"TransNetV2 found {len(scenes_data)} scenes.")
    if not scenes_data:
        log_fn("No scenes found. Returning empty list.")
        return []

    # Extract cut points from the scene start times
    # TransNetV2 returns start_time and end_time as strings or floats
    try:
        cut_points = [float(scene["start_time"]) for scene in scenes_data[1:]]
    except Exception as e:
        log_fn(f"Failed to parse TransNetV2 output: {e}")
        return []

    # Filter out very short scenes to avoid pathological segments
    cut_points = sorted(cut_points)
    cut_points = merge_short_scenes([0.0] + cut_points, min_duration=0.25)[1:]

    output_pattern = os.path.join(output_dir, f"{file_name}_%04d.mp4")

    run_stage(
        50,
        f"Cutting {len(cut_points)} scenes...",
        lambda: run_ffmpeg_segment(video_path, output_pattern, cut_points, log_fn),
        log_fn
    )

    emit_progress(75, "Building scenes...")

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
