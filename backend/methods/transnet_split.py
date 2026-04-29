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
            # Silence deterministic warnings by setting CuBLAS config
            import os
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = True
            device_str = "cuda"
            device = torch.device("cuda")
            log_fn("CUDA available! Moving TransNetV2 to GPU...")
        else:
            device_str = "cpu"
            device = torch.device("cpu")
            log_fn("CUDA not found. Falling back to CPU...")

        # FIX: The transnetv2_pytorch package requires device to be passed in constructor
        # otherwise internal buffers/weights may remain on CPU despite .to(device)
        model = TransNetV2(device=device_str)
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
        
        # We implement a custom streaming batch processor to avoid loading the whole video into RAM
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0: fps = 23.976
        
        log_fn(f"Video Info: {width}x{height} @ {fps}fps, {total_frames} frames total.")
        
        predictions = []
        batch_size = 100 # Standard sequence length for TransNetV2
        
        # Pre-allocate a batch buffer for speed
        batch_frames = np.zeros((batch_size, 27, 48, 3), dtype=np.uint8)
        
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process the remaining frames in the last partial batch
                remainder = current_frame % batch_size
                if remainder > 0:
                    # Unsqueeze to add the batch dimension: [1, remainder, 27, 48, 3]
                    batch_tensor = torch.from_numpy(batch_frames[:remainder]).to(device).unsqueeze(0)
                    with torch.no_grad():
                        # model(batch_tensor) returns (logits, dict)
                        logits, _ = model(batch_tensor)
                        probs = torch.sigmoid(logits)
                        # We flatten to (remainder,) and append
                        predictions.append(probs.cpu().numpy().reshape(-1))
                break
                
            # Resize frame to model input size (27x48)
            # cv2.resize takes (width, height)
            resized = cv2.resize(frame, (48, 27))
            # Convert BGR to RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            batch_frames[current_frame % batch_size] = resized
            current_frame += 1
            
            if current_frame % batch_size == 0:
                # Convert batch to tensor and move to GPU
                # We MUST unsqueeze to make it 5D: [Batch=1, Time=100, H=27, W=48, C=3]
                batch_tensor = torch.from_numpy(batch_frames).to(device).unsqueeze(0)
                
                with torch.no_grad():
                    # Run inference on batch
                    logits, _ = model(batch_tensor)
                    # Apply sigmoid to get probabilities (0.0 to 1.0)
                    probs = torch.sigmoid(logits)
                    predictions.append(probs.cpu().numpy().reshape(-1))
                
                if current_frame % 500 == 0:
                    progress_val = 10 + int(30 * (current_frame / max(1, total_frames)))
                    emit_progress(progress_val, f"Analyzing frames: {current_frame}/{total_frames}")

        cap.release()
        
        if not predictions:
            log_fn("No frames processed or no predictions made.")
            return []
            
        # Concatenate all batch predictions into a single 1D array of probabilities
        full_predictions = np.concatenate(predictions, axis=0)
        # Note: TransNetV2 usually returns (batch, 1) or (batch, 2)
        # We flatten it to 1D array of probabilities
        if len(full_predictions.shape) > 1:
            full_predictions = full_predictions.flatten()

        scenes = model.predictions_to_scenes(full_predictions)
        
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
