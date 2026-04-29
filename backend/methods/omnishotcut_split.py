import os
import sys
import time
import torch
import numpy as np
import cv2
import ffmpeg
from typing import Any

from utils.scene_utils import run_stage, run_ffmpeg_segment, collect_scenes, generate_thumbnails
from utils.video_utils import emit_progress, merge_short_scenes

# Add OmniShotCut to sys.path
OMNI_REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "OmniShotCut")
if OMNI_REPO_PATH not in sys.path:
    sys.path.append(OMNI_REPO_PATH)

def trim_scenes_omnishotcut(video_path: str, output_dir: str, log_fn, threshold: float = 0.4) -> list[dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    total_start = time.perf_counter()
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    def run_inference():
        # Local imports for OmniShotCut architecture
        try:
            from architecture.backbone import build_backbone
            from architecture.transformer import build_transformer
            from architecture.model import OmniShotCut
            from datasets.transforms import Video_Augmentation_Transform
        except ImportError as e:
            log_fn(f"ERROR: Could not import OmniShotCut components: {e}")
            log_fn(f"Make sure OmniShotCut is in {OMNI_REPO_PATH}")
            return []

        # Load Checkpoint
        checkpoint_path = os.path.join(OMNI_REPO_PATH, "checkpoints", "OmniShotCut_ckpt.pth")
        if not os.path.exists(checkpoint_path):
            log_fn(f"ERROR: Checkpoint not found at {checkpoint_path}")
            return []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_fn(f"Loading OmniShotCut to {device}...")
        
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_args = state_dict['args']
        
        backbone = build_backbone(model_args)
        transformer = build_transformer(model_args)
        model = OmniShotCut(
            backbone,
            transformer,
            num_intra_relation_classes=model_args.num_intra_relation_classes,
            num_inter_relation_classes=model_args.num_inter_relation_classes,
            num_frames=model_args.max_process_window_length,
            num_queries=model_args.num_queries,
            aux_loss=model_args.aux_loss,
        )
        model.load_state_dict(state_dict['model'], strict=True)
        model.to(device)
        model.eval()

        video_transform = Video_Augmentation_Transform(set_type="val")
        
        # Get Video Info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0: fps = 23.976
        
        process_width, process_height = model_args.process_width, model_args.process_height
        log_fn(f"OmniShotCut: Extracting frames @ {process_width}x{process_height}...")

        # We use the ffmpeg pipe approach from their inference.py
        # FIX: Force constant FPS filter to handle Variable Frame Rate (VFR) anime correctly
        try:
            video_stream, err = ffmpeg.input(
                video_path
            ).filter(
                'fps', fps=fps
            ).output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{process_width}x{process_height}",
            ).run(
                capture_stdout=True, capture_stderr=True
            )
        except Exception as e:
            log_fn(f"FFmpeg error: {e}")
            return []

        video_np_full = np.frombuffer(video_stream, np.uint8).reshape(-1, process_height, process_width, 3)
        log_fn(f"Video loaded: {len(video_np_full)} frames.")

        # Proper context windowing logic from OmniShotCut's inference.py
        max_window = model_args.max_process_window_length
        # Cap context to 25% of window size to avoid infinite loops (stride must be > 0)
        num_context = min(25, max_window // 4) 
        
        def split_videos_with_context(video, chunk_size, context_frames):
            total_num, H, W, C = video.shape
            # Padding at the very beginning
            black_pad = np.zeros((context_frames, H, W, C), dtype=video.dtype)
            video_padded = np.concatenate([black_pad, video], axis=0)
            
            stride = max(1, chunk_size - 2 * context_frames)
            cur_idx = 0
            chunks = []
            while cur_idx < total_num:
                chunk = video_padded[cur_idx : cur_idx + chunk_size]
                num_pad_at_end = 0
                if len(chunk) < chunk_size:
                    num_pad_at_end = chunk_size - len(chunk)
                    padding = np.zeros((num_pad_at_end, H, W, C), dtype=video.dtype)
                    chunk = np.concatenate([chunk, padding], axis=0)
                
                chunks.append((chunk, num_pad_at_end))
                cur_idx += stride
            return chunks

        def prune_context_results(ranges, window_size, context_frames):
            pruned = []
            for start, end in ranges:
                # Skip shots that are entirely within the left or right context buffers
                if end <= context_frames: continue
                if start >= window_size - context_frames: break
                
                # Align to non-context space
                aligned_start = max(start, context_frames) - context_frames
                aligned_end = min(end, window_size - context_frames) - context_frames
                pruned.append([aligned_start, aligned_end])
            return pruned

        pred_ranges_full = []
        chunks = split_videos_with_context(video_np_full, max_window, num_context)
        
        log_fn(f"OmniShotCut: Analyzing {len(chunks)} chunks (Context: {num_context}, Sensitivity: {round((1-threshold)*100)}%)...")
        
        global_frame_offset = 0
        stride = max(1, max_window - 2 * num_context)
        
        for i, (chunk_np, num_pad) in enumerate(chunks):
            video_tensor = video_transform(chunk_np).unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs = model(video_tensor)
            
            # Extract probabilities
            probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]  
            range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]
            
            query_range_idx = range_probas.argmax(dim=-1)
            
            # Extract raw ranges for this chunk with sensitivity filtering
            chunk_ranges = []
            start_f = 0
            for k in range(len(query_range_idx)):
                end_f = int(query_range_idx[k].detach().cpu())
                if start_f >= end_f: continue
                
                # SENSITIVITY FILTERING:
                # Inter label mapping: 0: new_start, 1: hard_cut, 2: trans_source, 3: transition, 4: sudden_jump
                # If we want more sensitivity (lower threshold), we allow shots that have any cut-like label.
                p_cut = probas_inter[k, 0] + probas_inter[k, 1] + probas_inter[k, 3] + probas_inter[k, 4]
                
                # --- DEBUG LOGGING ---
                if p_cut > 0.05:
                    curr_frame = global_frame_offset + start_f
                    log_fn(f"DEBUG OmniShotCut | Time: ~{curr_frame/fps:.2f}s (Frame {curr_frame}) | Prob: {p_cut:.4f}")
                
                if p_cut > threshold:
                    chunk_ranges.append([start_f, end_f])
                
                start_f = end_f
                if end_f >= max_window - num_pad: break
            
            # Prune context padding
            pruned_ranges = prune_context_results(chunk_ranges, max_window, num_context)
            
            # Map to global timeline
            for start, end in pruned_ranges:
                # Merge logic if the first range of this chunk continues from the last range of previous chunk
                # For simplicity in this cut-based approach, we just add the global offset
                pred_ranges_full.append([global_frame_offset + start, global_frame_offset + end])
            
            global_frame_offset += stride
            
            if i % 1 == 0:
                emit_progress(10 + int(30 * (i / len(chunks))), f"OmniShotCut chunk {i+1}/{len(chunks)}")

        # IMPROVED: Proximity-based cut merging
        # If two chunks detect a cut at slightly different frames (e.g. 100 and 101),
        # we merge them to avoid "double-cutting" long videos.
        raw_cuts = sorted(list(set([r[0] for r in pred_ranges_full if r[0] > 0])))
        if not raw_cuts:
            return []

        unique_cuts = []
        if raw_cuts:
            unique_cuts.append(raw_cuts[0])
            for next_cut in raw_cuts[1:]:
                # If the next cut is within 5 frames of the last one, it's likely the same boundary
                if next_cut - unique_cuts[-1] > 5:
                    unique_cuts.append(next_cut)
                else:
                    # Optional: Take the middle point or keep the first one
                    pass

        return [{"start_time": f / fps} for f in unique_cuts]

    scenes_data = run_stage(
        10,
        "Running OmniShotCut inference...",
        run_inference,
        log_fn
    )

    if not scenes_data:
        log_fn("No scenes found with OmniShotCut. Returning empty list.")
        return []

    cut_points = [float(scene["start_time"]) for scene in scenes_data]
    cut_points = merge_short_scenes([0.0] + cut_points, min_duration=0.25)[1:]

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
    log_fn(f"OmniShotCut found {len(final_scenes)} scenes.")
    return final_scenes
