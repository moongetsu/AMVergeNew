import os
import time
from typing import Any

from utils.scene_utils import run_stage, run_ffmpeg_segment, collect_scenes, generate_thumbnails
from utils.video_utils import generate_keyframes, merge_short_scenes, emit_progress

def trim_scenes_at_keyframes(video_path: str, output_dir: str, log_fn) -> list[dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    total_start = time.perf_counter()
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    keyframes = run_stage(
        10,
        "Extracting keyframes...",
        lambda: generate_keyframes(
            video_path=video_path,
            progress_cb=emit_progress,
            progress_base=10,
            progress_range=30,
            progress_interval_s=1.0,
        ),
        log_fn
    )

    log_fn(f"Keyframes found: {len(keyframes)}")
    if not keyframes:
        log_fn("No keyframes found. Returning empty scene list.")
        return []

    cut_points = sorted(keyframes[1:])
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
