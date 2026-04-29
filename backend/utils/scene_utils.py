import os
import subprocess
import tempfile
import sys
import av
from PIL import Image
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.video_utils import emit_progress, get_binary, merge_short_scenes

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
FFMPEG = get_binary("ffmpeg.exe")

def run_stage(percent: int, message: str, fn, log_fn):
    emit_progress(percent, message)
    try:
        return fn()
    except Exception as error:
        log_fn(f"ERROR during '{message}': {error}")
        raise

def format_timestamp(seconds: float) -> str:
    value = f"{float(seconds):.6f}"
    return value.rstrip("0").rstrip(".")

def make_thumbnail(clip_path: str, thumb_path: str, log_fn) -> None:
    thumb_width = 360
    thumb_quality = 80
    try:
        with av.open(clip_path) as container:
            if not container.streams.video:
                log_fn(f"Thumbnail skipped, no video stream: {clip_path}")
                return
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            for frame in container.decode(stream):
                image = frame.to_image()
                new_width = thumb_width
                new_height = max(1, int(new_width * image.height / image.width))
                image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
                image.save(thumb_path, "JPEG", quality=thumb_quality)
                return
            log_fn(f"Thumbnail skipped, no decodable frame: {clip_path}")
    except Exception as error:
        log_fn(f"Thumbnail failed for {clip_path}: {error}")

def generate_thumbnails(output_dir: str, scenes: list[dict[str, Any]], file_name: str, log_fn) -> None:
    total = len(scenes)
    if total == 0:
        return
    progress_step = max(1, total // 25)
    completed = 0

    def build_thumbnail(scene: dict[str, Any]) -> None:
        scene_index = scene["scene_index"]
        clip_path = os.path.join(output_dir, f"{file_name}_{scene_index:04d}.mp4")
        thumb_path = os.path.join(output_dir, f"{file_name}_{scene_index:04d}.jpg")
        if not os.path.exists(clip_path):
            log_fn(f"Thumbnail skipped, clip missing: {clip_path}")
            return
        make_thumbnail(clip_path, thumb_path, log_fn)

    emit_progress(90, f"Generating thumbnails... 0/{total}")
    max_workers = min(4, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(build_thumbnail, scene) for scene in scenes]
        for future in as_completed(futures):
            completed += 1
            try:
                future.result()
            except Exception as error:
                log_fn(f"Thumbnail worker failed: {error}")
            if completed % progress_step == 0 or completed == total:
                emit_progress(90, f"Generating thumbnails... {completed}/{total}")

def run_ffmpeg_segment(video_path: str, output_pattern: str, cut_points: list[float], log_fn) -> None:
    # --- FRAME PERFECT FIX ---
    # We force keyframes at exactly the cut points and re-encode using ultrafast x264.
    # This ensures cuts happen at the exact frame the AI detected, not just at the nearest keyframe.
    cmd = [
        FFMPEG, "-y", "-i", video_path,
        "-force_key_frames", ",".join(format_timestamp(p) for p in cut_points),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "copy",
        "-f", "segment",
        "-segment_times", ",".join(format_timestamp(point) for point in cut_points),
        "-reset_timestamps", "1", output_pattern
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    log_fn(result.stdout)
    log_fn(result.stderr)
    if result.returncode != 0:
        tail = result.stderr[-2000:] if result.stderr else "No stderr output."
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}: {tail}")

def collect_scenes(output_dir: str, file_name: str, cut_points: list[float]) -> list[dict[str, Any]]:
    final_scenes = []
    boundaries = [0.0] + cut_points
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else None
        out_path = os.path.join(output_dir, f"{file_name}_{index:04d}.mp4")
        thumb_path = os.path.join(output_dir, f"{file_name}_{index:04d}.jpg")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            final_scenes.append({
                "scene_index": index,
                "start": start,
                "end": end,
                "path": out_path,
                "thumbnail": thumb_path,
                "original_file": file_name,
            })
    return final_scenes
