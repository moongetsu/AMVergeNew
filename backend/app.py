import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import av
from PIL import Image

from methods import trim_scenes_at_keyframes, trim_scenes_transnetv2, trim_scenes_omnishotcut, run_hybrid_split

# Running commands like ffmpeg can open a command window on Windows.
# This prevents that when the backend is launched from the app.
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# sys.frozen is an attribute added by PyInstaller when running as an executable.
IS_EXECUTABLE = getattr(sys, "frozen", False)

if IS_EXECUTABLE:
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)


def get_log_dir() -> str:
    # In installed builds, the sidecar exe often lives under a read-only
    # install/resources directory. Always log to a user-writable location.
    base = (
        os.getenv("LOCALAPPDATA")
        or os.getenv("APPDATA")
        or tempfile.gettempdir()
    )

    return os.path.join(base, "AMVerge")


def ensure_log_dir() -> str:
    log_dir = get_log_dir()

    try:
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    except Exception:
        # Last-ditch fallback.
        return tempfile.gettempdir()


DEBUG_LOG_DIR = ensure_log_dir()
DEBUG_LOG = os.path.join(DEBUG_LOG_DIR, "backend_debug.txt")


def log(message: str) -> None:
    text = str(message)

    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass

    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as file:
            file.write(text + "\n")
    except Exception:
        pass




def main() -> int:
    # Protect stdout from being polluted by libraries or accidental prints.
    # We redirect everything to stderr, and only print the final JSON to the real stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        if len(sys.argv) > 1 and sys.argv[1] == "gpu_info":
            from utils.gpu_info import get_gpu_info
            print(json.dumps(get_gpu_info()), file=real_stdout)
            real_stdout.flush()
            return 0
        
        if len(sys.argv) > 1 and sys.argv[1] == "install_gpu":
            from utils.gpu_info import install_cuda
            print(json.dumps(install_cuda()), file=real_stdout)
            real_stdout.flush()
            return 0

        input_file = sys.argv[1]
        output_dir = sys.argv[2]
        method = sys.argv[3] if len(sys.argv) > 3 else "amverge"
        threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4

        if method == "transnetv2":
            scenes = trim_scenes_transnetv2(input_file, output_dir, log, threshold=threshold)
        elif method == "omnishotcut":
            scenes = trim_scenes_omnishotcut(input_file, output_dir, log, threshold=threshold)
        elif method == "hybrid":
            scenes = run_hybrid_split(input_file, output_dir, log, threshold=threshold)
        else:
            # 'amverge' or default falls back to keyframes
            scenes = trim_scenes_at_keyframes(input_file, output_dir, log)

        # Final JSON response to the real stdout.
        print(json.dumps(scenes), file=real_stdout)
        real_stdout.flush()

        return 0

    except Exception as error:
        import traceback

        log(f"FATAL ERROR: {error}")
        log(traceback.format_exc())

        # Always return valid JSON so Rust/React do not crash while parsing.
        print(json.dumps([]))
        print(f"debug_log_dir: {DEBUG_LOG_DIR}", file=sys.stderr)
        sys.stdout.flush()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())