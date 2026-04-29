import sys
import json
import subprocess
import os

def check_nvidia_gpu():
    """Check if NVIDIA GPU is present using nvidia-smi"""
    try:
        # On Windows, nvidia-smi is usually in the path, or at C:\Windows\System32\nvidia-smi.exe
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None

def get_gpu_info():
    """Detailed GPU info merge from user-provided method"""
    info = {
        "available": False,
        "name": "No NVIDIA GPU detected",
        "backend": "None",
        "torch_cuda": False,
        "nvidia_smi_found": False,
        "details": ""
    }
    
    # 1. Check nvidia-smi
    smi_output = check_nvidia_gpu()
    if smi_output:
        info["nvidia_smi_found"] = True
        info["details"] = smi_output
        info["name"] = smi_output.split(',')[0] if ',' in smi_output else smi_output

    # 2. Check Torch
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["available"] = True
            info["torch_cuda"] = True
            info["backend"] = "CUDA"
            info["name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        else:
            if info["nvidia_smi_found"]:
                info["backend"] = "CPU (GPU present but not used)"
            else:
                info["backend"] = "CPU Only"
    except ImportError:
        info["torch_version"] = "Not installed"
        info["backend"] = "PyTorch missing"

    return info

def get_cuda_install_command():
    """Logic from check_gpu.py to find the right install command"""
    cuda_version = None
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        version_str = parts[1].strip().split()[0]
                        cuda_version = float(version_str)
                        break
    except:
        pass
    
    if cuda_version:
        if cuda_version >= 12.4:
            return "cu124", "pip install torch torchvision scenedetect opencv-python --index-url https://download.pytorch.org/whl/cu124"
        elif cuda_version >= 12.1:
            return "cu121", "pip install torch torchvision scenedetect opencv-python --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_version >= 11.8:
            return "cu118", "pip install torch torchvision scenedetect opencv-python --index-url https://download.pytorch.org/whl/cu118"
    
    return "cu121", "pip install torch torchvision scenedetect opencv-python --index-url https://download.pytorch.org/whl/cu121"

def install_cuda():
    """Install CUDA PyTorch logic from check_gpu.py"""
    tag, cmd_str = get_cuda_install_command()
    python_exe = sys.executable
    
    try:
        # Uninstall first
        subprocess.run([python_exe, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "scenedetect", "opencv-python"], capture_output=True)
        
        # Install new
        # We split the command string but keep the index-url part together
        # cmd_str is "pip install torch torchvision --index-url https://..."
        parts = cmd_str.split()
        # [pip, install, torch, torchvision, --index-url, https://...]
        actual_args = [python_exe, "-m"] + parts
        
        # We run this and return progress if we were in a thread, but for a simple call:
        process = subprocess.run(actual_args, capture_output=True, text=True)
        
        if process.returncode == 0:
            return {"success": True, "message": f"Successfully installed CUDA PyTorch ({tag})"}
        else:
            return {"success": False, "message": process.stderr}
    except Exception as e:
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    # If called with 'install', do the install
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        print(json.dumps(install_cuda()))
    else:
        print(json.dumps(get_gpu_info()))
