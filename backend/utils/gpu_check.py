import json
import sys

def main():
    try:
        import torch
        is_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if is_available else None
        
        result = {
            "available": is_available,
            "name": device_name,
            "version": torch.version.cuda if is_available else None
        }
        
        print(json.dumps(result))
    except ImportError:
        print(json.dumps({"available": False, "name": None, "error": "PyTorch not installed"}))
    except Exception as e:
        print(json.dumps({"available": False, "name": None, "error": str(e)}))

if __name__ == "__main__":
    main()
