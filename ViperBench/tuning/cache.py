"""Load and save tuning cache."""
import json
import torch
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "results" / "tuning_cache.json"


def get_gpu_arch():
    """Return GPU architecture string for cache key."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    parts = name.replace("NVIDIA ", "").replace(" Generation", "").split()
    return "_".join(parts)


def load_cache():
    """Load tuning cache from JSON. Returns empty dict if not found."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache):
    """Save tuning cache to JSON."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def get_best_config(kernel_name, impl_type):
    """Get best config for a kernel. Returns None if not cached."""
    cache = load_cache()
    arch = get_gpu_arch()
    key = f"{kernel_name}/{impl_type}/{arch}"
    return cache.get(key)
