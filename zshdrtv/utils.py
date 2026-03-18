import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_opencv_log_level(level_name: str) -> None:
    level_name = level_name.upper()
    candidates = [
        getattr(cv2, f"LOG_LEVEL_{level_name}", None),
        getattr(getattr(cv2, "utils", object()), "logging", object()).__dict__.get(f"LOG_LEVEL_{level_name}"),
    ]
    for level in candidates:
        if level is None:
            continue
        try:
            cv2.setLogLevel(level)
            return
        except Exception:
            pass
        try:
            cv2.utils.logging.setLogLevel(level)
            return
        except Exception:
            pass


def configure_runtime(config: Dict, device: torch.device) -> None:
    runtime_cfg = config.get("performance", {})
    if device.type == "cuda":
        allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
        cudnn_benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        try:
            torch.set_float32_matmul_precision(str(runtime_cfg.get("float32_matmul_precision", "high")))
        except Exception:
            pass
    if runtime_cfg.get("suppress_opencv_warnings", True):
        _set_opencv_log_level(str(runtime_cfg.get("opencv_log_level", "ERROR")))


def read_hdr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32)


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def read_ir(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
    else:
        scale = float(image.max()) if image.size else 1.0
    scale = max(scale, 1.0)
    return image.astype(np.float32) / scale


def map_range(image: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    imin = float(image.min())
    imax = float(image.max())
    if math.isclose(imin, imax):
        return np.full_like(image, low, dtype=np.float32)
    return np.interp(image, [imin, imax], [low, high]).astype(np.float32)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().float().clamp_(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    return tensor.permute(1, 2, 0).numpy()


def save_hdr(path_without_suffix: str, image: np.ndarray, gamma: float = 2.2) -> None:
    image = np.asarray(image, dtype=np.float32)
    cv2.imwrite(path_without_suffix + ".hdr", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    tonemapped = np.clip(np.power(np.clip(image, 0.0, None), 1.0 / gamma), 0.0, 1.0)
    png = (tonemapped * 255.0).round().astype(np.uint8)
    cv2.imwrite(path_without_suffix + ".png", cv2.cvtColor(png, cv2.COLOR_RGB2BGR))


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return 20.0 * math.log10(1.0 / math.sqrt(max(mse, eps)))


def mu_tonemap(image: torch.Tensor, mu: float = 5000.0) -> torch.Tensor:
    return torch.log1p(mu * image.clamp_min(0.0)) / math.log1p(mu)


def pu_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_tm = mu_tonemap(pred)
    target_tm = mu_tonemap(target)
    mse = torch.mean((pred_tm - target_tm) ** 2).item()
    return 20.0 * math.log10(1.0 / math.sqrt(max(mse, eps)))


def latest_checkpoint(directory: str) -> Optional[Path]:
    candidates = sorted(Path(directory).glob("*.pth"))
    return candidates[-1] if candidates else None


def move_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def split_manifold_parameters(module: torch.nn.Module):
    manifold_params = []
    euclidean_params = []
    seen = set()
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if getattr(parameter, "manifold", None) is not None:
            manifold_params.append(parameter)
        else:
            euclidean_params.append(parameter)
    return manifold_params, euclidean_params


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
