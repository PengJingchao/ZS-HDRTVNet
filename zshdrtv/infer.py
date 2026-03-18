from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
import torch.nn.functional as F

from .data import AlignedRGBTHDRDataset
from .model import ZSHDRTVNet
from .utils import configure_runtime, ensure_dir, load_yaml, move_to_device, read_ir, read_rgb, save_hdr, tensor_to_numpy


def _optional_int(value):
    if value is None:
        return None
    return int(value)


def load_model(
    checkpoint_path: str,
    pretrained_backbone: bool,
    device: torch.device,
    use_temporal_branch: bool = True,
    fusion_variant: str = "dual",
    backbone_type: str = "resnet",
    model_config: Optional[Dict] = None,
) -> ZSHDRTVNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ZSHDRTVNet(
        pretrained_backbone=pretrained_backbone,
        use_temporal_branch=use_temporal_branch,
        fusion_variant=fusion_variant,
        backbone_type=backbone_type,
        model_config=model_config or {},
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


def _resolve_amp_dtype(config: Dict, device: torch.device):
    if device.type != "cuda":
        return None
    amp_mode = str(config.get("performance", {}).get("mixed_precision", "bf16")).lower()
    if amp_mode in {"off", "false", "none", "fp32"}:
        return None
    if amp_mode in {"bf16", "bfloat16", "auto"}:
        return torch.bfloat16
    if amp_mode in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported mixed precision mode: {amp_mode}")


def _autocast_context(config: Dict, device: torch.device):
    amp_dtype = _resolve_amp_dtype(config, device)
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def infer_image_set(config_path: str) -> None:
    config = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(config, device)
    model = load_model(
        config["checkpoint"],
        config["model"].get("pretrained_backbone", True),
        device,
        bool(config["model"].get("use_temporal_branch", True)),
        str(config["model"].get("fusion_variant", "dual")),
        str(config["model"].get("backbone_type", "resnet")),
        config.get("model", {}),
    )
    dataset_cfg = config["datasets"]["image"]
    dataset = AlignedRGBTHDRDataset(
        hdr_dir=dataset_cfg["hdr_dir"],
        rgb_dir=dataset_cfg["rgb_dir"],
        ir_dir=dataset_cfg["ir_dir"],
        index_path=dataset_cfg["test_index"],
        image_size=_optional_int(dataset_cfg.get("image_size", 256)),
        crop_size=_optional_int(dataset_cfg.get("crop_size")),
        augment=False,
        augment_pipeline=str(dataset_cfg.get("augment_pipeline", "basic")),
    )
    output_dir = ensure_dir(config["output"]["result_dir"])
    with torch.inference_mode():
        for item in dataset:
            batch = move_to_device({"rgbt": item["rgbt"].unsqueeze(0)}, device)
            with _autocast_context(config, device):
                prediction, _, _ = model(batch["rgbt"], return_fusion_pairs=False)
            save_hdr(str(output_dir / item["sample_id"]), tensor_to_numpy(prediction))
            print(f"saved {item['sample_id']}")


def _pair_frames(rgb_dir: str, ir_dir: str) -> List[Tuple[Path, Path]]:
    rgb_paths = sorted(Path(rgb_dir).glob("*.JPG")) + sorted(Path(rgb_dir).glob("*.jpg")) + sorted(Path(rgb_dir).glob("*.png"))
    pairs = []
    for rgb_path in rgb_paths:
        ir_name = rgb_path.name.replace("RGB", "T").replace(".JPG", ".tiff").replace(".jpg", ".tiff").replace(".png", ".tiff")
        ir_path = Path(ir_dir) / ir_name
        if ir_path.exists():
            pairs.append((rgb_path, ir_path))
    return pairs


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> Tuple[torch.Tensor, Tuple[int, int]]:
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (height, width)


def _crop_to_size(tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_hw
    return tensor[:, :, :target_h, :target_w]


def infer_video_sequence(config_path: str) -> None:
    config = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(config, device)
    model = load_model(
        config["checkpoint"],
        config["model"].get("pretrained_backbone", True),
        device,
        bool(config["model"].get("use_temporal_branch", True)),
        str(config["model"].get("fusion_variant", "dual")),
        str(config["model"].get("backbone_type", "resnet")),
        config.get("model", {}),
    )
    pairs = _pair_frames(config["video"]["rgb_dir"], config["video"]["ir_dir"])
    output_dir = ensure_dir(config["output"]["result_dir"])
    pad_multiple = int(config["video"].get("pad_to_multiple", 32))
    state = None
    with torch.inference_mode():
        for rgb_path, ir_path in pairs:
            rgb_np = read_rgb(str(rgb_path))
            ir_np = read_ir(str(ir_path))
            if ir_np.shape[:2] != rgb_np.shape[:2]:
                ir_np = cv2.resize(ir_np, (rgb_np.shape[1], rgb_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0)
            ir = torch.from_numpy(ir_np).unsqueeze(0).unsqueeze(0)
            rgbt = torch.cat((rgb, ir), dim=1).to(device)
            rgbt, original_hw = _pad_to_multiple(rgbt, multiple=pad_multiple)
            if state is not None and state[0].shape[-2:] != rgbt.shape[-2:]:
                state = None
            with _autocast_context(config, device):
                prediction, state, _ = model(rgbt, state, return_fusion_pairs=False)
            prediction = _crop_to_size(prediction, original_hw)
            if state is not None:
                state = (state[0].detach(), state[1].detach())
            save_hdr(str(output_dir / rgb_path.stem), tensor_to_numpy(prediction))
            print(f"saved {rgb_path.stem}")
