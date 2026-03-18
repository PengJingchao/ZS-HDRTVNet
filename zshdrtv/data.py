import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AddNoise, RandomAffine, RandomColorDisturbance, RandomErasing, RandomHorizontalFlip, RandomResizedCrop, RandomRotation, Resize
from .utils import map_range, read_hdr, read_ir, read_rgb


def luminance(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]


def apply_crf(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return np.clip(image, 0.0, 1.0) ** (1.0 / gamma)


def clahe_grayscale(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    image_uint8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(image_uint8)
    return enhanced.astype(np.float32) / 255.0


def create_ldr_stack_from_hdr(image: np.ndarray, gamma: float = 2.2) -> List[np.ndarray]:
    lum = luminance(image)
    valid = lum[lum > 0]
    if valid.size == 0:
        return [apply_crf(image, gamma=gamma)]
    min_exposure = math.floor(math.log2(valid.max() + 1e-6) + 1)
    max_exposure = math.ceil(math.log2(valid.min() + 1e-6) + 1)
    t_min = -(min_exposure - 1)
    t_max = -(max_exposure + 1)
    if t_max < t_min:
        t_min = -min_exposure
        t_max = -max_exposure
    if math.isclose(t_min, t_max):
        return [apply_crf(image, gamma=gamma)]
    exposures = 2 ** np.arange(t_min + (t_max - t_min) / 6.0, t_max + (t_max - t_min) / 6.0, (t_max - t_min) / 4.0)
    stack = []
    for exposure in exposures:
        stack.append(apply_crf(image * exposure, gamma=gamma))
    return stack or [apply_crf(image, gamma=gamma)]


def resize_triplet(hdr: np.ndarray, rgb: np.ndarray, ir: np.ndarray, size: int):
    hdr = cv2.resize(hdr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    ir = cv2.resize(ir, (size, size), interpolation=cv2.INTER_LINEAR)
    return hdr, rgb, ir


def random_crop_triplet(hdr: np.ndarray, rgb: np.ndarray, ir: np.ndarray, size: int):
    height, width = hdr.shape[:2]
    if height < size or width < size:
        return resize_triplet(hdr, rgb, ir, size)
    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    hdr = hdr[top : top + size, left : left + size]
    rgb = rgb[top : top + size, left : left + size]
    ir = ir[top : top + size, left : left + size]
    return hdr, rgb, ir


def random_flip_triplet(hdr: np.ndarray, rgb: np.ndarray, ir: np.ndarray):
    if random.random() < 0.5:
        hdr = np.ascontiguousarray(np.flip(hdr, axis=1))
        rgb = np.ascontiguousarray(np.flip(rgb, axis=1))
        ir = np.ascontiguousarray(np.flip(ir, axis=1))
    return hdr, rgb, ir


def random_rotate_triplet(hdr: np.ndarray, rgb: np.ndarray, ir: np.ndarray):
    rotate = random.randint(0, 3)
    if rotate:
        hdr = np.ascontiguousarray(np.rot90(hdr, rotate))
        rgb = np.ascontiguousarray(np.rot90(rgb, rotate))
        ir = np.ascontiguousarray(np.rot90(ir, rotate))
    return hdr, rgb, ir


def to_tensor_rgb(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))


def to_tensor_ir(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32)).unsqueeze(0)


def normalize_sample_id(raw_value: str) -> str:
    sample_id = raw_value.strip()
    if not sample_id:
        return sample_id
    sample_id = Path(sample_id).stem
    if sample_id.upper().startswith("RGB"):
        sample_id = sample_id[3:]
    elif sample_id.upper().startswith("T"):
        sample_id = sample_id[1:]
    return sample_id


class AlignedRGBTHDRDataset(Dataset):
    def __init__(
        self,
        hdr_dir: str,
        rgb_dir: str,
        ir_dir: str,
        index_path: str,
        image_size: Optional[int] = 256,
        crop_size: Optional[int] = None,
        augment: bool = True,
        augment_pipeline: str = "basic",
    ):
        self.hdr_dir = Path(hdr_dir)
        self.rgb_dir = Path(rgb_dir)
        self.ir_dir = Path(ir_dir)
        self.image_size = image_size
        self.crop_size = crop_size
        self.augment = augment
        self.augment_pipeline = augment_pipeline.lower()
        self.ids = [
            normalize_sample_id(line)
            for line in Path(index_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        target_size = crop_size if crop_size is not None else image_size
        if self.augment_pipeline == "reference":
            self.ldr_add_noise_transform = AddNoise(0.3, (0.8, 1.2)) if augment else AddNoise(0.0, (1.0, 1.0))
            self.color_transform = RandomColorDisturbance(probability=0.3 if augment else 0.0)
            if augment and target_size is not None:
                self.reference_transform = [
                    RandomErasing(probability=0.3, mean=[0.0, 0.0, 0.0]),
                    RandomHorizontalFlip(),
                    RandomRotation(10),
                    RandomResizedCrop(target_size, scale=(0.4, 1.0), ratio=(1.0, 1.0)),
                    RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
                ]
            elif target_size is not None:
                self.reference_transform = [Resize([target_size, target_size])]
            else:
                self.reference_transform = []
        else:
            self.ldr_add_noise_transform = None
            self.color_transform = None
            self.reference_transform = []

    def __len__(self) -> int:
        return len(self.ids)

    def _apply_reference_transform(self, image: np.ndarray, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for transform in self.reference_transform:
            image = transform(image)
        return image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_id = self.ids[index]
        hdr = read_hdr(str(self.hdr_dir / f"RGB{sample_id}.hdr"))
        rgb = read_rgb(str(self.rgb_dir / f"RGB{sample_id}.JPG"))
        ir = read_ir(str(self.ir_dir / f"T{sample_id}.tiff"))

        if self.augment_pipeline == "reference":
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            rgb = self.ldr_add_noise_transform(rgb)
            rgb, hdr = self.color_transform(ldr_img=rgb, hdr_img=hdr)
            rgb = rgb.astype(np.float32) / 255.0
            hdr = map_range(hdr)
            seed = np.random.randint(4354356)
            hdr = self._apply_reference_transform(hdr, seed)
            rgb = self._apply_reference_transform(rgb, seed)
            ir = self._apply_reference_transform(ir, seed)
        else:
            hdr = map_range(hdr)
            if self.crop_size is not None:
                hdr, rgb, ir = random_crop_triplet(hdr, rgb, ir, self.crop_size)
            elif self.image_size is not None:
                hdr, rgb, ir = resize_triplet(hdr, rgb, ir, self.image_size)
            if self.augment:
                hdr, rgb, ir = random_flip_triplet(hdr, rgb, ir)
                hdr, rgb, ir = random_rotate_triplet(hdr, rgb, ir)
        return {
            "rgbt": torch.cat((to_tensor_rgb(rgb), to_tensor_ir(ir)), dim=0),
            "hdr": to_tensor_rgb(hdr),
            "rgb": to_tensor_rgb(rgb),
            "ir": to_tensor_ir(ir),
            "sample_id": sample_id,
        }


class HDRVideoSequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        sequence_length: int = 7,
        image_size: int = 256,
        sample_exposure_index: Optional[int] = None,
        pseudo_ir_mode: str = "luminance",
        limit_videos: Optional[int] = None,
        include_videos: Optional[str] = None,
        deterministic: bool = False,
    ):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.sample_exposure_index = sample_exposure_index
        self.pseudo_ir_mode = pseudo_ir_mode
        self.deterministic = deterministic
        allowed_videos = None
        if include_videos is not None:
            allowed_videos = {
                line.strip()
                for line in Path(include_videos).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        self.videos = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir():
                continue
            if allowed_videos is not None and path.name not in allowed_videos:
                continue
            frames = sorted(list(path.glob("*.hdr")) + list(path.glob("*.exr")))
            if len(frames) >= sequence_length:
                self.videos.append(frames)
        if limit_videos is not None:
            self.videos = self.videos[:limit_videos]

    def __len__(self) -> int:
        return len(self.videos)

    def _pseudo_ir(self, hdr: np.ndarray) -> np.ndarray:
        if self.pseudo_ir_mode == "luminance":
            return luminance(apply_crf(hdr))
        if self.pseudo_ir_mode == "clahe":
            return clahe_grayscale(luminance(apply_crf(hdr)))
        raise ValueError(f"Unsupported pseudo IR mode: {self.pseudo_ir_mode}")

    def _load_sequence_window(self, selected: Sequence[Path]) -> Dict[str, torch.Tensor]:
        hdr_list = []
        rgb_list = []
        ir_list = []
        rgbt_list = []
        for frame_path in selected:
            hdr = map_range(read_hdr(str(frame_path)))
            stack = create_ldr_stack_from_hdr(hdr)
            choice = self.sample_exposure_index
            if choice is None:
                choice = len(stack) // 2 if self.deterministic else random.randint(0, len(stack) - 1)
            rgb = stack[min(choice, len(stack) - 1)]
            ir = self._pseudo_ir(hdr)
            hdr, rgb, ir = resize_triplet(hdr, rgb, ir, self.image_size)
            hdr_tensor = to_tensor_rgb(hdr)
            rgb_tensor = to_tensor_rgb(rgb)
            ir_tensor = to_tensor_ir(ir)
            hdr_list.append(hdr_tensor)
            rgb_list.append(rgb_tensor)
            ir_list.append(ir_tensor)
            rgbt_list.append(torch.cat((rgb_tensor, ir_tensor), dim=0))
        return {
            "hdr": torch.stack(hdr_list, dim=0),
            "rgb": torch.stack(rgb_list, dim=0),
            "ir": torch.stack(ir_list, dim=0),
            "rgbt": torch.stack(rgbt_list, dim=0),
            "video_name": selected[0].parent.name,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        frames = self.videos[index]
        max_start = len(frames) - self.sequence_length
        for _ in range(16):
            start = max_start // 2 if self.deterministic else random.randint(0, max_start)
            selected = frames[start : start + self.sequence_length]
            try:
                return self._load_sequence_window(selected)
            except Exception:
                continue

        for start in range(max_start + 1):
            selected = frames[start : start + self.sequence_length]
            try:
                return self._load_sequence_window(selected)
            except Exception:
                continue

        raise RuntimeError(f"No readable {self.sequence_length}-frame window found in video {frames[0].parent}")
