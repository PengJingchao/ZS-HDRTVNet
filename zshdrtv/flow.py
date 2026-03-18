from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def _resize_min(image1: torch.Tensor, image2: torch.Tensor, minimum: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = image1.shape
    scale = max(minimum / float(height), minimum / float(width), 1.0)
    if scale == 1.0:
        return image1, image2
    new_size = (int(round(height * scale)), int(round(width * scale)))
    return (
        F.interpolate(image1, size=new_size, mode="bilinear", align_corners=False),
        F.interpolate(image2, size=new_size, mode="bilinear", align_corners=False),
    )


class RAFTFlowEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = Raft_Small_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_small(weights=self.weights, progress=False).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        original_height, original_width = image1.shape[-2:]
        image1, image2 = _resize_min(image1, image2)
        processed_1, processed_2 = self.transforms(image1, image2)
        flow = self.model(processed_1, processed_2)[-1]
        if flow.shape[-2:] != (original_height, original_width):
            scale_y = original_height / float(flow.shape[-2])
            scale_x = original_width / float(flow.shape[-1])
            flow = F.interpolate(flow, size=(original_height, original_width), mode="bilinear", align_corners=False)
            flow[:, 0] *= scale_x
            flow[:, 1] *= scale_y
        return flow


def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = image.shape
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device),
        torch.linspace(-1.0, 1.0, width, device=image.device),
        indexing="ij",
    )
    base_grid = torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    flow_grid = torch.empty_like(base_grid)
    flow_grid[..., 0] = flow[:, 0] / max((width - 1.0) / 2.0, 1.0)
    flow_grid[..., 1] = flow[:, 1] / max((height - 1.0) / 2.0, 1.0)
    sample_grid = base_grid + flow_grid
    return F.grid_sample(image, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
