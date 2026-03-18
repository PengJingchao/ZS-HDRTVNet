from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .flow import warp_with_flow


def rgb2lab(rgb: torch.Tensor) -> torch.Tensor:
    cov_rgb2lms = torch.tensor(
        [[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]],
        device=rgb.device,
        dtype=rgb.dtype,
    )
    cov_lms2lab = torch.tensor(
        [[0.577350, 0.577350, 0.577350], [0.408248, 0.408248, -0.816497], [0.707107, -0.707107, 0.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    )
    lms = torch.einsum("bchw,dc->bdhw", rgb.clamp_min(1e-6), cov_rgb2lms)
    lab = torch.einsum("bchw,dc->bdhw", torch.log10(lms), cov_lms2lab)
    return torch.nan_to_num(lab, nan=0.0, posinf=0.0, neginf=0.0)


def _eotf_pq(rgb: torch.Tensor) -> torch.Tensor:
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    rgb = rgb.clamp(1e-10, 1.0)
    x1 = rgb.pow(1.0 / m2)
    x2 = (x1 - c1).clamp_min(0.0)
    x3 = c2 - c3 * x1
    return (x2 / x3).pow(1.0 / m1) * 10000.0


def _eotf_pq_inverse(rgb: torch.Tensor) -> torch.Tensor:
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    rgb = (rgb / 10000.0).clamp(1e-10, 1.0)
    x1 = c1 + c2 * rgb.pow(m1)
    x2 = 1.0 + c3 * rgb.pow(m1)
    return (x1 / x2).pow(m2)


def hdr_to_ictcp(rgb: torch.Tensor) -> torch.Tensor:
    linear_rgb = _eotf_pq(rgb)
    lr, lg, lb = torch.chunk(linear_rgb, chunks=3, dim=1)
    l = (1688.0 * lr + 2146.0 * lg + 262.0 * lb) / 4096.0
    m = (683.0 * lr + 2951.0 * lg + 462.0 * lb) / 4096.0
    s = (99.0 * lr + 309.0 * lg + 3688.0 * lb) / 4096.0
    lms = torch.cat([l, m, s], dim=1)
    pq_lms = _eotf_pq_inverse(lms)
    el, em, es = torch.chunk(pq_lms, chunks=3, dim=1)
    i = (2048.0 * el + 2048.0 * em) / 4096.0
    t = (6610.0 * el - 13613.0 * em + 7003.0 * es) / 4096.0
    p = (17933.0 * el - 17390.0 * em - 543.0 * es) / 4096.0
    return torch.cat([i, t, p], dim=1)


def gradient_smooth_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = torch.abs(prediction[:, :, :, :-1] - prediction[:, :, :, 1:])
    pred_dy = torch.abs(prediction[:, :, :-1, :] - prediction[:, :, 1:, :])
    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    smoothness = 0.5 * (
        F.smooth_l1_loss(pred_dx, target_dx) + F.smooth_l1_loss(pred_dy, target_dy)
    )
    return F.smooth_l1_loss(prediction, target) + 0.5 * smoothness


class ReconstructionLoss(nn.Module):
    def __init__(self, mode: str = "lab_l1"):
        super().__init__()
        self.mode = mode.lower()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "l1":
            return F.l1_loss(prediction, target)
        if self.mode == "l2":
            return F.mse_loss(prediction, target)
        if self.mode == "lab_l1":
            return F.smooth_l1_loss(rgb2lab(prediction), rgb2lab(target))
        if self.mode in {"gradient_smooth_l1", "grad_smooth_l1"}:
            return gradient_smooth_l1_loss(prediction, target)
        if self.mode in {"itp_l1", "ictcp_l1"}:
            return F.l1_loss(hdr_to_ictcp(prediction), hdr_to_ictcp(target))
        raise ValueError(f"Unsupported reconstruction loss: {self.mode}")


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList(
            [
                nn.Sequential(*features[:4]),
                nn.Sequential(*features[4:9]),
                nn.Sequential(*features[9:16]),
                nn.Sequential(*features[16:23]),
            ]
        )
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = prediction.new_tensor(0.0)
        weights = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0]
        x = prediction
        y = target
        for weight, module in zip(weights, self.slices):
            x = module(x)
            y = module(y)
            loss = loss + weight * F.l1_loss(x, y)
        return loss


class FusionAlignmentLoss(nn.Module):
    def __init__(self, mode: str = "kl", eps: float = 1e-8):
        super().__init__()
        self.mode = mode.lower()
        self.eps = eps

    def forward(self, fusion_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not fusion_pairs:
            raise ValueError("fusion_pairs must not be empty")
        total = fusion_pairs[0][0].new_tensor(0.0)
        for rgb_dist, ir_dist in fusion_pairs:
            rgb_dist = rgb_dist.clamp_min(self.eps)
            ir_dist = ir_dist.clamp_min(self.eps)
            if self.mode == "kl":
                total = total + torch.mean(torch.sum(rgb_dist * (rgb_dist.log() - ir_dist.log()), dim=1))
            elif self.mode == "mse":
                total = total + F.mse_loss(rgb_dist, ir_dist)
            elif self.mode in {"cosine", "cos"}:
                total = total + (1.0 - F.cosine_similarity(rgb_dist, ir_dist, dim=1, eps=self.eps).mean())
            else:
                raise ValueError(f"Unsupported fusion alignment loss: {self.mode}")
        return total / len(fusion_pairs)


def temporal_consistency_loss(
    outputs: List[torch.Tensor],
    sdr_frames: List[torch.Tensor],
    flow_estimator: nn.Module,
    short_weight: float = 100.0,
    long_weight: float = 100.0,
    alpha: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(outputs) <= 1:
        zero = outputs[0].new_tensor(0.0)
        return zero, zero

    short_term = outputs[0].new_tensor(0.0)
    long_term = outputs[0].new_tensor(0.0)

    for index in range(1, len(outputs)):
        flow = flow_estimator(sdr_frames[index], sdr_frames[index - 1])
        warped_sdr = warp_with_flow(sdr_frames[index - 1], flow)
        warped_hdr = warp_with_flow(outputs[index - 1], flow)
        noc_mask = torch.exp(-alpha * ((sdr_frames[index] - warped_sdr) ** 2).sum(dim=1, keepdim=True))
        short_term = short_term + F.l1_loss(outputs[index] * noc_mask, warped_hdr * noc_mask)

    anchor_output = outputs[0].detach()
    anchor_sdr = sdr_frames[0].detach()
    for index in range(2, len(outputs)):
        flow = flow_estimator(sdr_frames[index], anchor_sdr)
        warped_sdr = warp_with_flow(anchor_sdr, flow)
        warped_hdr = warp_with_flow(anchor_output, flow)
        noc_mask = torch.exp(-alpha * ((sdr_frames[index] - warped_sdr) ** 2).sum(dim=1, keepdim=True))
        long_term = long_term + F.l1_loss(outputs[index] * noc_mask, warped_hdr * noc_mask)

    return short_weight * short_term, long_weight * long_term
