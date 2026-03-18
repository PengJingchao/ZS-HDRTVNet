import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mctorch.nn as mnn
import torch
import torch.nn as nn


def _to_2tuple(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class LocalContextExtractor(nn.Module):
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        reduced = max(8, dim // reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, reduced, kernel_size=1, bias=True),
            nn.Conv2d(reduced, reduced, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(reduced, dim, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        batch, channels, _, _ = x.shape
        weights = self.fc(self.pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return x * weights


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        size = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, self.k, self.v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ self.k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(tokens, tokens, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ self.v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        x = self.proj_drop(self.proj(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        size = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        q = self.q(x).reshape(batch_windows, tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3) * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(tokens, tokens, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        x = self.proj_drop(self.proj(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        batch, _, channels = x.shape
        height, width = x_size
        return x.transpose(1, 2).view(batch, channels, height, width)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.lce = LocalContextExtractor(dim)
        mask = self._build_mask(input_resolution) if self.shift_size > 0 else None
        self.register_buffer("attn_mask", mask, persistent=False)

    def _build_mask(self, x_size: Tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        counter = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = counter
                counter += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, _, channels = x.shape
        shortcut = x
        norm_x = self.norm1(x).view(batch, height, width, channels)
        local_context = norm_x.permute(0, 3, 1, 2)

        if self.shift_size > 0:
            shifted = torch.roll(norm_x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = norm_x

        windows = window_partition(shifted, self.window_size).view(-1, self.window_size * self.window_size, channels)
        mask = self.attn_mask if x_size == self.input_resolution else self._build_mask(x_size).to(x.device)
        attended = self.attn(windows, mask=mask if self.shift_size > 0 else None)
        attended = attended.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(attended, self.window_size, height, width)

        if self.shift_size > 0:
            shifted = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.drop_path(shifted.view(batch, height * width, channels))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        local_context = self.lce(local_context).view(batch, channels, height * width).permute(0, 2, 1)
        return x + local_context


class FusionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.cross_attn = CrossAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim * 2, int(dim * mlp_ratio), out_features=dim, drop=drop)
        mask = self._build_mask(input_resolution) if self.shift_size > 0 else None
        self.register_buffer("attn_mask", mask, persistent=False)

    def _build_mask(self, x_size: Tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        counter = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = counter
                counter += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], kv: Dict[str, torch.Tensor]) -> torch.Tensor:
        height, width = x_size
        batch, _, channels = x.shape
        shortcut = x
        norm_x = self.norm1(x).view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted = torch.roll(norm_x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = norm_x

        windows = window_partition(shifted, self.window_size).view(-1, self.window_size * self.window_size, channels)
        mask = self.attn_mask if x_size == self.input_resolution else self._build_mask(x_size).to(x.device)
        self_windows = self.attn(windows, mask=mask if self.shift_size > 0 else None)
        cross_windows = self.cross_attn(windows, kv["k"], kv["v"], mask=mask if self.shift_size > 0 else None)

        self_windows = self_windows.view(-1, self.window_size, self.window_size, channels)
        cross_windows = cross_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_self = window_reverse(self_windows, self.window_size, height, width)
        shifted_cross = window_reverse(cross_windows, self.window_size, height, width)

        if self.shift_size > 0:
            shifted_self = torch.roll(shifted_self, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            shifted_cross = torch.roll(shifted_cross, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        self_tokens = shortcut + self.drop_path(shifted_self.view(batch, height * width, channels))
        cross_tokens = shortcut + self.drop_path(shifted_cross.view(batch, height * width, channels))
        fusion_tokens = torch.cat((self.norm2(self_tokens), self.norm3(cross_tokens)), dim=-1)
        return self_tokens + self.drop_path(self.mlp(fusion_tokens))


class IRTransformerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rates: Sequence[float],
    ):
        super().__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                SelfAttentionBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=drop_path_rates[index],
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.patch_embed = PatchEmbed(dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(dim)
        self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = x
        for block in self.blocks:
            x = block(x, x_size)
        kv = {"k": self.blocks[-1].attn.k, "v": self.blocks[-1].attn.v}
        feature = self.patch_unembed(x, x_size)
        feature = self.dilated_conv(feature)
        x = self.patch_embed(feature) + residual
        return x, kv


class RGBTransformerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rates: Sequence[float],
    ):
        super().__init__()
        self.fusion = FusionBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rates[0],
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=drop_path_rates[index],
                )
                for index in range(1, depth)
            ]
        )
        self.patch_embed = PatchEmbed(dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(dim)
        self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], kv: Dict[str, torch.Tensor]) -> torch.Tensor:
        residual = x
        x = self.fusion(x, x_size, kv)
        for block in self.blocks:
            x = block(x, x_size)
        feature = self.patch_unembed(x, x_size)
        feature = self.dilated_conv(feature)
        return self.patch_embed(feature) + residual


class TransformerIRBranch(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        embed_dim: int = 60,
        depths: Sequence[int] = (2, 1, 1),
        num_heads: Sequence[int] = (6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_f = nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_first = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.patch_embed = PatchEmbed(embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        resolution = (img_size, img_size)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.layers = nn.ModuleList()
        for depth, heads in zip(depths, num_heads):
            stage = IRTransformerStage(
                dim=embed_dim,
                input_resolution=resolution,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rates=dpr[offset : offset + depth],
            )
            self.layers.append(stage)
            offset += depth

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        base_feature = self.conv_f(x)
        stem_feature = self.conv_first(base_feature)
        x_size = stem_feature.shape[-2:]
        tokens = self.patch_embed(stem_feature)
        stage_features = []
        kvs = []
        for layer in self.layers:
            tokens, kv = layer(tokens, x_size)
            stage_features.append(self.patch_unembed(tokens, x_size))
            kvs.append(kv)
        encoded_feature = self.patch_unembed(self.norm(tokens), x_size)
        return {
            "base_feature": base_feature,
            "stem_feature": stem_feature,
            "stage_features": stage_features,
            "kvs": kvs,
            "encoded_feature": encoded_feature,
        }


class TransformerRGBBranch(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        embed_dim: int = 60,
        depths: Sequence[int] = (6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_f = nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_first = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.patch_embed = PatchEmbed(embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        resolution = (img_size, img_size)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.layers = nn.ModuleList()
        for depth, heads in zip(depths, num_heads):
            stage = RGBTransformerStage(
                dim=embed_dim,
                input_resolution=resolution,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rates=dpr[offset : offset + depth],
            )
            self.layers.append(stage)
            offset += depth

    def forward(self, rgb: torch.Tensor, kvs: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        base_feature = self.conv_f(rgb)
        stem_feature = self.conv_first(base_feature)
        x_size = stem_feature.shape[-2:]
        tokens = self.patch_embed(stem_feature)
        stage_features = []
        for layer, kv in zip(self.layers, kvs):
            tokens = layer(tokens, x_size, kv)
            stage_features.append(self.patch_unembed(tokens, x_size))
        encoded_feature = self.patch_unembed(self.norm(tokens), x_size)
        return {
            "base_feature": base_feature,
            "stem_feature": stem_feature,
            "stage_features": stage_features,
            "encoded_feature": encoded_feature,
        }


class TransformerRGBTBackbone(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        embed_dim: int = 60,
        ir_depths: Sequence[int] = (2, 1, 1),
        ir_num_heads: Sequence[int] = (6, 6, 6),
        hdr_depths: Sequence[int] = (6, 6, 6),
        hdr_num_heads: Sequence[int] = (6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        fusion_variant: str = "dual",
        ir_checkpoint_path: Optional[str] = None,
        freeze_ir_branch: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_variant = fusion_variant.lower()
        self.freeze_ir_branch = freeze_ir_branch
        self.ir_branch = TransformerIRBranch(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=ir_depths,
            num_heads=ir_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.rgb_branch = TransformerRGBBranch(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=hdr_depths,
            num_heads=hdr_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if ir_checkpoint_path:
            self._load_ir_branch_checkpoint(ir_checkpoint_path)
        if self.freeze_ir_branch:
            self.ir_branch.eval()
            for parameter in self.ir_branch.parameters():
                parameter.requires_grad = False

        self.fusion_channels = [embed_dim, embed_dim, embed_dim, embed_dim]
        if self.fusion_variant == "dual":
            self.fusion_modules_rgb = nn.ModuleList([self._make_fusion_projection(embed_dim) for _ in range(4)])
            self.fusion_modules_ir = nn.ModuleList([self._make_fusion_projection(embed_dim) for _ in range(4)])
        elif self.fusion_variant == "shared":
            self.fusion_modules_shared = nn.ModuleList([self._make_fusion_projection(embed_dim) for _ in range(4)])
        elif self.fusion_variant != "none":
            raise ValueError(f"Unsupported fusion variant: {fusion_variant}")

    def _make_fusion_projection(self, channels: int) -> nn.Module:
        return mnn.rConv2d(channels, channels, kernel_size=1, weight_manifold=mnn.Stiefel)

    def _apply_fusion_projection(
        self,
        index: int,
        rgb_feature: torch.Tensor,
        ir_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_variant == "dual":
            return self.fusion_modules_rgb[index](rgb_feature), self.fusion_modules_ir[index](ir_feature)
        if self.fusion_variant == "shared":
            projection = self.fusion_modules_shared[index]
            return projection(rgb_feature), projection(ir_feature)
        return rgb_feature, ir_feature

    def _load_ir_branch_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                remapped[key[len("backbone.") :]] = value
            else:
                remapped[key] = value
        self.ir_branch.load_state_dict(remapped, strict=False)

    def forward(
        self,
        rgbt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        rgb = rgbt[:, :3]
        ir = rgbt[:, 3:4]
        if self.freeze_ir_branch:
            self.ir_branch.eval()
            with torch.no_grad():
                ir_outputs = self.ir_branch(ir)
        else:
            ir_outputs = self.ir_branch(ir)

        rgb_outputs = self.rgb_branch(rgb, ir_outputs["kvs"])
        fusion_pairs = []
        if self.fusion_variant != "none":
            rgb_features = [rgb_outputs["stem_feature"]] + rgb_outputs["stage_features"]
            ir_features = [ir_outputs["stem_feature"]] + ir_outputs["stage_features"]
            for index, (rgb_feature, ir_feature) in enumerate(zip(rgb_features, ir_features)):
                fusion_pairs.append(self._apply_fusion_projection(index, rgb_feature, ir_feature))

        return (
            rgb_outputs["base_feature"],
            rgb_outputs["stem_feature"],
            rgb_outputs["encoded_feature"],
            fusion_pairs,
        )
