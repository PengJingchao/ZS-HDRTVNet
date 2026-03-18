from typing import Dict, List, Optional, Tuple

import mctorch.nn as mnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet101_Weights

from .transformer_backbone import TransformerRGBTBackbone


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    return mnn.rConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        weight_manifold=mnn.Euclidean,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return mnn.rConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        weight_manifold=mnn.Euclidean,
    )


class ConvLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        padding = kernel_size // 2
        self.gates = mnn.rConv2d(
            input_size + hidden_size,
            4 * hidden_size,
            kernel_size,
            padding=padding,
            weight_manifold=mnn.Euclidean,
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = x.shape
        if prev_state is None:
            zeros = torch.zeros(batch_size, self.hidden_size, height, width, device=x.device, dtype=x.dtype)
            prev_state = (zeros, zeros)

        hidden_prev, cell_prev = prev_state
        gates = self.gates(torch.cat((x, hidden_prev), dim=1))
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, dim=1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = remember_gate * cell_prev + in_gate * cell_gate
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class TwoStreamBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class TwoStreamResNet101(nn.Module):
    def __init__(self, block=Bottleneck, layers=(3, 4, 23, 3), fusion_variant: str = "dual"):
        super().__init__()
        self.inplanes = 64
        self.fusion_variant = fusion_variant.lower()
        self.conv1 = mnn.rConv2d(9, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, weight_manifold=mnn.Euclidean)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fusion_channels = [64, 256, 512, 1024]
        if self.fusion_variant == "dual":
            self.fusion_modules_rgb = nn.ModuleList([self._make_fusion_projection(channels) for channels in self.fusion_channels])
            self.fusion_modules_ir = nn.ModuleList([self._make_fusion_projection(channels) for channels in self.fusion_channels])
        elif self.fusion_variant == "shared":
            self.fusion_modules_shared = nn.ModuleList([self._make_fusion_projection(channels) for channels in self.fusion_channels])
        elif self.fusion_variant == "none":
            pass
        else:
            raise ValueError(f"Unsupported fusion variant: {fusion_variant}")

    def _make_fusion_projection(self, channels: int) -> nn.Module:
        return mnn.rConv2d(channels, channels, kernel_size=1, weight_manifold=mnn.Stiefel)

    def _apply_fusion_projection(self, index: int, rgb_feature: torch.Tensor, ir_feature: torch.Tensor):
        if self.fusion_variant == "dual":
            rgb_projected = self.fusion_modules_rgb[index](rgb_feature)
            ir_projected = self.fusion_modules_ir[index](ir_feature)
        elif self.fusion_variant == "shared":
            projection = self.fusion_modules_shared[index]
            rgb_projected = projection(rgb_feature)
            ir_projected = projection(ir_feature)
        else:
            rgb_projected = rgb_feature
            ir_projected = ir_feature
        return rgb_projected, ir_projected

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [TwoStreamBottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, rgb: torch.Tensor, ir1: torch.Tensor, ir2: torch.Tensor, ir3: torch.Tensor, ir4: torch.Tensor):
        original = rgb
        x = self.relu(self.bn1(self.conv1(rgb)))
        stem = self.maxpool(x)

        rgb1, ir1 = self._apply_fusion_projection(0, stem, ir1)
        f2 = self.layer1(rgb1 + ir1)

        rgb2, ir2 = self._apply_fusion_projection(1, f2, ir2)
        f3 = self.layer2(rgb2 + ir2)

        rgb3, ir3 = self._apply_fusion_projection(2, f3, ir3)
        f4 = self.layer3(rgb3 + ir3)

        rgb4, ir4 = self._apply_fusion_projection(3, f4, ir4)
        f5 = self.layer4(rgb4 + ir4)

        fusion_pairs = [(rgb1, ir1), (rgb2, ir2), (rgb3, ir3), (rgb4, ir4)]
        return x, f2, f3, f4, f5, original[:, :3], fusion_pairs


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, up_in_channels: Optional[int] = None, up_out_channels: Optional[int] = None):
        super().__init__()
        up_in_channels = up_in_channels or in_channels
        up_out_channels = up_out_channels or out_channels
        self.up = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
        self.block = UNetConvBlock(in_channels, out_channels)

    def center_crop(self, layer: torch.Tensor, target_hw) -> torch.Tensor:
        _, _, height, width = layer.shape
        target_h, target_w = target_hw
        diff_y = (height - target_h) // 2
        diff_x = (width - target_w) // 2
        return layer[:, :, diff_y : diff_y + target_h, diff_x : diff_x + target_w]

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.center_crop(skip, x.shape[-2:])
        return self.block(torch.cat((x, skip), dim=1))


class ZSHDRTVNet(nn.Module):
    def __init__(
        self,
        pretrained_backbone: bool = True,
        use_temporal_branch: bool = True,
        fusion_variant: str = "dual",
        backbone_type: str = "resnet",
        model_config: Optional[Dict] = None,
    ):
        super().__init__()
        model_config = model_config or {}
        temporal_scale_init = float(model_config.get("temporal_scale_init", 0.0))
        self.use_temporal_branch = use_temporal_branch
        self.fusion_variant = fusion_variant.lower()
        self.backbone_type = str(model_config.get("backbone_type", backbone_type)).lower()
        self.fusion_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Softmax(dim=1))

        if self.backbone_type == "transformer":
            embed_dim = int(model_config.get("transformer_embed_dim", 60))
            self.transformer_backbone = TransformerRGBTBackbone(
                img_size=int(model_config.get("transformer_img_size", 512)),
                embed_dim=embed_dim,
                ir_depths=tuple(model_config.get("transformer_ir_depths", (2, 1, 1))),
                ir_num_heads=tuple(model_config.get("transformer_ir_num_heads", (6, 6, 6))),
                hdr_depths=tuple(model_config.get("transformer_hdr_depths", (6, 6, 6))),
                hdr_num_heads=tuple(model_config.get("transformer_hdr_num_heads", (6, 6, 6))),
                window_size=int(model_config.get("transformer_window_size", 8)),
                mlp_ratio=float(model_config.get("transformer_mlp_ratio", 4.0)),
                drop_rate=float(model_config.get("transformer_drop_rate", 0.0)),
                attn_drop_rate=float(model_config.get("transformer_attn_drop_rate", 0.0)),
                drop_path_rate=float(model_config.get("transformer_drop_path_rate", 0.1)),
                fusion_variant=self.fusion_variant,
                ir_checkpoint_path=model_config.get("transformer_ir_checkpoint"),
                freeze_ir_branch=bool(model_config.get("transformer_freeze_ir_branch", True)),
            )
            self.temporal = ConvLSTM(input_size=embed_dim, hidden_size=embed_dim, kernel_size=3)
            self.temporal_scale = nn.Parameter(torch.full((1,), temporal_scale_init))
            self.transformer_body_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
            self.transformer_head = nn.Sequential(
                nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
            return

        self.encoder = TwoStreamResNet101(fusion_variant=self.fusion_variant)
        self.temporal = ConvLSTM(input_size=2048, hidden_size=2048, kernel_size=3)
        self.temporal_scale = nn.Parameter(torch.full((1,), temporal_scale_init))
        self.decoder1 = UNetUpBlock(2048, 1024)
        self.decoder2 = UNetUpBlock(1024, 512)
        self.decoder3 = UNetUpBlock(512, 256)
        self.decoder4 = UNetUpBlock(128 + 64, 128, up_in_channels=256, up_out_channels=128)
        self.decoder5 = UNetUpBlock(64 + 3, 64, up_in_channels=128, up_out_channels=64)
        self.head = nn.Sequential(
            mnn.rConv2d(64, 3, kernel_size=1, weight_manifold=mnn.Euclidean),
            nn.Sigmoid(),
        )

        weights = ResNet101_Weights.DEFAULT if pretrained_backbone else None
        self.ir_branch = nn.ModuleList(list(torchvision.models.resnet101(weights=weights).children())[:-3])
        self.fusion_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Softmax(dim=1))
        self._replace_ir_conv2d()

    def _replace_ir_conv2d(self) -> None:
        def find_parent(module: nn.Module, name: str):
            parts = name.split(".")
            parent = module
            for part in parts[:-1]:
                parent = parent._modules[part]
            return parent, parts[-1]

        for name, layer in list(self.ir_branch.named_modules()):
            if not isinstance(layer, nn.Conv2d):
                continue
            parent, child_name = find_parent(self.ir_branch, name)
            parent._modules[child_name] = mnn.rConv2d(
                layer.in_channels,
                layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
                weight_manifold=mnn.Euclidean,
            )

    def forward(
        self,
        rgbt: torch.Tensor,
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_fusion_pairs: bool = True,
    ):
        if rgbt.shape[1] != 4:
            raise ValueError(f"Expected 4-channel RGBT input, got {rgbt.shape[1]}")

        if self.backbone_type == "transformer":
            base_feature, stem_feature, encoded_feature, fusion_pairs = self.transformer_backbone(rgbt)
            if self.use_temporal_branch:
                state = self.temporal(encoded_feature, prev_state)
                temporal_feature = encoded_feature + self.temporal_scale * state[0]
            else:
                state = None
                temporal_feature = encoded_feature

            refined_feature = self.transformer_body_conv(temporal_feature + stem_feature)
            hdr = self.transformer_head(base_feature + refined_feature).clamp(0.0, 1.0)

            if return_fusion_pairs:
                pooled_pairs = []
                if self.fusion_variant == "none":
                    return hdr, state, []
                for rgb_feature, ir_feature in fusion_pairs:
                    pooled_pairs.append(
                        (
                            self.fusion_pool(rgb_feature).flatten(1),
                            self.fusion_pool(ir_feature).flatten(1),
                        )
                    )
            else:
                pooled_pairs = []

            return hdr, state, pooled_pairs

        rgb = rgbt[:, :3]
        ir = rgbt[:, 3:].expand(-1, 3, -1, -1)
        rgb_9 = rgb.repeat(1, 3, 1, 1)

        ir0 = self.ir_branch[0](ir)
        ir0 = self.ir_branch[1](ir0)
        ir0 = self.ir_branch[2](ir0)
        ir1 = self.ir_branch[3](ir0)
        ir2 = self.ir_branch[4](ir1)
        ir3 = self.ir_branch[5](ir2)
        ir4 = self.ir_branch[6](ir3)

        stem, enc1, enc2, enc3, enc4, input_rgb, fusion_pairs = self.encoder(rgb_9, ir1, ir2, ir3, ir4)
        if self.use_temporal_branch:
            state = self.temporal(enc4, prev_state)
            temporal_feature = enc4 + self.temporal_scale * state[0]
        else:
            state = None
            temporal_feature = enc4

        dec1 = self.decoder1(temporal_feature, enc3)
        dec2 = self.decoder2(dec1, enc2)
        dec3 = self.decoder3(dec2, enc1)
        dec4 = self.decoder4(dec3, stem)
        dec5 = self.decoder5(dec4, input_rgb)
        hdr = self.head(dec5) + input_rgb
        hdr = hdr.clamp(0.0, 1.0)

        if return_fusion_pairs:
            pooled_pairs = []
            if self.fusion_variant == "none":
                return hdr, state, []
            for rgb_feature, ir_feature in fusion_pairs:
                pooled_pairs.append(
                    (
                        self.fusion_pool(rgb_feature).flatten(1),
                        self.fusion_pool(ir_feature).flatten(1),
                    )
                )
        else:
            pooled_pairs = []

        return hdr, state, pooled_pairs
