import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from .data import AlignedRGBTHDRDataset, HDRVideoSequenceDataset
from .flow import RAFTFlowEstimator
from .losses import FusionAlignmentLoss, ReconstructionLoss, VGGPerceptualLoss, temporal_consistency_loss
from .model import ZSHDRTVNet
from .optim import RiemannianAdam
from .utils import configure_runtime, count_parameters, ensure_dir, latest_checkpoint, load_yaml, move_to_device, psnr, pu_psnr, set_seed


@dataclass
class TrainContext:
    model: ZSHDRTVNet
    optimizer: RiemannianAdam
    scheduler: MultiStepLR
    recon_loss: ReconstructionLoss
    perceptual_loss: Optional[VGGPerceptualLoss]
    fusion_loss: FusionAlignmentLoss
    flow_estimator: Optional[RAFTFlowEstimator]
    device: torch.device
    amp_dtype: Optional[torch.dtype]
    grad_scaler: Optional[torch.cuda.amp.GradScaler]
    base_lrs: Tuple[float, ...]
    warmup_steps: int


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _resolve_amp_dtype(config: Dict, device: torch.device) -> Optional[torch.dtype]:
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


def _autocast_context(context: TrainContext):
    if context.device.type != "cuda" or context.amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=context.amp_dtype)


def _detach_state(state):
    if state is None:
        return None
    return state[0].detach(), state[1].detach()


def _resolve_checkpoint_dtype(config: Dict) -> Optional[torch.dtype]:
    mode = str(config.get("output", {}).get("checkpoint_dtype", "fp32")).lower()
    if mode in {"fp32", "float32", "full"}:
        return None
    if mode in {"fp16", "float16", "half"}:
        return torch.float16
    if mode in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported checkpoint dtype: {mode}")


def _checkpoint_model_state(model: torch.nn.Module, config: Dict) -> Dict[str, torch.Tensor]:
    checkpoint_dtype = _resolve_checkpoint_dtype(config)
    state = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        if checkpoint_dtype is not None and torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=checkpoint_dtype)
        state[name] = tensor
    return state


def _optimizer_step(context: TrainContext, loss: torch.Tensor) -> None:
    context.optimizer.zero_grad(set_to_none=True)
    if context.grad_scaler is not None:
        context.grad_scaler.scale(loss).backward()
        context.grad_scaler.step(context.optimizer)
        context.grad_scaler.update()
    else:
        loss.backward()
        context.optimizer.step()


def _apply_warmup(context: TrainContext, step: int) -> None:
    if context.warmup_steps <= 0:
        return
    if step <= 0 or step > context.warmup_steps:
        return
    scale = min(1.0, float(step) / float(context.warmup_steps))
    for group, base_lr in zip(context.optimizer.param_groups, context.base_lrs):
        group["lr"] = base_lr * scale


def _loader_kwargs(config: Dict, batch_size: int, shuffle: bool, drop_last: bool):
    loader_cfg = config["train"]
    num_workers = int(loader_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": bool(loader_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
    return kwargs


def _set_module_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _configure_stage_trainability(model: ZSHDRTVNet, stage: str, config: Dict) -> None:
    if stage == "image":
        _set_module_requires_grad(model, True)
        temporal_trainable = bool(config.get("train", {}).get("enable_temporal_in_image_stage", False))
        _set_module_requires_grad(model.temporal, temporal_trainable)
        if hasattr(model, "temporal_scale"):
            model.temporal_scale.requires_grad = temporal_trainable
        model.use_temporal_branch = temporal_trainable
        return

    if stage == "video":
        _set_module_requires_grad(model, False)
        _set_module_requires_grad(model.temporal, True)
        if hasattr(model, "temporal_scale"):
            model.temporal_scale.requires_grad = True
        model.use_temporal_branch = True
        return

    if stage == "joint":
        _set_module_requires_grad(model, True)
        model.use_temporal_branch = True
        return

    raise ValueError(f"Unsupported training stage: {stage}")


def create_model_and_losses(config: Dict, device: torch.device) -> TrainContext:
    stage = config["train"]["stage"].lower()
    default_temporal = stage in {"video", "joint"}
    model = ZSHDRTVNet(
        pretrained_backbone=config["model"].get("pretrained_backbone", True),
        use_temporal_branch=bool(config["model"].get("use_temporal_branch", default_temporal)),
        fusion_variant=str(config["model"].get("fusion_variant", "dual")),
        backbone_type=str(config["model"].get("backbone_type", "resnet")),
        model_config=config.get("model", {}),
    ).to(device)
    _configure_stage_trainability(model, stage, config)
    base_lr = float(config["train"]["lr"])
    temporal_lr_scale = float(config["train"].get("temporal_lr_scale", 1.0))
    trainable_named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if temporal_lr_scale != 1.0:
        temporal_params = [
            parameter
            for name, parameter in trainable_named_parameters
            if name.startswith("temporal.") or name == "temporal_scale"
        ]
        other_params = [
            parameter
            for name, parameter in trainable_named_parameters
            if not (name.startswith("temporal.") or name == "temporal_scale")
        ]
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr})
        if temporal_params:
            param_groups.append({"params": temporal_params, "lr": base_lr * temporal_lr_scale})
    else:
        param_groups = [parameter for _, parameter in trainable_named_parameters]
    optimizer = RiemannianAdam(
        param_groups,
        lr=base_lr,
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    milestones = list(config["train"].get("lr_milestones", [20000, 40000, 60000, 80000]))
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=float(config["train"].get("lr_gamma", 0.5)))
    recon_loss = ReconstructionLoss(config["losses"].get("reconstruction", "lab_l1"))
    perceptual_weight = float(config["losses"].get("perceptual_weight", 1.0))
    perceptual_loss = VGGPerceptualLoss().to(device) if perceptual_weight > 0 else None
    fusion_loss = FusionAlignmentLoss(config["losses"].get("fusion_mode", "kl"))
    temporal_enabled = stage in {"video", "joint"} and (
        float(config["losses"].get("short_temporal_weight", 0.0)) > 0
        or float(config["losses"].get("long_temporal_weight", 0.0)) > 0
    )
    flow_estimator = RAFTFlowEstimator().to(device) if temporal_enabled else None
    amp_dtype = _resolve_amp_dtype(config, device)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16) if device.type == "cuda" else None
    warmup_steps = int(config["train"].get("warmup_steps", config["train"].get("warmup_iter", 0)))
    return TrainContext(
        model,
        optimizer,
        scheduler,
        recon_loss,
        perceptual_loss,
        fusion_loss,
        flow_estimator,
        device,
        amp_dtype,
        grad_scaler,
        tuple(group["lr"] for group in optimizer.param_groups),
        warmup_steps,
    )


def build_image_loaders(config: Dict):
    dataset_cfg = config["datasets"]["image"]
    train_dataset = AlignedRGBTHDRDataset(
        hdr_dir=dataset_cfg["hdr_dir"],
        rgb_dir=dataset_cfg["rgb_dir"],
        ir_dir=dataset_cfg["ir_dir"],
        index_path=dataset_cfg["train_index"],
        image_size=_optional_int(dataset_cfg.get("train_image_size", dataset_cfg.get("image_size", 256))),
        crop_size=_optional_int(dataset_cfg.get("train_crop_size")),
        augment=True,
        augment_pipeline=str(dataset_cfg.get("train_augment_pipeline", "basic")),
    )
    val_dataset = AlignedRGBTHDRDataset(
        hdr_dir=dataset_cfg["hdr_dir"],
        rgb_dir=dataset_cfg["rgb_dir"],
        ir_dir=dataset_cfg["ir_dir"],
        index_path=dataset_cfg["val_index"],
        image_size=_optional_int(dataset_cfg.get("val_image_size", dataset_cfg.get("image_size", 256))),
        crop_size=_optional_int(dataset_cfg.get("val_crop_size")),
        augment=False,
        augment_pipeline=str(dataset_cfg.get("val_augment_pipeline", "basic")),
    )
    loader_cfg = config["train"]
    train_loader = DataLoader(
        train_dataset,
        **_loader_kwargs(config, batch_size=int(loader_cfg.get("batch_size", 4)), shuffle=True, drop_last=True),
    )
    val_loader = DataLoader(
        val_dataset,
        **_loader_kwargs(config, batch_size=1, shuffle=False, drop_last=False),
    )
    return train_loader, val_loader


def build_video_loader(config: Dict):
    dataset_cfg = config["datasets"]["video"]
    dataset = HDRVideoSequenceDataset(
        root=dataset_cfg["video_root"],
        sequence_length=int(dataset_cfg.get("sequence_length", 7)),
        image_size=int(dataset_cfg.get("image_size", 256)),
        sample_exposure_index=dataset_cfg.get("sample_exposure_index"),
        pseudo_ir_mode=dataset_cfg.get("pseudo_ir_mode", "luminance"),
        limit_videos=dataset_cfg.get("limit_videos"),
        include_videos=dataset_cfg.get("train_list"),
    )
    return DataLoader(
        dataset,
        **_loader_kwargs(config, batch_size=int(config["train"].get("video_batch_size", 1)), shuffle=True, drop_last=True),
    )


def build_video_loaders(config: Dict):
    dataset_cfg = config["datasets"]["video"]
    train_dataset = HDRVideoSequenceDataset(
        root=dataset_cfg["video_root"],
        sequence_length=int(dataset_cfg.get("sequence_length", 7)),
        image_size=int(dataset_cfg.get("image_size", 256)),
        sample_exposure_index=dataset_cfg.get("sample_exposure_index"),
        pseudo_ir_mode=dataset_cfg.get("pseudo_ir_mode", "luminance"),
        limit_videos=dataset_cfg.get("limit_videos"),
        include_videos=dataset_cfg.get("train_list"),
        deterministic=False,
    )
    train_loader = DataLoader(
        train_dataset,
        **_loader_kwargs(config, batch_size=int(config["train"].get("video_batch_size", 1)), shuffle=True, drop_last=True),
    )

    val_list = dataset_cfg.get("val_list")
    if not val_list:
        return train_loader, None

    val_dataset = HDRVideoSequenceDataset(
        root=dataset_cfg["video_root"],
        sequence_length=int(dataset_cfg.get("val_sequence_length", dataset_cfg.get("sequence_length", 7))),
        image_size=int(dataset_cfg.get("val_image_size", dataset_cfg.get("image_size", 256))),
        sample_exposure_index=dataset_cfg.get("val_sample_exposure_index"),
        pseudo_ir_mode=dataset_cfg.get("pseudo_ir_mode", "luminance"),
        limit_videos=dataset_cfg.get("val_limit_videos"),
        include_videos=val_list,
        deterministic=True,
    )
    val_loader = DataLoader(
        val_dataset,
        **_loader_kwargs(config, batch_size=1, shuffle=False, drop_last=False),
    )
    return train_loader, val_loader


def save_checkpoint(context: TrainContext, config: Dict, stage: str, epoch: int, step: int) -> Path:
    checkpoint_dir = ensure_dir(config["output"]["checkpoint_dir"])
    output_cfg = config.get("output", {})
    overwrite_best = bool(output_cfg.get("overwrite_best", True))
    overwrite_regular = bool(output_cfg.get("overwrite_regular", False))
    if stage.endswith("_best") and overwrite_best:
        path = checkpoint_dir / f"{stage}.pth"
    elif overwrite_regular:
        path = checkpoint_dir / f"{stage}.pth"
    else:
        path = checkpoint_dir / f"{stage}_epoch_{epoch:04d}_step_{step:07d}.pth"

    payload = {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "model": _checkpoint_model_state(context.model, config),
        "config": config,
    }
    if output_cfg.get("save_optimizer_state", True):
        payload["optimizer"] = context.optimizer.state_dict()
    if output_cfg.get("save_scheduler_state", True):
        payload["scheduler"] = context.scheduler.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(context: TrainContext, checkpoint_path: str) -> Tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=context.device)
    context.model.load_state_dict(checkpoint["model"], strict=False)
    if "optimizer" in checkpoint:
        context.optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        context.scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


def initialize_model(context: TrainContext, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=context.device)
    context.model.load_state_dict(checkpoint["model"], strict=False)


def train_image_epoch(context: TrainContext, loader: DataLoader, config: Dict, step: int) -> int:
    context.model.train()
    weights = config["losses"]
    log_every = int(config["train"].get("log_every", 10))
    steps_per_epoch = int(config["train"].get("steps_per_epoch", 0))
    for batch_index, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        batch = move_to_device(batch, context.device)
        with _autocast_context(context):
            prediction, _, fusion_pairs = context.model(batch["rgbt"], return_fusion_pairs=True)
            recon = context.recon_loss(prediction, batch["hdr"])
            perceptual = (
                context.perceptual_loss(prediction, batch["hdr"])
                if context.perceptual_loss is not None
                else prediction.new_tensor(0.0)
            )
            fusion = context.fusion_loss(fusion_pairs) if fusion_pairs else prediction.new_tensor(0.0)
            total = (
                float(weights.get("reconstruction_weight", 1.0)) * recon
                + float(weights.get("perceptual_weight", 1.0)) * perceptual
                + float(weights.get("fusion_weight", 1.0)) * fusion
            )
        _apply_warmup(context, step + 1)
        _optimizer_step(context, total)
        context.scheduler.step()
        step += 1
        if batch_index % log_every == 0:
            batch_time = time.perf_counter() - step_start
            samples_per_second = batch["rgbt"].shape[0] / max(batch_time, 1e-6)
            print(
                f"[image] step={step} recon={recon.item():.4f} "
                f"perc={perceptual.item():.4f} fusion={fusion.item():.4f} total={total.item():.4f} "
                f"time={batch_time:.3f}s samples/s={samples_per_second:.2f}"
            )
        if steps_per_epoch > 0 and batch_index >= steps_per_epoch:
            break
    return step


def train_joint_epoch(
    context: TrainContext,
    image_loader: DataLoader,
    video_loader: DataLoader,
    config: Dict,
    step: int,
) -> int:
    context.model.train()
    log_every = int(config["train"].get("log_every", 10))
    steps_per_epoch = int(config["train"].get("steps_per_epoch", 0))
    max_steps = steps_per_epoch if steps_per_epoch > 0 else min(len(image_loader), len(video_loader)) * 2
    image_steps_per_video_step = max(1, int(config["train"].get("joint_image_steps_per_video_step", 1)))
    cycle = image_steps_per_video_step + 1
    image_iter = iter(image_loader)
    video_iter = iter(video_loader)

    for batch_index in range(1, max_steps + 1):
        step_start = time.perf_counter()
        train_image_batch = (batch_index % cycle) != 0

        if train_image_batch:
            try:
                batch = next(image_iter)
            except StopIteration:
                image_iter = iter(image_loader)
                batch = next(image_iter)
            batch = move_to_device(batch, context.device)
            weights = config["losses"]
            with _autocast_context(context):
                prediction, _, fusion_pairs = context.model(batch["rgbt"], return_fusion_pairs=True)
                recon = context.recon_loss(prediction, batch["hdr"])
                perceptual = (
                    context.perceptual_loss(prediction, batch["hdr"])
                    if context.perceptual_loss is not None
                    else prediction.new_tensor(0.0)
                )
                fusion = context.fusion_loss(fusion_pairs) if fusion_pairs else prediction.new_tensor(0.0)
                total = (
                    float(weights.get("reconstruction_weight", 1.0)) * recon
                    + float(weights.get("perceptual_weight", 1.0)) * perceptual
                    + float(weights.get("fusion_weight", 1.0)) * fusion
                )
            _apply_warmup(context, step + 1)
            _optimizer_step(context, total)
            context.scheduler.step()
            step += 1
            if batch_index % log_every == 0:
                batch_time = time.perf_counter() - step_start
                samples_per_second = batch["rgbt"].shape[0] / max(batch_time, 1e-6)
                print(
                    f"[joint:image] step={step} recon={recon.item():.4f} "
                    f"perc={perceptual.item():.4f} fusion={fusion.item():.4f} total={total.item():.4f} "
                    f"time={batch_time:.3f}s samples/s={samples_per_second:.2f}"
                )
            continue

        try:
            batch = next(video_iter)
        except StopIteration:
            video_iter = iter(video_loader)
            batch = next(video_iter)
        batch = move_to_device(batch, context.device)
        weights = config["losses"]
        sequence = batch["rgbt"]
        hdr_gt = batch["hdr"]
        sdr = batch["rgb"]
        outputs = []
        reconstruction = sequence.new_tensor(0.0)
        perceptual = sequence.new_tensor(0.0)
        state = None

        for frame_index in range(sequence.shape[1]):
            with _autocast_context(context):
                prediction, state, _ = context.model(sequence[:, frame_index], state, return_fusion_pairs=False)
                reconstruction = reconstruction + context.recon_loss(prediction, hdr_gt[:, frame_index])
                if context.perceptual_loss is not None:
                    perceptual = perceptual + context.perceptual_loss(prediction, hdr_gt[:, frame_index])
            outputs.append(prediction.float())
            state = _detach_state(state)

        if context.flow_estimator is not None:
            outputs_list = list(outputs)
            sdr_list = [sdr[:, frame_index] for frame_index in range(sdr.shape[1])]
            short_term, long_term = temporal_consistency_loss(
                outputs_list,
                sdr_list,
                context.flow_estimator,
                short_weight=float(weights.get("short_temporal_weight", 100.0)),
                long_weight=float(weights.get("long_temporal_weight", 100.0)),
                alpha=float(weights.get("occlusion_alpha", 50.0)),
            )
        else:
            short_term = sequence.new_tensor(0.0)
            long_term = sequence.new_tensor(0.0)

        total = (
            float(weights.get("reconstruction_weight", 1.0)) * reconstruction / sequence.shape[1]
            + float(weights.get("perceptual_weight", 1.0)) * perceptual / sequence.shape[1]
            + short_term
            + long_term
        )
        _apply_warmup(context, step + 1)
        _optimizer_step(context, total)
        context.scheduler.step()
        step += 1
        if batch_index % log_every == 0:
            batch_time = time.perf_counter() - step_start
            frames_per_second = (sequence.shape[0] * sequence.shape[1]) / max(batch_time, 1e-6)
            print(
                f"[joint:video] step={step} recon={(reconstruction / sequence.shape[1]).item():.4f} "
                f"perc={(perceptual / sequence.shape[1]).item():.4f} "
                f"short={short_term.item():.4f} long={long_term.item():.4f} total={total.item():.4f} "
                f"time={batch_time:.3f}s frames/s={frames_per_second:.2f}"
            )
    return step


def evaluate_image(context: TrainContext, loader: DataLoader) -> Dict[str, float]:
    context.model.eval()
    avg_psnr = 0.0
    avg_pu_psnr = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, context.device)
            with _autocast_context(context):
                prediction, _, _ = context.model(batch["rgbt"], return_fusion_pairs=False)
            avg_psnr += psnr(prediction, batch["hdr"])
            avg_pu_psnr += pu_psnr(prediction, batch["hdr"])
            count += 1
    return {"psnr": avg_psnr / max(count, 1), "pu_psnr": avg_pu_psnr / max(count, 1)}


def train_video_epoch(context: TrainContext, loader: DataLoader, config: Dict, step: int) -> int:
    context.model.train()
    weights = config["losses"]
    log_every = int(config["train"].get("log_every", 10))
    steps_per_epoch = int(config["train"].get("steps_per_epoch", 0))
    for batch_index, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        batch = move_to_device(batch, context.device)
        sequence = batch["rgbt"]
        hdr_gt = batch["hdr"]
        sdr = batch["rgb"]
        outputs = []
        reconstruction = sequence.new_tensor(0.0)
        perceptual = sequence.new_tensor(0.0)
        state = None

        for frame_index in range(sequence.shape[1]):
            with _autocast_context(context):
                prediction, state, _ = context.model(sequence[:, frame_index], state, return_fusion_pairs=False)
                reconstruction = reconstruction + context.recon_loss(prediction, hdr_gt[:, frame_index])
                if context.perceptual_loss is not None:
                    perceptual = perceptual + context.perceptual_loss(prediction, hdr_gt[:, frame_index])
            outputs.append(prediction.float())
            state = _detach_state(state)

        if context.flow_estimator is not None:
            outputs_list = list(outputs)
            sdr_list = [sdr[:, frame_index] for frame_index in range(sdr.shape[1])]
            short_term, long_term = temporal_consistency_loss(
                outputs_list,
                sdr_list,
                context.flow_estimator,
                short_weight=float(weights.get("short_temporal_weight", 100.0)),
                long_weight=float(weights.get("long_temporal_weight", 100.0)),
                alpha=float(weights.get("occlusion_alpha", 50.0)),
            )
        else:
            short_term = sequence.new_tensor(0.0)
            long_term = sequence.new_tensor(0.0)
        total = (
            float(weights.get("reconstruction_weight", 1.0)) * reconstruction / sequence.shape[1]
            + float(weights.get("perceptual_weight", 1.0)) * perceptual / sequence.shape[1]
            + short_term
            + long_term
        )
        _apply_warmup(context, step + 1)
        _optimizer_step(context, total)
        context.scheduler.step()
        step += 1
        if batch_index % log_every == 0:
            batch_time = time.perf_counter() - step_start
            frames_per_second = (sequence.shape[0] * sequence.shape[1]) / max(batch_time, 1e-6)
            print(
                f"[video] step={step} recon={(reconstruction / sequence.shape[1]).item():.4f} "
                f"perc={(perceptual / sequence.shape[1]).item():.4f} "
                f"short={short_term.item():.4f} long={long_term.item():.4f} total={total.item():.4f} "
                f"time={batch_time:.3f}s frames/s={frames_per_second:.2f}"
            )
        if steps_per_epoch > 0 and batch_index >= steps_per_epoch:
            break
    return step


def evaluate_video(context: TrainContext, loader: DataLoader, config: Dict) -> Dict[str, float]:
    context.model.eval()
    weights = config["losses"]
    metrics = {
        "video_psnr": 0.0,
        "video_pu_psnr": 0.0,
        "video_recon": 0.0,
        "video_perc": 0.0,
        "video_short": 0.0,
        "video_long": 0.0,
        "video_total": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, context.device)
            sequence = batch["rgbt"]
            hdr_gt = batch["hdr"]
            sdr = batch["rgb"]
            outputs = []
            reconstruction = sequence.new_tensor(0.0)
            perceptual = sequence.new_tensor(0.0)
            avg_psnr = 0.0
            avg_pu_psnr = 0.0
            state = None

            for frame_index in range(sequence.shape[1]):
                with _autocast_context(context):
                    prediction, state, _ = context.model(sequence[:, frame_index], state, return_fusion_pairs=False)
                    reconstruction = reconstruction + context.recon_loss(prediction, hdr_gt[:, frame_index])
                    if context.perceptual_loss is not None:
                        perceptual = perceptual + context.perceptual_loss(prediction, hdr_gt[:, frame_index])
                outputs.append(prediction.float())
                state = _detach_state(state)
                avg_psnr += psnr(prediction.float(), hdr_gt[:, frame_index].float())
                avg_pu_psnr += pu_psnr(prediction.float(), hdr_gt[:, frame_index].float())

            if context.flow_estimator is not None:
                outputs_list = list(outputs)
                sdr_list = [sdr[:, frame_index] for frame_index in range(sdr.shape[1])]
                short_term, long_term = temporal_consistency_loss(
                    outputs_list,
                    sdr_list,
                    context.flow_estimator,
                    short_weight=float(weights.get("short_temporal_weight", 100.0)),
                    long_weight=float(weights.get("long_temporal_weight", 100.0)),
                    alpha=float(weights.get("occlusion_alpha", 50.0)),
                )
            else:
                short_term = sequence.new_tensor(0.0)
                long_term = sequence.new_tensor(0.0)

            recon_value = float((reconstruction / sequence.shape[1]).item())
            perc_value = float((perceptual / sequence.shape[1]).item())
            short_value = float(short_term.item())
            long_value = float(long_term.item())
            total_value = (
                float(weights.get("reconstruction_weight", 1.0)) * recon_value
                + float(weights.get("perceptual_weight", 1.0)) * perc_value
                + short_value
                + long_value
            )
            metrics["video_psnr"] += avg_psnr / sequence.shape[1]
            metrics["video_pu_psnr"] += avg_pu_psnr / sequence.shape[1]
            metrics["video_recon"] += recon_value
            metrics["video_perc"] += perc_value
            metrics["video_short"] += short_value
            metrics["video_long"] += long_value
            metrics["video_total"] += total_value
            count += 1

    for key in metrics:
        metrics[key] /= max(count, 1)
    return metrics


def run_training(config_path: str) -> None:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 10)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(config, device)
    context = create_model_and_losses(config, device)

    stage = config["train"]["stage"].lower()
    print(f"Stage: {stage}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(parameter.numel() for parameter in context.model.parameters()):,}")
    print(f"Trainable parameters: {count_parameters(context.model.parameters()):,}")
    print(f"Mixed precision: {context.amp_dtype if context.amp_dtype is not None else 'off'}")
    print(f"Fusion loss: {context.fusion_loss.mode}")
    print(f"Fusion variant: {context.model.fusion_variant}")
    print(f"Backbone: {context.model.backbone_type}")
    print(f"Temporal branch: {'on' if context.model.use_temporal_branch else 'off'}")

    start_epoch = 0
    step = 0
    init_checkpoint = config["train"].get("init_checkpoint")
    if init_checkpoint:
        initialize_model(context, init_checkpoint)
        print(f"Initialized model weights from {init_checkpoint}")
    resume = config["train"].get("resume")
    if resume:
        start_epoch, step = load_checkpoint(context, resume)
        print(f"Resumed from {resume} at epoch={start_epoch}, step={step}")

    image_train_loader = image_val_loader = None
    video_loader = video_val_loader = None
    if stage in {"image", "joint"}:
        image_train_loader, image_val_loader = build_image_loaders(config)
    if stage in {"video", "joint"}:
        video_loader, video_val_loader = build_video_loaders(config)

    best_metric = str(config["train"].get("best_metric", "psnr")).lower()
    best_mode = str(config["train"].get("best_mode", "max")).lower()
    if best_mode not in {"max", "min"}:
        raise ValueError(f"Unsupported best mode: {best_mode}")
    best_value = float("-inf") if best_mode == "max" else float("inf")
    total_epochs = int(config["train"].get("epochs", 1))
    validate_every = int(config["train"].get("validate_every", 1))
    save_every = int(config["train"].get("save_every", 1))
    steps_per_epoch = int(config["train"].get("steps_per_epoch", 0))
    print(f"Best metric: {best_metric} ({best_mode})")
    if steps_per_epoch > 0:
        print(f"Steps per epoch: {steps_per_epoch}")

    def resolve_best_metric(metrics: Dict[str, float], metric_name: str, phase: str) -> Tuple[str, float] | None:
        if metric_name in metrics:
            return metric_name, float(metrics[metric_name])
        if phase == "image":
            aliases = {
                "image_psnr": "psnr",
                "image_pu_psnr": "pu_psnr",
            }
        else:
            aliases = {
                "psnr": "video_psnr",
                "pu_psnr": "video_pu_psnr",
                "recon": "video_recon",
                "perc": "video_perc",
                "short": "video_short",
                "long": "video_long",
                "total": "video_total",
            }
        alias = aliases.get(metric_name)
        if alias is None or alias not in metrics:
            return None
        return alias, float(metrics[alias])

    for epoch in range(start_epoch + 1, total_epochs + 1):
        epoch_start = time.perf_counter()
        if stage == "joint" and image_train_loader is not None and video_loader is not None:
            step = train_joint_epoch(context, image_train_loader, video_loader, config, step)
        elif stage in {"image", "joint"} and image_train_loader is not None:
            step = train_image_epoch(context, image_train_loader, config, step)
        if stage == "video" and video_loader is not None:
            step = train_video_epoch(context, video_loader, config, step)

        if image_val_loader is not None and validate_every > 0 and epoch % validate_every == 0:
            print(f"[val_start] epoch={epoch} mode=image")
            metrics = evaluate_image(context, image_val_loader)
            print(f"[val] epoch={epoch} psnr={metrics['psnr']:.3f} pu_psnr={metrics['pu_psnr']:.3f}")
            resolved = resolve_best_metric(metrics, best_metric, "image")
            if resolved is None:
                raise ValueError(f"Unsupported best metric for image validation: {best_metric}")
            resolved_name, current_value = resolved
            is_better = current_value > best_value if best_mode == "max" else current_value < best_value
            if is_better:
                best_value = current_value
                path = save_checkpoint(context, config, f"{stage}_best", epoch, step)
                print(f"Saved best checkpoint to {path} ({resolved_name}={current_value:.3f})")

        if video_val_loader is not None and validate_every > 0 and epoch % validate_every == 0:
            print(f"[val_start] epoch={epoch} mode=video")
            metrics = evaluate_video(context, video_val_loader, config)
            print(
                f"[video_val] epoch={epoch} psnr={metrics['video_psnr']:.3f} "
                f"pu_psnr={metrics['video_pu_psnr']:.3f} recon={metrics['video_recon']:.4f} "
                f"short={metrics['video_short']:.4f} long={metrics['video_long']:.4f} total={metrics['video_total']:.4f}"
            )
            resolved = resolve_best_metric(metrics, best_metric, "video")
            if resolved is not None and stage == "video":
                resolved_name, current_value = resolved
                is_better = current_value > best_value if best_mode == "max" else current_value < best_value
                if is_better:
                    best_value = current_value
                    path = save_checkpoint(context, config, f"{stage}_best", epoch, step)
                    print(f"Saved best checkpoint to {path} ({resolved_name}={current_value:.3f})")

        if save_every > 0 and epoch % save_every == 0:
            path = save_checkpoint(context, config, stage, epoch, step)
            print(f"Saved checkpoint to {path}")
        print(f"[epoch] epoch={epoch} elapsed={time.perf_counter() - epoch_start:.2f}s")
