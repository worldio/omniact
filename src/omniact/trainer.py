"""Training loop for :class:`OmniActModel` (P1: morphology-agnostic single-pass flow matching).

P1 trains on ALL ``(key, embodiment)`` samples uniformly from ONE :class:`OmniActDataset`: every step is
a single forward + per-node flow-matching velocity loss, with terminal (non-actuated) nodes masked out
via ``axis_mask``. The embodiment enters only through the per-sample node geometry + graph, so droid and
xarm7 share every weight and cross-embodiment transfer rides the shared representation — there is no
explicit pairing / invariance / cross-decode here (those are P2, which the model already supports).

Handles DDP, StatefulDataLoader, fsspec checkpoints and wandb, with zero-impact resume (restore loader
state mid-epoch + per-rank RNG; load to CPU first).
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import os
import random
import tempfile
import time
from dataclasses import dataclass, field

import fsspec
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from .ddp import get_local_rank, get_rank, get_world_size, is_main_process, node_local_download
from .metrics import RunningStats
from .ratio_sampler import RatioSampler
from .dataset import NORMALIZATION_FIELDS, collate
from .model import OmniActModel, OmniActModelConfig, checkpoint_config


@dataclass
class OmniActTrainConfig:
    policy: OmniActModelConfig = field(default_factory=OmniActModelConfig)

    # Training
    batch_size: int = 64
    num_epochs: int = 10000
    max_steps: int = 0
    learning_rate: float = 5e-4
    weight_decay: float = 1e-10
    grad_clip_norm: float = 1.0
    warmup_steps: int = 200
    pose_loss_weight: float = 1.0      # weight on the TCP pose flow loss added to the joint flow loss

    # Data
    num_workers: int = 16
    prefetch_factor: int = 2
    persistent_workers: bool = True
    sampled_length: int | None = None   # per-epoch length in samples (None = natural len(dataset))

    # Logging
    log_interval: int = 10
    val_interval: int = 0
    limit_val_batches: int = 20
    wandb_project: str = "omniact"
    wandb_run_name: str | None = None
    use_wandb: bool = True

    # Checkpointing
    checkpoint_output: str | None = None
    checkpoint_interval: int = 1000
    named_checkpoint_interval: int = 10000
    load_model_from_ckpt: str | None = None

    # torch.compile (train step only; validation stays eager)
    compile: bool = False
    compile_mode: str = "default"

    # DDP gradient bucket size.
    ddp_bucket_cap_mb: int = 400

    device: str = "cuda"


class OmniActTrainer:
    def __init__(
        self,
        config: OmniActTrainConfig,
        train_dataset: Dataset,
        val_datasets: dict[str, Dataset],
    ):
        """Build the model + optimizer + loaders. Copy the dataset's scales, pose origin, and ramp powers
        into model buffers for checkpoint storage and policy decoding."""
        self.config = config
        self.device = torch.device(config.device)

        self.model = OmniActModel(config.policy).to(self.device)
        torch.set_float32_matmul_precision("high")
        self._raw_model = self.model
        # Carry the dataset's normalization (fixed defaults or --stats-sample-size override) into model buffers
        # for checkpoint storage and policy decoding.
        base_ds = train_dataset
        while isinstance(base_ds, Subset):      # reach the base dataset through any Subset nesting
            base_ds = base_ds.dataset
        for name in NORMALIZATION_FIELDS:
            getattr(self._raw_model, name).copy_(getattr(base_ds, name).to(self.device))
        ddp_kwargs: dict = {"bucket_cap_mb": config.ddp_bucket_cap_mb}
        if self.device.type == "cuda":
            ddp_kwargs["device_ids"] = [get_local_rank()]
        self.model = DDP(self.model, **ddp_kwargs)
        if config.compile:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
            self.model = torch.compile(self.model, mode=config.compile_mode, dynamic=False)
            logger.info(f"torch.compile enabled (mode={config.compile_mode})")
        logger.info(
            f"OmniActModel params: {self._raw_model.num_trainable_params():,} trainable "
            f"| trunk=0..{self._raw_model.q}, head={self._raw_model.q + 1}..{config.policy.policy_blocks - 1} "
            f"(rank {get_rank()}/{get_world_size()})"
        )

        # Optimizer: core + heads at lr; SigLIP (if trainable) at lr/10.
        siglip_params = [p for p in self._raw_model.siglip.parameters() if p.requires_grad]
        other_params = [
            p for n, p in self._raw_model.named_parameters()
            if p.requires_grad and not n.startswith("siglip.")
        ]
        param_groups = [{"params": other_params, "lr": config.learning_rate}]
        if siglip_params:
            param_groups.append({"params": siglip_params, "lr": config.learning_rate / 10})
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay, eps=1e-4, fused=True)
        if config.warmup_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-8, total_iters=config.warmup_steps)
        else:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=0)

        # Data: one StatefulDataLoader over all (key, embodiment) samples.
        loader_kwargs = dict(num_workers=config.num_workers, pin_memory=True)
        if config.num_workers > 0:
            loader_kwargs["persistent_workers"] = config.persistent_workers
            loader_kwargs["prefetch_factor"] = config.prefetch_factor
        self.sampler = RatioSampler(train_dataset, sampled_length=config.sampled_length, shuffle=True)
        self.train_loader = StatefulDataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=False, sampler=self.sampler,
            drop_last=True, collate_fn=collate, **loader_kwargs)
        self.val_loaders = {
            name: DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True, collate_fn=collate,
            )
            for name, dataset in val_datasets.items()
            if len(dataset) > 0
        }

        # Checkpoint output + auto-resume.
        self.global_step = 0
        self.checkpoint_output = config.checkpoint_output
        self._active_upload_processes: list[multiprocessing.Process] = []
        if self.checkpoint_output is not None:
            fs, _, paths = fsspec.get_fs_token_paths(self.checkpoint_output)
            fs.makedirs(self.checkpoint_output, exist_ok=True)
            logger.info(f"Checkpoint output: {self.checkpoint_output}")
            if fs.exists(f"{paths[0]}/last.pt"):
                self._load_training_checkpoint(f"{self.checkpoint_output}/last.pt")
        if config.load_model_from_ckpt is not None:
            self._load_model_weights(config.load_model_from_ckpt)

        self.wandb_run = None
        if config.use_wandb and is_main_process():
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project, name=config.wandb_run_name,
                    config={"policy": dataclasses.asdict(config.policy),
                            "train": {k: v for k, v in vars(config).items() if k != "policy"}})
                logger.info(f"WandB run: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"Failed to init WandB: {e}")

    # -----------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        out: dict = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v   # key/embodiment are lists
        if out["views"].dtype == torch.uint8:   # uint8 from the loader -> [-1, 1] on GPU
            out["views"] = out["views"].float() / 127.5 - 1.0
        return out

    def _head_masks(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-(B, T, N) keep masks for the joint and pose heads: the dataset's ``action_mag_mask`` (real action
        nodes minus big movers, per step) split by head — actuated nodes for the joint loss, the TCP node for
        the pose loss. Disjoint, so each head's loss counts only its own kept nodes."""
        m = self._raw_model
        keep = batch["action_mag_mask"]                                       # (B, T, N)
        c = max(1, int(getattr(getattr(m, "config", None), "flow_noise_samples", 1)))   # a mask row per draw
        return ((m.action_joint_mask(batch).unsqueeze(1) & keep).repeat_interleave(c, 0),
                (m.action_pose_mask(batch).unsqueeze(1) & keep).repeat_interleave(c, 0))

    def _flow_losses(self, out: dict, batch: dict) -> dict[str, torch.Tensor]:
        """Joint + TCP-pose flow-matching velocity losses, each masked to its head's kept action nodes (via the
        dataset's ``action_mag_mask``), plus the ``pose_loss_weight``-weighted total used for the backward pass."""
        m = self._raw_model
        joint_mask, pose_mask = self._head_masks(batch)
        joint = m.action_loss(out["v_pred"], out["v_target"], joint_mask)
        pose = m.action_loss(out["v_pred_pose"], out["v_target_pose"], pose_mask)
        return {"joint": joint, "pose": pose, "total": joint + self.config.pose_loss_weight * pose}

    def _tick(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    def train(self):
        logger.info(
            f"Starting OmniAct training: epochs={self.config.num_epochs}, "
            f"batch_size={self.config.batch_size}, lr={self.config.learning_rate}")
        steps_per_epoch = len(self.train_loader)
        start_epoch = self.global_step // steps_per_epoch if steps_per_epoch else 0
        start_step = self.global_step % steps_per_epoch if steps_per_epoch else 0
        for epoch in range(start_epoch, self.config.num_epochs):
            self._train_epoch(epoch, skip_steps=start_step)
            start_step = 0
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
        self.global_step -= 1
        ckpt = self._build_checkpoint()          # ALL ranks — _build_checkpoint's all_gather_object
        self.global_step += 1                    # is collective; rank-0-only here would deadlock the rest
        if is_main_process():
            self._upload_checkpoint_from_dict(ckpt, "last")
            final_step = ckpt["global_step"]
            if (self.config.named_checkpoint_interval > 0
                    and final_step % self.config.named_checkpoint_interval == 0):
                self._upload_checkpoint_from_dict(ckpt, f"step_{final_step:06d}")
            self._wait_for_uploads()
        logger.info("OmniAct training complete!")
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _train_epoch(self, epoch: int, skip_steps: int = 0):
        from tqdm import tqdm
        self.model.train()
        self.sampler.set_epoch(epoch)
        it = iter(self.train_loader)
        n = len(self.train_loader)
        pbar = tqdm(range(skip_steps, n), total=n - skip_steps, desc=f"Epoch {epoch}",
                    disable=not is_main_process())
        for _ in pbar:
            t0 = self._tick()
            raw = next(it)
            t_data = self._tick()
            batch = self._to_device(raw)
            # torch.compile() guards on non-tensor dict values, and key/embodiment differ every batch — passing
            # them to the compiled module would recompile every step. The model reads only tensors.
            model_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(model_batch)
                losses = self._flow_losses(out, batch)
                loss = losses["total"]
            self.optimizer.zero_grad()
            loss.backward()
            pre_clip = None
            if self.config.grad_clip_norm > 0:
                pre_clip = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            if self.global_step % self.config.log_interval == 0 and self.wandb_run is not None:
                log = {"train/flow_loss": loss.item(), "train/joint_flow_loss": losses["joint"].item(),
                       "train/pose_flow_loss": losses["pose"].item(),
                       "train/lr": self.optimizer.param_groups[0]["lr"],
                       "timing/data": t_data - t0, "timing/step": self._tick() - t0}
                if pre_clip is not None:
                    log["grad_norm/pre_clip"] = pre_clip.item()
                self.wandb_run.log(log, step=self.global_step)
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if (self.val_loaders and self.config.val_interval > 0
                    and self.global_step > 0 and self.global_step % self.config.val_interval == 0):
                self._validate(epoch)
                if get_world_size() > 1:
                    torch.distributed.barrier()
                self.model.train()

            if self.global_step > 0 and self.global_step % self.config.checkpoint_interval == 0:
                ckpt = self._build_checkpoint()
                if is_main_process():
                    self._upload_checkpoint_from_dict(ckpt, "last")
                    if self.global_step % self.config.named_checkpoint_interval == 0:
                        self._upload_checkpoint_from_dict(ckpt, f"step_{self.global_step:06d}")

            self.global_step += 1
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

    @torch.no_grad()
    def _validate(self, epoch: int):
        if not self.val_loaders or not is_main_process():
            return
        self.model.eval()
        logged_metrics: dict[str, float | int] = {}
        for split_name, loader in self.val_loaders.items():
            stats: dict[str, RunningStats] = {}
            num_samples = 0

            def track(key, value):
                stats.setdefault(key, RunningStats()).update(value)

            for i, raw in enumerate(loader):
                if self.config.limit_val_batches > 0 and i >= self.config.limit_val_batches:
                    break
                num_samples += len(raw["key"])
                batch = self._to_device(raw)
                out = self._raw_model(batch)
                losses = self._flow_losses(out, batch)
                track("flow_loss", losses["total"].item())
                track("joint_flow_loss", losses["joint"].item())
                track("pose_flow_loss", losses["pose"].item())
                # Open-loop MSE: denoise the chunk, compare to the GT relative targets over the kept head nodes.
                joint_mask, pose_mask = self._head_masks(batch)
                joint_pred, pose_pred = self._raw_model.predict(batch, n_steps=batch["actions"].shape[1])
                track("action_mse", self._raw_model.action_loss(joint_pred, batch["actions"], joint_mask).item())
                track("pose_mse", self._raw_model.action_loss(pose_pred, batch["node_action_pose"], pose_mask).item())

            if not stats:
                continue
            logged_metrics.update(
                {f"val/{split_name}/{key}": value.mean for key, value in stats.items()}
            )
            logged_metrics[f"val/{split_name}/num_samples"] = num_samples
            logger.info(
                f"Val {split_name} epoch {epoch}: "
                + " | ".join(f"{key}={value.mean:.4f}" for key, value in stats.items())
                + f" | num_samples={num_samples}"
            )

        if self.wandb_run is not None and logged_metrics:
            self.wandb_run.log(logged_metrics, step=self.global_step)

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _build_checkpoint(self) -> dict:
        # Per-rank RNG states for a reproducible (zero-impact) resume. All ranks
        # contribute via all_gather so one checkpoint file restores every rank.
        local_rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            local_rng["torch_cuda"] = torch.cuda.get_rng_state(self.device)
        if get_world_size() > 1:
            rng_gather: list[dict | None] = [None] * get_world_size()
            torch.distributed.all_gather_object(rng_gather, local_rng)
            rng_states = {r: s for r, s in enumerate(rng_gather)}
        else:
            rng_states = {0: local_rng}
        return {
            "global_step": self.global_step + 1,
            "omniact_config": dataclasses.asdict(self.config.policy),
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            # Single StatefulDataLoader -> one entry "main"; restored mid-epoch only.
            "dataloader_states": {"main": self.train_loader.state_dict()},
            "rng_states": rng_states,
        }

    @staticmethod
    def _upload_file(src_path: str, dst_path: str, delete_src: bool = True):
        try:
            fs, _, _ = fsspec.get_fs_token_paths(dst_path)
            staging = f"{dst_path}.{os.urandom(4).hex()}.tmp"
            fs.put(src_path, staging)
            fs.mv(staging, dst_path)
            logger.info(f"Upload completed: {dst_path}")
            if delete_src and os.path.exists(src_path):
                os.remove(src_path)
        except Exception as e:
            logger.error(f"Error uploading checkpoint: {e}")

    def _upload_checkpoint_from_dict(self, checkpoint: dict, name: str):
        if self.checkpoint_output is None:
            return
        fd, tmp_path = tempfile.mkstemp(suffix=".pt", prefix="omniact_ckpt_")
        os.close(fd)
        torch.save(checkpoint, tmp_path)
        dst = f"{self.checkpoint_output}/{name}.pt"
        logger.info(f"Saving OmniAct checkpoint at step {self.global_step} -> {dst}")
        proc = multiprocessing.Process(target=self._upload_file, args=(tmp_path, dst, True))
        proc.start()
        self._active_upload_processes.append(proc)
        self._active_upload_processes = [p for p in self._active_upload_processes if p.is_alive()]

    def _wait_for_uploads(self):
        for p in self._active_upload_processes:
            p.join()
        self._active_upload_processes.clear()

    def _load_training_checkpoint(self, path: str):
        # Load to CPU first (optimizer state ~2x params; map_location=cuda spikes each rank's GPU).
        with node_local_download(path, logger) as local_path:
            ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
        trained_horizon = OmniActModelConfig(**checkpoint_config(ckpt)).action_horizon
        if trained_horizon != self.config.policy.action_horizon:
            raise ValueError(f"Resume action_horizon mismatch: checkpoint={trained_horizon}, "
                             f"training={self.config.policy.action_horizon}; match --window-size to the checkpoint")
        self._raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]

        # At an epoch boundary the saved loader state encodes "iteration fully consumed";
        # replaying it yields an immediately-exhausted iterator. Restore mid-epoch only
        # (zero-impact resume: same batches, same order).
        steps_per_epoch = len(self.train_loader)
        at_epoch_boundary = steps_per_epoch > 0 and self.global_step % steps_per_epoch == 0
        if not at_epoch_boundary and "dataloader_states" in ckpt:
            self.train_loader.load_state_dict(ckpt["dataloader_states"]["main"])

        # Restore per-rank RNG so flow-matching noise / view sampling resume identically.
        if "rng_states" in ckpt:
            rng = ckpt["rng_states"][get_rank()]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.random.set_rng_state(rng["torch_cpu"].cpu())
            if "torch_cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng["torch_cuda"].cpu(), self.device)

        logger.info(f"Resumed OmniAct training from {path} (step {self.global_step}; "
                    f"{'fresh epoch' if at_epoch_boundary else 'mid-epoch loader restored'})")

    def _load_model_weights(self, path: str):
        with node_local_download(path, logger) as local_path:
            ckpt = torch.load(local_path, map_location=self.device, weights_only=False)
        self._raw_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded OmniAct model weights from {path}")
