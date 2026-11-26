"""Train the morphology-agnostic OmniAct policy on prepared MolmoSpaces WODS.

Each key is ``gen_<seed>/<task-config>/<scene>_house_<h>_traj<NN>`` and carries node geometry,
graph, proprioception, and relative actions. Embodiments share every model weight.
See docs/omniact_pipeline.md for text embedding and video chunk preparation.

Usage:
    PYTHONUNBUFFERED=1 uv run torchrun --standalone --nproc-per-node=8 scripts/train_omniact.py \\
        --data "$TRAIN_WODS_ROOT" --window-size 32 --chunk-frames 16 \\
        --model-size 300M --batch-size 16 \\
        --checkpoint-base <checkpoint root> --run-name omniact_001

    # Local smoke (tiny random-SigLIP, no wandb):
    PYTHONUNBUFFERED=1 uv run torchrun --standalone --nproc-per-node=1 scripts/train_omniact.py \\
        --data "$TRAIN_WODS_ROOT" \\
        --policy-dim 128 --policy-blocks 6 --policy-heads 4 \\
        --window-size 8 --batch-size 4 --max-steps 5 --no-wandb --siglip-random-init
"""
import argparse
import sys

from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OmniAct policy (P1)")
    p.add_argument("--data", required=True, help="Node-graph WODS dataset (geometry + relative action)")
    p.add_argument("--tasks", nargs="+", default=None, help="Restrict to task-config name prefixes")
    p.add_argument("--window-size", type=int, default=32, help="Sequence window; action chunk = window-1")
    p.add_argument("--chunk-frames", type=int, default=16, help="Frames per video shard")
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--clip-sigma", type=float, default=None,
                   help="If set, clamp the normalized per-node actions to [-clip_sigma, clip_sigma].")
    p.add_argument("--action-mag-mask-sigma", type=float, default=None,
                   help="If set, drop from the loss any action node whose z-scored relative target exceeds this "
                        "many σ (filters out steps that move too much); None => train on all action nodes.")
    p.add_argument("--stats-sample-size", type=int, default=0,
                   help=">0: refit joint/pose RMS scales from N training keys after removing the step ramps; "
                        "keep the pose origin and ramp powers fixed.")
    p.add_argument("--subsample-train-embodiments", type=int, default=0,
                    help="Train on exactly N post-split embodiments (0 = all); with the same "
                         "--split-seed, smaller N sets are subsets of larger ones. "
                         "Validation categories reflect the selected training exposure.")
    p.add_argument("--trajectory-fraction", type=float, default=1.0,
                    help="Keep this fraction of each training embodiment's trajectories "
                         "(1.0 = all). Complements --subsample-train-embodiments: it shrinks "
                         "the corpus while holding the embodiment count fixed. With the same "
                         "--split-seed, smaller fractions are subsets of larger ones.")

    # Model
    p.add_argument("--policy-dim", type=int, default=768)
    p.add_argument("--policy-blocks", type=int, default=12)
    p.add_argument("--policy-heads", type=int, default=12)
    p.add_argument("--model-size", type=str, default=None,
                   help="Target core size (e.g. 300M) via the transfer scaler; overrides --policy-dim/blocks/heads.")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--align-start", type=int, default=None)
    p.add_argument("--align-end", type=int, default=None)
    p.add_argument("--n-node", type=int, default=None,
                   help="Node-graph frame width / id_embed size (default: MAX_NODES=12; variable-N "
                        "embodiments are right-aligned into it).")
    p.add_argument("--n-points", type=int, default=2)
    p.add_argument("--fourier-bands", type=int, default=8)
    p.add_argument("--max-spd", type=int, default=8)
    p.add_argument("--shared-view-geometry", action="store_true",
                   help="Ablation: give every view view-0's node coordinates, so geometry is no longer "
                        "registered to the view it accompanies. Coordinate statistics are unchanged; only "
                        "the per-view correspondence is removed. Contrast with --no-geometry, which "
                        "removes the geometry entirely.")
    p.add_argument("--no-geometry", action="store_true",
                   help="Ablation: drop the per-node geometry path (proprio + id_embed only; proprio stays on).")
    p.add_argument("--siglip-model", type=str, default="google/siglip2-base-patch16-naflex")
    p.add_argument("--no-freeze-siglip", action="store_true")
    p.add_argument("--siglip-random-init", action="store_true", help="Test only: random SigLIP weights")
    p.add_argument("--text-embed-dim", type=int, default=3584)
    p.add_argument("--max-views", type=int, default=3)
    p.add_argument("--max-action-len", type=int, default=64)
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--denoise-steps", type=int, default=10)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pose-loss-weight", type=float, default=1.0,
                   help="Weight on the TCP pose flow loss added to the joint flow loss.")
    p.add_argument("--val-interval", type=int, default=0,
                   help="Run offline validation every N steps; 0 (default) never runs validation.")
    p.add_argument("--num-workers", type=int, default=54)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the train step (validation stays eager). Off by default.")
    p.add_argument("--compile-mode", type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode (default: default).")
    p.add_argument("--ddp-bucket-cap-mb", type=int, default=400, help="DDP gradient bucket size.")

    # Checkpoint / logging
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint-base", type=str, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--load-model-from-ckpt", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="omniact")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for training RNGs; does not affect data split membership (default: 42)",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used only for deterministic scene/embodiment split membership (default: 42)",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    import random

    import numpy as np
    import torch
    from wods.v2.item import list_wods_keys

    from omniact.ddp import cleanup_ddp, is_main_process, setup_ddp
    from omniact.dataset import (
        MAX_NODES,
        OmniActDataset,
    )
    from omniact.split import (
        SPLIT_ALGORITHM,
        VAL_UNSEEN_BOTH,
        VAL_UNSEEN_EMBODIMENT,
        VAL_UNSEEN_SCENE,
        _split_key,
        key_set_sha256,
        split_train_val_keys,
    )
    from omniact.model import OmniActModelConfig
    from omniact.trainer import OmniActTrainConfig, OmniActTrainer

    args.device = setup_ddp()
    if args.checkpoint_base is not None and args.run_name is None:
        logger.error("--checkpoint-base requires --run-name")
        sys.exit(1)
    checkpoint_output = (f"{args.checkpoint_base.rstrip('/')}/{args.run_name}"
                         if (args.run_name and args.checkpoint_base) else None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # n_node: the fixed node-graph frame width (id_embed / action_id_embed / action_std size). Default
    # MAX_NODES (8-arm ceiling); variable-N embodiments (10/11/12) are right-aligned into it by the dataset.
    n_node = args.n_node if args.n_node is not None else MAX_NODES

    # Reserve axes over the full universe before task filtering and subsampling.
    # Validation categories use actual exposure in the final training keys.
    all_keys = list(list_wods_keys(args.data))
    key_split = split_train_val_keys(
        all_keys, args.split_seed, tasks=args.tasks,
        subsample_train_embodiments=args.subsample_train_embodiments,
        train_trajectory_fraction=args.trajectory_fraction,
    )
    selected_train_embodiments = {_split_key(key)[1] for key in key_split.train_keys}

    dataset = OmniActDataset(
        path=args.data, image_size=tuple(args.image_size), window_size=args.window_size,
        normalize_actions=True, text_embed_dim=args.text_embed_dim,
        tasks=args.tasks, read_text=not args.no_text,
        clip_sigma=args.clip_sigma, n_node=n_node,
        chunk_frames=args.chunk_frames, use_image_augmentation=True,
        shared_view_geometry=args.shared_view_geometry,
        action_mag_mask_sigma=args.action_mag_mask_sigma,
    )
    del all_keys
    logger.info(f"n_node={n_node}, embodiments={dataset.embodiments}")
    logger.info(
        f"Training embodiments: requested={args.subsample_train_embodiments or 'all'} "
        f"selected={len(selected_train_embodiments)} "
        f"trajectory_fraction={args.trajectory_fraction} "
        f"train_keys={len(key_split.train_keys)} split_seed={args.split_seed}"
    )

    # --model-size auto-scales dim/blocks/heads via the transfer parameter curve.
    pdim, pblocks, pheads = args.policy_dim, args.policy_blocks, args.policy_heads
    if args.model_size:
        from omniact.blocks import scale_transfer_config
        _sc = scale_transfer_config(args.model_size, siglip_dim=768, text_embed_dim=args.text_embed_dim,
                                    max_views=args.max_views, max_action_len=args.max_action_len)
        pdim, pblocks, pheads = _sc.policy_dim, _sc.policy_blocks, _sc.policy_heads
        logger.info(f"--model-size {args.model_size} -> dim={pdim} blocks={pblocks} heads={pheads}")

    policy_config = OmniActModelConfig(
        policy_dim=pdim, policy_blocks=pblocks, policy_heads=pheads,
        dropout=args.dropout, align_start=args.align_start, align_end=args.align_end,
        n_node=n_node, n_points=args.n_points, fourier_bands=args.fourier_bands, max_spd=args.max_spd,
        siglip_model=args.siglip_model, freeze_siglip=not args.no_freeze_siglip,
        siglip_random_init=args.siglip_random_init, denoise_steps=args.denoise_steps,
        max_views=args.max_views, max_action_len=args.max_action_len,
        action_horizon=args.window_size - 1, text_embed_dim=args.text_embed_dim,
        action_clip_sigma=args.clip_sigma, use_geometry=not args.no_geometry,
    )
    logger.info(f"Model: dim={policy_config.policy_dim}, blocks={policy_config.policy_blocks}, "
                f"heads={policy_config.policy_heads}, p,q={policy_config.resolve_pq()}")

    train_idx = [i for i, key in enumerate(dataset.items) if key in key_split.train_keys]
    val_indices = {
        name: [i for i, key in enumerate(dataset.items) if key in keys]
        for name, keys in key_split.validation_keys.items()
    }
    if not train_idx:
        raise ValueError(f"Global split leaves no training keys for tasks={args.tasks}")
    empty_val_splits = [name for name, indices in val_indices.items() if not indices]
    if empty_val_splits:
        raise ValueError(
            f"Global split leaves empty validation sets {empty_val_splits} for tasks={args.tasks}"
        )
    if args.stats_sample_size > 0:
        dataset.update_action_std([
            dataset.items[index] for index in train_idx[:args.stats_sample_size]
        ])
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_datasets = {
        name: torch.utils.data.Subset(dataset, indices)
        for name, indices in val_indices.items()
    }
    train_key_hash = key_set_sha256(dataset.items[index] for index in train_idx)
    if is_main_process():
        validation_keys = key_split.validation_keys
        validation_embodiments = sorted({
            _split_key(key)[1]
            for key in validation_keys[VAL_UNSEEN_EMBODIMENT] | validation_keys[VAL_UNSEEN_BOTH]
        })
        validation_scenes = sorted({
            _split_key(key)[0]
            for key in validation_keys[VAL_UNSEEN_SCENE] | validation_keys[VAL_UNSEEN_BOTH]
        })
        logger.info(
            f"Generated training set: count={len(train_idx)} "
            f"embodiments={len(selected_train_embodiments)} sha256={train_key_hash} "
            f"split_seed={args.split_seed} algorithm={SPLIT_ALGORITHM}"
        )
        logger.info(f"Validation embodiments ({len(validation_embodiments)}): {validation_embodiments}")
        logger.info(f"Validation scenes ({len(validation_scenes)}): {validation_scenes}")
        for name, keys in validation_keys.items():
            logger.info(f"Validation {name}: count={len(keys)} sha256={key_set_sha256(keys)}")

    train_config = OmniActTrainConfig(
        policy=policy_config, batch_size=args.batch_size, num_epochs=args.num_epochs,
        max_steps=args.max_steps, learning_rate=args.lr, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip, pose_loss_weight=args.pose_loss_weight, num_workers=args.num_workers,
        persistent_workers=args.persistent_workers, val_interval=args.val_interval,
        use_wandb=not args.no_wandb, wandb_project=args.wandb_project, wandb_run_name=args.run_name,
        checkpoint_output=checkpoint_output, checkpoint_interval=args.checkpoint_interval,
        load_model_from_ckpt=args.load_model_from_ckpt, device=args.device,
        compile=args.compile, compile_mode=args.compile_mode,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
    )

    trainer = OmniActTrainer(
        train_config,
        train_dataset=train_ds,
        val_datasets=val_datasets,
    )
    trainer.train()
    cleanup_ddp()


if __name__ == "__main__":
    main()
