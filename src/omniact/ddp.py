"""Distributed Data Parallel (DDP) utilities.

All training is launched via ``torchrun`` (even single-GPU uses
``torchrun --nproc_per_node=1``), so the process group is always available.
``setup_ddp`` picks ``nccl``+``cuda:LOCAL_RANK`` when there are enough GPUs
for every local rank, else falls the whole node back to ``gloo``+``cpu``.
"""

import contextlib
import hashlib
import logging
import os
import tempfile
from collections.abc import Iterator
from datetime import timedelta

import torch
import torch.distributed as dist

from .fsspec_retry import retry_fsspec_download_forever


def setup_ddp() -> str:
    """Initialize the process group and resolve this rank's device.

    Always launched via torchrun, so LOCAL_RANK / LOCAL_WORLD_SIZE env vars
    are expected.  Returns the resolved device string (``cuda:LOCAL_RANK`` or
    ``cpu``) so callers can wire it directly into config without recomputing.

    All local ranks on a node agree on the device by comparing GPU count to
    ``LOCAL_WORLD_SIZE`` — if there aren't enough GPUs for every rank, the
    whole node falls back to CPU (with gloo).  This covers CUDA-less dev
    boxes and multi-rank-on-fewer-GPUs smoke tests under one rule.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    if torch.cuda.is_available() and torch.cuda.device_count() >= local_world_size:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(timeout=timedelta(days=1))
        return f"cuda:{local_rank}"

    # CPU fallback must request gloo explicitly.
    dist.init_process_group(backend="gloo", timeout=timedelta(days=1))
    return "cpu"


def cleanup_ddp() -> None:
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def get_rank() -> int:
    """Global rank."""
    return dist.get_rank()


def get_local_rank() -> int:
    """Local rank on this node."""
    return int(os.environ["LOCAL_RANK"])


def get_world_size() -> int:
    """Total number of processes."""
    return dist.get_world_size()


def is_main_process() -> bool:
    """True on rank 0."""
    return get_rank() == 0


@contextlib.contextmanager
def node_local_download(
    remote_path: str, logger: logging.Logger | None = None
) -> Iterator[str]:
    """Download a remote file once per node to a node-local path.

    Local-rank-0 on each node fetches the file (via fsspec) to a
    deterministic path under ``$SLURM_TMPDIR`` (or the system temp dir),
    other ranks wait at a barrier, then all ranks may read the local
    path. On exit, a second barrier ensures every rank has finished
    reading before local-rank-0 removes the file.
    """
    tmp_dir = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    key = hashlib.md5(remote_path.encode()).hexdigest()[:16]
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    local_path = os.path.join(tmp_dir, f"vv_ckpt_{jobid}_{key}.pt")

    if get_local_rank() == 0:
        if logger is not None:
            logger.info(f"Downloading {remote_path} -> {local_path} (local rank 0)")
        retry_fsspec_download_forever(
            remote_path,
            local_path,
            description=f"node-local checkpoint {remote_path}",
        )

    dist.barrier()
    try:
        yield local_path
    finally:
        dist.barrier()
        if get_local_rank() == 0:
            try:
                os.remove(local_path)
            except OSError:
                pass
