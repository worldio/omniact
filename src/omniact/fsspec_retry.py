"""Retry helpers for idempotent fsspec reads used by training datasets."""

from __future__ import annotations

import contextlib
import os
import random
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


class FsspecOperationTimeout(TimeoutError):
    """Raised when one fsspec operation attempt exceeds its local deadline."""


@dataclass(frozen=True)
class FsspecRetryConfig:
    attempt_timeout_s: float = 120.0
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    jitter_s: float = 1.0

    @classmethod
    def from_env(cls) -> "FsspecRetryConfig":
        return cls(
            attempt_timeout_s=float(os.getenv("OMNIACT_FSSPEC_ATTEMPT_TIMEOUT_S", "120")),
            initial_backoff_s=float(os.getenv("OMNIACT_FSSPEC_INITIAL_BACKOFF_S", "1")),
            max_backoff_s=float(os.getenv("OMNIACT_FSSPEC_MAX_BACKOFF_S", "60")),
            jitter_s=float(os.getenv("OMNIACT_FSSPEC_JITTER_S", "1")),
        )

    @classmethod
    def download_from_env(cls) -> "FsspecRetryConfig":
        return cls(
            attempt_timeout_s=float(os.getenv("OMNIACT_FSSPEC_DOWNLOAD_TIMEOUT_S", "1800")),
            initial_backoff_s=float(os.getenv("OMNIACT_FSSPEC_INITIAL_BACKOFF_S", "1")),
            max_backoff_s=float(os.getenv("OMNIACT_FSSPEC_MAX_BACKOFF_S", "60")),
            jitter_s=float(os.getenv("OMNIACT_FSSPEC_JITTER_S", "1")),
        )


def fsspec_storage_options() -> dict:
    """Options for a fresh fsspec filesystem in a DataLoader worker."""
    return {"skip_instance_cache": True}


def make_fsspec_filesystem(path: str):
    import fsspec

    fs, _, _ = fsspec.get_fs_token_paths(path, storage_options=fsspec_storage_options())
    return fs


def retry_fsspec_operation_forever(
    operation: Callable[[], T],
    *,
    reset: Callable[[], None],
    description: str,
    config: FsspecRetryConfig,
) -> T:
    """Run an idempotent fsspec read until it succeeds or fails permanently.

    The operation must be safe to repeat. For datasets, call this around one
    sample read so each retry returns the same key/index.
    """
    attempt = 0
    backoff = config.initial_backoff_s

    while True:
        attempt += 1
        try:
            with _operation_deadline(config.attempt_timeout_s):
                return operation()
        except _permanent_errors():
            raise
        except BaseException as exc:
            # Default is retry. Set OMNIACT_FSSPEC_RAISE_ON_UNKNOWN=1 to fail
            # fast on the unrecognised instead.
            raise_on_unknown = os.getenv(
                "OMNIACT_FSSPEC_RAISE_ON_UNKNOWN", "0"
            ).lower() in ("1", "true")
            if raise_on_unknown and not _is_retryable_fsspec_error(exc):
                raise
            reset()
            sleep_s = backoff + random.random() * config.jitter_s
            logger.warning(
                "Retrying fsspec read after failure: {} attempt={} sleep={:.1f}s error={!r}",
                description,
                attempt,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)
            backoff = min(backoff * 2, config.max_backoff_s)


def retry_fsspec_download_forever(
    remote_path: str,
    local_path: str | os.PathLike[str],
    *,
    description: str | None = None,
    config: FsspecRetryConfig | None = None,
) -> None:
    """Download a fsspec path, retrying transient failures forever."""
    local = os.fspath(local_path)
    fs_holder: dict[str, object] = {}

    def _get_fs():
        fs = fs_holder.get("fs")
        if fs is None:
            fs = make_fsspec_filesystem(remote_path)
            fs_holder["fs"] = fs
        return fs

    def _reset() -> None:
        close_fsspec_filesystem(fs_holder.pop("fs", None))
        with contextlib.suppress(OSError):
            os.remove(local)

    def _download() -> None:
        fs = _get_fs()
        fs.get(remote_path, local)

    try:
        retry_fsspec_operation_forever(
            _download,
            reset=_reset,
            description=description or f"download {remote_path}",
            config=config or FsspecRetryConfig.download_from_env(),
        )
    finally:
        close_fsspec_filesystem(fs_holder.pop("fs", None))


def close_fsspec_filesystem(fs: object | None) -> None:
    if fs is None:
        return

    close = getattr(fs, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()

    clear_cache = getattr(type(fs), "clear_instance_cache", None)
    if callable(clear_cache):
        with contextlib.suppress(Exception):
            clear_cache()


@contextlib.contextmanager
def _operation_deadline(seconds: float | None):
    if (
        not seconds
        or os.name != "posix"
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _handle_timeout(_signum: int, _frame: object) -> None:
        raise FsspecOperationTimeout(f"fsspec operation exceeded {seconds:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _permanent_errors() -> tuple[type[BaseException], ...]:
    return FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError, KeyError


def _is_retryable_fsspec_error(exc: BaseException) -> bool:
    if isinstance(exc, (FsspecOperationTimeout, TimeoutError, OSError)):
        return True

    try:
        from fsspec.exceptions import FSTimeoutError

        if isinstance(exc, FSTimeoutError):
            return True
    except Exception:
        pass

    module = type(exc).__module__
    if module.startswith(("aiohttp", "requests", "urllib3")):
        return True

    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "loop is not running",
                "event loop is closed",
                "session is closed",
                "timeout",
                "fork-safe",
            )
        )

    return False
