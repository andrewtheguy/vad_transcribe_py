"""Shared PyTorch CPU threading helpers for Hugging Face backends."""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_INTEROP_LOCK = threading.Lock()
_interop_threads: int | None = None


def prime_torch_cpu_thread_env(num_threads: int) -> int:
    """Prime common CPU thread env vars before torch initializes its backends."""
    threads = max(1, int(num_threads))
    for env_var in _CPU_THREAD_ENV_VARS:
        os.environ[env_var] = str(threads)
    return threads


def configure_torch_cpu_threads(num_threads: int) -> tuple[int, int | None]:
    """Configure PyTorch CPU threading for a single-process ASR workload."""
    import torch

    threads = prime_torch_cpu_thread_env(num_threads)
    torch.set_num_threads(threads)

    # Keep inter-op parallelism at 1 so a single inference request does not
    # oversubscribe CPU cores on top of intra-op kernel threading.
    with _INTEROP_LOCK:
        global _interop_threads
        if _interop_threads is None:
            try:
                torch.set_num_interop_threads(1)
                _interop_threads = 1
            except RuntimeError as exc:
                logger.debug("Could not set torch inter-op threads: %s", exc)
                if hasattr(torch, "get_num_interop_threads"):
                    try:
                        _interop_threads = int(torch.get_num_interop_threads())
                    except RuntimeError:
                        _interop_threads = None

    intra_threads = int(torch.get_num_threads())
    interop_threads = _interop_threads
    logger.info(
        "Configured PyTorch CPU threading: intra_op=%d, inter_op=%s",
        intra_threads,
        "unknown" if interop_threads is None else interop_threads,
    )
    return intra_threads, interop_threads
