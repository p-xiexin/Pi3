import time

import torch


_START_TIME = 0.0


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic() -> None:
    global _START_TIME
    _synchronize()
    _START_TIME = time.perf_counter()


def toc(label: str | None = None) -> float:
    _synchronize()
    elapsed_seconds = time.perf_counter() - _START_TIME
    if label is not None:
        print(f"{label}: {1000.0 * elapsed_seconds:.3f} ms")
    return elapsed_seconds
