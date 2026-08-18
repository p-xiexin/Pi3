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


def toc(label: str) -> None:
    _synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - _START_TIME)
    print(f"{label}: {elapsed_ms:.3f} ms")
