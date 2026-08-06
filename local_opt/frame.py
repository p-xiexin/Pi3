"""Frame views and batched frame state for single-window SfM."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class Frame:
    """Bind one image and its camera-frame geometry to one SE(3) state."""

    frame_id: int
    I: torch.Tensor
    X_C: torch.Tensor
    C: torch.Tensor
    K: torch.Tensor
    delta: torch.Tensor
    T_WC: torch.Tensor

    @property
    def D(self) -> torch.Tensor:
        """Return the point-map depth in camera coordinates."""

        return self.X_C[..., 2]


@dataclass(frozen=True)
class Frames:
    """Own the batched tensors for one image window.

    Indexing returns a lightweight ``Frame`` view. Batched algorithms operate
    directly on these tensors, so per-frame debugging does not introduce a
    second copy or require stacking Python objects back into tensors.
    """

    Is: torch.Tensor
    Xs_C: torch.Tensor
    Cs: torch.Tensor
    Ks: torch.Tensor
    deltas: torch.Tensor
    T_WCs: torch.Tensor

    def __len__(self) -> int:
        return self.Is.shape[0]

    def __getitem__(self, index: int) -> Frame:
        return Frame(
            frame_id=int(index),
            I=self.Is[index],
            X_C=self.Xs_C[index],
            C=self.Cs[index],
            K=self.Ks[index],
            delta=self.deltas[index],
            T_WC=self.T_WCs[index],
        )

    def with_optimization(
        self,
        T_WCs: torch.Tensor,
        Xs_C: torch.Tensor | None = None,
    ) -> Frames:
        """Return the window with optimized poses and optional point-map depths."""

        return replace(
            self,
            T_WCs=T_WCs,
            Xs_C=self.Xs_C if Xs_C is None else Xs_C,
        )


__all__ = ["Frame", "Frames"]
