from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vstools import core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser
from .nnedi3 import NNEDI3

__all__ = [
    'ZNEDI3', 'Znedi3', 'Znedi3SS', 'Znedi3SR', 'Znedi3DR'
]


@dataclass
class ZNEDI3(_Antialiaser):
    nsize: int = 4
    nns: int = 4
    qual: int = 2
    etype: int = 0
    pscrn: int = 2

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return NNEDI3.get_aa_args(self, clip, **kwargs)  # type: ignore

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        interpolated = core.znedi3.nnedi3(
            clip, self.field, double_y or not self.drop_fields, **kwargs
        )

        return self.shift_interpolate(clip, interpolated, double_y)

    _shift = 0.5


class Znedi3SS(ZNEDI3, SuperSampler):
    ...


class Znedi3SR(ZNEDI3, SingleRater):
    ...


class Znedi3DR(ZNEDI3, DoubleRater):
    ...


class Znedi3(ZNEDI3, Antialiaser):
    ...
