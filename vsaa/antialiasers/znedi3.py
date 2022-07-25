from dataclasses import dataclass
from typing import Any

import vapoursynth as vs

from vsaa.antialiasers.nnedi3 import NNEDI3

from ..abstract import _Antialiaser, Antialiaser, DoubleRater, SingleRater, SuperSampler

__all__ = ['Znedi3', 'Znedi3SS', 'Znedi3SR', 'Znedi3DR']

core = vs.core


@dataclass
class ZNEDI3(_Antialiaser):
    nsize: int = 4
    nns: int = 4
    qual: int = 2
    etype: int = 0
    pscrn: int = 2

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return NNEDI3.get_aa_args(self, clip, **kwargs)  # type: ignore

    def _interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        return core.znedi3.nnedi3(clip, self.field, double_y, **kwargs)

    _shift = 0.5


class Znedi3SS(ZNEDI3, SuperSampler):
    ...


class Znedi3SR(ZNEDI3, SingleRater):
    ...


class Znedi3DR(ZNEDI3, DoubleRater):
    ...


class Znedi3(NNEDI3, Antialiaser):
    ...
