from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from vstools import core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser, _FullInterpolate

__all__ = [
    'NNEDI3', 'Nnedi3', 'Nnedi3SS', 'Nnedi3SR', 'Nnedi3DR'
]


@dataclass
class NNEDI3(_FullInterpolate, _Antialiaser):
    nsize: int = 4
    nns: int = 4
    qual: int = 2
    etype: int = 0
    pscrn: int = 2

    opencl: bool = dc_field(default=False, kw_only=True)

    def _full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.opencl

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        assert clip.format
        is_float = clip.format.sample_type == vs.FLOAT
        pscrn = 1 if is_float else self.pscrn
        return dict(nsize=self.nsize, nns=self.nns, qual=self.qual, etype=self.etype, pscrn=pscrn)

    def _interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        interpolated = core.nnedi3.nnedi3(
            clip, self.field, double_y or not self.drop_fields, **kwargs
        )

        return self._shift_interpolated(clip, interpolated, double_y)

    def _full_interpolate(self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any) -> vs.VideoNode:
        return core.nnedi3cl.NNEDI3CL(clip, self.field, double_y, double_x, **kwargs)

    _shift = 0.5


class Nnedi3SS(NNEDI3, SuperSampler):
    ...


class Nnedi3SR(NNEDI3, SingleRater):
    ...


class Nnedi3DR(NNEDI3, DoubleRater):
    ...


class Nnedi3(NNEDI3, Antialiaser):
    ...
