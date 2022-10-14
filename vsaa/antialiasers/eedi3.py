from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from vstools import core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser
from .nnedi3 import Nnedi3

__all__ = [
    'EEDI3', 'Eedi3', 'Eedi3SS', 'Eedi3SR', 'Eedi3DR'
]


@dataclass
class EEDI3(_Antialiaser):
    alpha: float = 0.25
    beta: float = 0.5
    gamma: float = 40
    nrad: int = 2
    mdis: int = 20

    opencl: bool = dc_field(default=False, kw_only=True)

    mclip: vs.VideoNode | None = None
    sclip_aa: type[Antialiaser] | Antialiaser | None = Nnedi3(nsize=4, nns=4, qual=2, etype=1)

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma, nrad=self.nrad, mdis=self.mdis)

    def _interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        if not isinstance(self.sclip_aa, Antialiaser):
            self.sclip_aa = self.sclip_aa()

        if self.sclip_aa:
            sclip_args = self.sclip_aa.get_aa_args(clip, **kwargs)
            if double_y:
                sclip_args |= self.sclip_aa.get_ss_args(clip, **kwargs)
            else:
                sclip_args |= self.sclip_aa.get_sr_args(clip, **kwargs)

            sclip = self.sclip_aa._interpolate(clip, double_y or not self.drop_fields)
        else:
            sclip = None

        interpolated = core.eedi3m.EEDI3(
            clip, self.field, double_y or not self.drop_fields, sclip=sclip, **kwargs
        )

        return self._shift_interpolated(clip, interpolated, double_y)

    _shift = 0.5


class Eedi3SS(EEDI3, SuperSampler):
    ...


class Eedi3SR(EEDI3, SingleRater):
    _mclips: tuple[vs.VideoNode, vs.VideoNode] | None = None

    def get_sr_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        if not self.mclip:
            return {}

        if not self._mclips:
            self._mclips = (self.mclip, self.mclip.std.Transpose())

        if self.mclip.width == clip.width and self.mclip.height == clip.height:
            return dict(mclip=self._mclips[0])
        else:
            return dict(mclip=self._mclips[1])


class Eedi3DR(EEDI3, DoubleRater):
    ...


class Eedi3(Eedi3SR, Antialiaser):
    ...
