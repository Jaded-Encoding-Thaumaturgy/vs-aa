from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from vstools import core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser, _FullInterpolate

__all__ = [
    'EEDI2', 'Eedi2', 'Eedi2SS', 'Eedi2SR', 'Eedi2DR'
]


@dataclass
class EEDI2(_FullInterpolate, _Antialiaser):
    mthresh: int = 10
    lthresh: int = 20
    vthresh: int = 20
    estr: int = 2
    dstr: int = 4
    maxd: int = 24
    pp: int = 1

    cuda: bool = dc_field(default=False, kw_only=True)

    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.cuda and x and y

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return dict(
            mthresh=self.mthresh, lthresh=self.lthresh, vthresh=self.vthresh,
            estr=self.estr, dstr=self.dstr, maxd=self.maxd, pp=self.pp
        )

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        if self.cuda:
            interpolated = core.eedi2cuda.EEDI2(clip, self.field, **kwargs)
        else:
            interpolated = core.eedi2.EEDI2(clip, self.field, **kwargs)

        if double_y:
            return interpolated

        if self.drop_fields:
            interpolated = interpolated.std.SeparateFields(not self.field)[::2]

            return self._shifter.shift(interpolated, (0.5 - 0.75 * self.field, 0))

        shift = (self._shift * int(not self.field), 0)

        if self._scaler:
            return self._scaler.scale(interpolated, clip.width, clip.height, shift)

        return self._shifter.scale(interpolated, clip.width, clip.height, shift)

    def full_interpolate(self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any) -> vs.VideoNode:
        return core.eedi2cuda.Enlarge2(clip, **kwargs)

    _shift = -0.5


class Eedi2SS(EEDI2, SuperSampler):
    ...


class Eedi2SR(EEDI2, SingleRater):
    ...


class Eedi2DR(EEDI2, DoubleRater):
    ...


class Eedi2(EEDI2, Antialiaser):
    ...
