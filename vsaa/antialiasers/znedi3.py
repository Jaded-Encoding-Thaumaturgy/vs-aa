from __future__ import annotations

from dataclasses import dataclass

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler
from . import nnedi3

__all__ = [
    'Znedi3', 'Znedi3DR'
]


@dataclass
class ZNEDI3(nnedi3.NNEDI3):
    def __post_init__(self) -> None:
        import warnings
        warnings.warn(
            'Znedi3 class is deprecated! Nnedi3 uses znedi3 by default, please use that.\n'
            'Will be removed on the next major bump.'
        )
        return super().__post_init__()


class Znedi3SS(ZNEDI3, SuperSampler):
    ...


class Znedi3SR(ZNEDI3, SingleRater):
    ...


class Znedi3DR(ZNEDI3, DoubleRater):
    ...


class Znedi3(ZNEDI3, Antialiaser):
    ...
