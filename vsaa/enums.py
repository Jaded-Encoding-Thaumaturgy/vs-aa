from enum import IntEnum
from typing import Tuple

__all__ = [
    'AADirection'
]


class AADirection(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1
    BOTH = 2

    def to_yx(self) -> Tuple[bool, bool]:
        if self == AADirection.VERTICAL:
            return (True, False)

        if self == AADirection.HORIZONTAL:
            return (False, True)

        return (True, True)
