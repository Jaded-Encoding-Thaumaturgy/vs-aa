from __future__ import annotations

from vstools import CustomIntEnum

__all__ = [
    'AADirection'
]


class AADirection(CustomIntEnum):
    VERTICAL = 0
    HORIZONTAL = 1
    BOTH = 2

    def to_yx(self) -> tuple[bool, bool]:
        if self == AADirection.VERTICAL:
            return (True, False)

        if self == AADirection.HORIZONTAL:
            return (False, True)

        return (True, True)
