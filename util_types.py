
import typing
from dataclasses import dataclass, astuple
from dataclasses_json import DataClassJsonMixin

Rotation = typing.Literal['ROTATE_90_CLOCKWISE',
                          'ROTATE_90_COUNTERCLOCKWISE', 'ROTATE_180'] | None

Point = tuple[int, int]
Vector = tuple[int, int]
Line = tuple[int, int, int, int]  # x1,y1,x2,y2
Circle = tuple[int, int, int]  # x,y,r


@dataclass
class Rect(DataClassJsonMixin):
    x: int
    y: int
    width: int
    height: int

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class Range(DataClassJsonMixin):
    min: float
    max: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class GaugeOption(DataClassJsonMixin):
    name: str
    rect: Rect
    angles: Range
    values: Range
    location: str | None = None
    unit: str | None = None
    rotation: Rotation = None
