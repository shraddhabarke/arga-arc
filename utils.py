from enum import Enum
import typing as t

TASK_IDS_TYPE = t.Literal[
    "08ed6ac7",
    "1e0a9b12",
    "25ff71a9",
    "3906de3d",
    "4258a5f9",
    "50cb2852",
    "543a7ed5",
    "6455b5f5",
    "67385a82",
    "694f12f3",
    "6e82a1ae",
    "7f4411dc",
    "a79310a0",
    "aedd82e4",
    "b1948b0a",
    "b27ca6d3",
    "bb43febb",
    "c8f0f002",
    "d2abd087",
    "dc1df850",
    "ea32f347",
]
TASK_IDS: t.List[TASK_IDS_TYPE] = list(t.get_args(TASK_IDS_TYPE))


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


class Rotation(str, Enum):
    CW = "CW"
    CCW = "CCW"
    CW2 = "CW2"


class Mirror(str, Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    DIAGONAL_LEFT = "DIAGONAL_LEFT"  # \
    DIAGONAL_RIGHT = "DIAGONAL_RIGHT"  # /


class ImagePoints(str, Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


class RelativePosition(str, Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    MIDDLE = "MIDDLE"


class ObjectProperty(Enum):
    SYMMETRICAL = 0
    HOLLOW = 1
