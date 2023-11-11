from enum import Enum
from typing import Union, List, Dict, Iterator, Any, Tuple, Optional

class Types(Enum):
    TRANSFORMS = "Transforms"
    COLOR = "Color"
    DIRECTION = "Direction"
    OVERLAP = "Overlap"
    ROTATION_ANGLE = "Rotation_Angle"
    MIRROR_AXIS = "Mirror_Axis"
    SYMMETRY_AXIS = "Symmetry_Axis"

class TransformASTNode:
    def __init__(self, children = None):
        self.code: str = self.__class__.__name__
        self.size: int = 1
        self.children: List[TransformASTNode] = children if children else []
        self.childTypes: List[Types] = []
        self.nodeType: Types

class Color(TransformASTNode, Enum):
    C0 =  0
    C1 =  1
    C2 =  2
    C3 =  3
    C4 =  4
    C5 =  5
    C6 =  6
    C7 =  7
    C8 =  8
    C9 =  9
    LEAST = "least"
    MOST =  "most"

    def __init__(self, value=None):
        super().__init__(Types.COLOR)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []  # Explicitly set children to an empty list
        self.nodeType = Types.COLOR
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    def apply(self, task, children):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Direction(TransformASTNode, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"
    nodeType = Types.DIRECTION
    def __init__(self, value):
        super().__init__(Types.DIRECTION)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.nodeType = Types.DIRECTION
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, task, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Overlap(TransformASTNode, Enum):
    TRUE = True
    FALSE = False
    nodeType = Types.OVERLAP
    def __init__(self, value):
        super().__init__(Types.OVERLAP)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.nodeType = Types.OVERLAP
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, task, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Rotation_Angle(TransformASTNode, Enum):
    CW = "270"
    CCW = "90"
    CW2 = "180"
    nodeType = Types.ROTATION_ANGLE
    def __init__(self, value):
        super().__init__(Types.ROTATION_ANGLE)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.nodeType = Types.ROTATION_ANGLE
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, task, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Symmetry_Axis(TransformASTNode, Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    DIAGONAL_LEFT = "DIAGONAL_LEFT"
    DIAGONAL_RIGHT = "DIAGONAL_RIGHT"
    nodeType = Types.SYMMETRY_AXIS
    def __init__(self, value):
        super().__init__(Types.SYMMETRY_AXIS)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, task, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Mirror_Axis(TransformASTNode, Enum): # TODO: fix the semantics of Mirror_Axis
    X_AXIS = "X_AXIS"
    Y_AXIS = "Y_AXIS"
    nodeType = Types.MIRROR_AXIS
    def __init__(self, value):
        super().__init__(Types.MIRROR_AXIS)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, task, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Transforms(TransformASTNode):
    nodeType = Types.TRANSFORMS
    arity = 2
    childTypes = [Types.TRANSFORMS, Types.TRANSFORMS]
    def __init__(self, transform1: 'Transforms' = None, transform2: 'Transforms' = None):
        super().__init__()
        self.children = [transform1, transform2] if transform1 and transform2 else []
        self.size = sum(t.size for t in self.children) if transform1 and transform2 else 0
        self.code = "[" + ", ".join(t.code for t in self.children) + "]" if transform1 and transform2 else ''
        self.nodeType = Types.TRANSFORMS
        self.childTypes = [Types.TRANSFORMS, Types.TRANSFORMS]

    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class NoOp(Transforms):
    _instance = None  # Single instance storage
    arity = 0
    nodeType = Types.TRANSFORMS

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NoOp, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized') or not self.initialized:
            super(Transforms, self).__init__()
            self.code = "NoOp"
            self.size = 1
            self.children = []
            self.initialized = True

    def apply(self, task, children=None):
        return self

    @classmethod
    def get_all_values(cls):
        return [cls._instance or cls()]

class UpdateColor(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.COLOR]
    def __init__(self, color: Color):
        super().__init__()
        self.children = [color]
        self.size = 1 + color.size
        self.code = f"updateColor({color.code})"
        self.childTypes = [Types.COLOR]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class MoveNode(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.DIRECTION]
    def __init__(self, direction: Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [direction]
        self.size = 1 + direction.size
        self.code = f"moveNode({direction.code})"
        self.childTypes = [Types.DIRECTION]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class ExtendNode(Transforms):
    arity = 2
    nodeType = Types.TRANSFORMS
    childTypes = [Types.DIRECTION, Types.OVERLAP]
    def __init__(self, direction: Direction, overlap: Overlap):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [direction, overlap]
        self.size = 1 + direction.size + overlap.size
        self.code = f"extendNode({direction.code}, {overlap.code})"
        self.childTypes = [Types.DIRECTION, Types.OVERLAP]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0], children[1])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance
    
class MoveNodeMax(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.DIRECTION]
    def __init__(self, direction: Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [direction]
        self.size = 1 + direction.size
        self.code = f"moveNodeMax({direction.code})"
        self.childTypes = [Types.DIRECTION]
    
    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class RotateNode(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.ROTATION_ANGLE]
    def __init__(self, rotation_angle: Rotation_Angle):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [rotation_angle]
        self.size = 1 + rotation_angle.size
        self.code = f"rotateNode({rotation_angle.code})"
        self.childTypes = [Types.ROTATION_ANGLE]
    
    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class AddBorder(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.COLOR]
    def __init__(self, color: Color):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [color]
        self.size = 1 + color.size
        self.code = f"addBorder({color.code})"
        self.childTypes = [Types.COLOR]
    
    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class FillRectangle(Transforms):
    arity = 2
    nodeType = Types.TRANSFORMS
    childTypes = [Types.COLOR, Types.OVERLAP]
    def __init__(self, color: Color, overlap: Overlap):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [color, overlap]
        self.size = 1 + color.size + overlap.size
        self.code = f"fillRectangle({color.code}, {overlap.code})"
        self.childTypes = [Types.COLOR, Types.OVERLAP]
    
    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0], children[1])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class HollowRectangle(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.COLOR]
    def __init__(self, color: Color):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [color]
        self.size = 1 + color.size
        self.code = f"hollowRectangle({color.code})"
        self.childTypes = [Types.COLOR]
    
    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class Mirror(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.MIRROR_AXIS]
    def __init__(self, mirror_axis: Mirror_Axis):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [mirror_axis]
        self.size = 1 + mirror_axis.size
        self.code = f"mirror({mirror_axis.code})"
        self.childTypes = [Types.MIRROR_AXIS]
        self.arity = 1
    
    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance

class Flip(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [Types.SYMMETRY_AXIS]
    def __init__(self, mirror_direction: Symmetry_Axis):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [mirror_direction]
        self.size = 1 + mirror_direction.size
        self.code = f"flip({mirror_direction.code})"
        self.arity = 1
        self.childTypes = [Types.SYMMETRY_AXIS]
    
    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance