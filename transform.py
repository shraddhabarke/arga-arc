from enum import Enum
from typing import Union, List, Dict, Iterator, Any, Tuple, Optional

class Types(Enum):
    TRANSFORMS = "Transforms"
    COLOR = "Color"
    DIRECTION = "Dir"
    OVERLAP = "Overlap"
    ROTATION_ANGLE = "Rotation_Angle"
    SYMMETRY_AXIS = "Symmetry_Axis"
    IMAGE_POINTS = "ImagePoints"
    RELATIVE_POSITION = "RelativePosition"
    OBJECT_ID = "ObjectId"
    NO_OP = "No_Op"
    VARIABLE = "Variable"


class TransformASTNode:
    def __init__(self, children=None):
        self.code: str = self.__class__.__name__
        self.size: int = 1
        self.children: List[TransformASTNode] = children if children else []
        self.childTypes: List[Types] = []
        self.nodeType: Types

# Variable type is for variable objects


class Variable(TransformASTNode):
    nodeType = Types.VARIABLE

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.nodeType = Types.VARIABLE
        self.code = f"{self.name}"
        self.size = 1
        self.children = None
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    def apply(self, task, children, filter):
        return self


class Color(TransformASTNode, Enum):
    black = "O"
    blue = "B"
    red = "R"
    green = "G"
    yellow = "Y"
    grey = "X"
    cyan = "C"
    most = "most"
    fuchsia = "F"
    orange = "A"
    brown = "W"
    least = "least"

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

    @classmethod
    @property
    def nodeType(cls):
        return Types.COLOR

    def apply(self, task, children=None, filter=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class Dir(TransformASTNode, Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"
    UP_LEFT = "UL"
    UP_RIGHT = "UR"
    DOWN_LEFT = "DL"
    DOWN_RIGHT = "DR"

    def __init__(self, value):
        super().__init__(Types.DIRECTION)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.children = []
        self.nodeType = Types.DIRECTION
        self.values = []
        self.size = 1

    @classmethod
    @property
    def arity(cls):
        return 0

    @classmethod
    @property
    def nodeType(cls):
        return Types.DIRECTION

    def apply(self, task, children=None, filter=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class Overlap(TransformASTNode, Enum):
    TRUE = True
    FALSE = False

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

    def apply(self, task, children=None, filter=None):
        return self

    @classmethod
    @property
    def nodeType(cls):
        return Types.OVERLAP

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class Rotation_Angle(TransformASTNode, Enum):
    CCW = "90"
    CW = "270"
    CW2 = "180"

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

    @classmethod
    @property
    def nodeType(cls):
        return Types.ROTATION_ANGLE

    def apply(self, task, children=None, filter=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class Symmetry_Axis(TransformASTNode, Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    DIAGONAL_LEFT = "DIAGONAL_LEFT"
    DIAGONAL_RIGHT = "DIAGONAL_RIGHT"

    def __init__(self, value):
        super().__init__(Types.SYMMETRY_AXIS)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.nodeType = Types.SYMMETRY_AXIS
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    @classmethod
    @property
    def nodeType(cls):
        return Types.SYMMETRY_AXIS

    def apply(self, task, children=None, filter=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class ImagePoints(TransformASTNode, Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"

    def __init__(self, value):
        super().__init__(Types.IMAGE_POINTS)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.nodeType = Types.IMAGE_POINTS
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    @classmethod
    @property
    def nodeType(cls):
        return Types.IMAGE_POINTS

    def apply(self, task, children=None, filter=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class RelativePosition(TransformASTNode, Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    MIDDLE = "MIDDLE"

    def __init__(self, value):
        super().__init__(Types.RELATIVE_POSITION)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.nodeType = Types.RELATIVE_POSITION
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    @classmethod
    @property
    def nodeType(cls):
        return Types.RELATIVE_POSITION

    def apply(self, task, children=None, filter=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class ObjectIdValue:
    arity = 0

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = Types.OBJECT_ID
        self.code = f"OBJECT_ID.{enum_value.name}"
        self.size = 1
        self.children = []
        self.values = []

    def apply(cls, task, children=None, filter=None):
        return cls


class ObjectId(TransformASTNode):
    _all_values = set()
    arity = 0
    nodeType = Types.OBJECT_ID

    def __new__(cls, enum_value):
        instance = ObjectIdValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    def get_all_values(cls):
        return list(cls._enum_members)


def setup_objectids(task):
    task_ids = [id for id in range(len(task.static_objects_for_insertion[task.abstraction]))] + [
        -1]
    _id_additional = {f'{item}': int(item) for item in task_ids}
    IdEnum = Enum("IdEnum", {**_id_additional})

    _ids = []

    for name, member in IdEnum.__members__.items():
        setattr(ObjectId, name, ObjectIdValue(member))
        _ids.append(ObjectId(member))
    ObjectId._enum_members = _ids


class NoOp(TransformASTNode):
    _instance = None  # Single instance storage
    arity = 0
    nodeType = Types.NO_OP
    childTypes = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    @classmethod
    @property
    def arity(cls):
        return 0

    def __init__(self):
        if not hasattr(self, 'initialized') or not self.initialized:
            super().__init__()
            self.code = "NoOp"
            self.size = 1
            self.children = []
            self.values = []
            self.childTypes = []
            self.initialized = True

    @classmethod
    def apply(self, task, children=None, filter=None):
        self.code = "NoOp"
        self.size = 1
        self.values = task.input_abstracted_graphs_original[task.abstraction]
        return self

    @classmethod
    def get_all_values(cls):
        return [cls._instance or cls()]


class Transforms(TransformASTNode):
    nodeType = Types.TRANSFORMS
    arity = 2
    childTypes = [[Types.TRANSFORMS, Types.TRANSFORMS]]
    default_size = 1

    def __init__(self, transform1: 'Transforms' = None, transform2: 'Transforms' = None):
        super().__init__()
        self.children = [transform1,
                        transform2] if transform1 and transform2 else []
        self.size = sum(
            t.size for t in self.children) if transform1 and transform2 else 0
        self.code = "[" + ", ".join(t.code for t in self.children) + "]"
        self.nodeType = Types.TRANSFORMS
        self.childTypes = [Types.TRANSFORMS, Types.TRANSFORMS]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0], children[1])
        values = task.sequence_transform_values(
            filter, [children[0], children[1]])
        instance.values = values
        return instance


class UpdateColor(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [[Types.COLOR], [Types.VARIABLE]]
    default_size = 1

    def __init__(self, color: Union[Color, Variable]):
        super().__init__()
        self.children = [color]
        self.size = self.default_size + \
            sum(child.size for child in self.children)
        if isinstance(color, Color):
            self.code = f"updateColor({color.code})"
            self.childTypes = [Types.COLOR]
        elif isinstance(color, Variable):
            self.code = f"updateColor({color.code}.color)"
            self.childTypes = [Types.VARIABLE]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        if isinstance(children[0], Color):
            instance.values = task.transform_values(filter, instance)
        elif isinstance(children[0], Variable):
            instance.values = task.var_transform_values(filter, instance)
        return instance


class MoveNode(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [[Types.DIRECTION], [Types.VARIABLE]]
    default_size = 1

    def __init__(self, dir: Union[Dir, Variable]):
        super().__init__()
        self.children = [dir]
        self.size = self.default_size + dir.size
        if isinstance(dir, Dir):
            self.code = f"moveNode({dir.code})"
            self.childTypes = [Types.DIRECTION]
        elif isinstance(dir, Variable):
            self.code = f"moveNode({dir.code}.direction)"
            self.childTypes = [Types.VARIABLE]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        if isinstance(children[0], Dir):
            instance.values = task.transform_values(filter, instance)
        elif isinstance(children[0], Variable):
            instance.values = task.var_transform_values(filter, instance)
        return instance


class ExtendNode(Transforms):
    arity = 2
    nodeType = Types.TRANSFORMS
    childTypes = [[Types.DIRECTION, Types.OVERLAP],
                  [Types.VARIABLE, Types.OVERLAP]]
    default_size = 1

    def __init__(self, dir: Union[Dir, Variable], overlap: Overlap):
        super().__init__()
        self.children = [dir, overlap]
        self.size = self.default_size + overlap.size + dir.size
        if isinstance(dir, Dir):
            self.code = f"extendNode({dir.code}, {overlap.code})"
            self.childTypes = [Types.DIRECTION, Types.OVERLAP]
        elif isinstance(dir, Variable):
            self.code = f"extendNode({dir.code}.direction, {overlap.code})"
            self.childTypes = [Types.VARIABLE, Types.OVERLAP]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0], children[1])
        if isinstance(children[0], Dir):
            instance.values = task.transform_values(filter, instance)
        elif isinstance(children[0], Variable):
            instance.values = task.var_transform_values(filter, instance)
        return instance


class MoveNodeMax(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [[Types.DIRECTION], [Types.VARIABLE]]
    default_size = 1

    def __init__(self, dir: Union[Dir, Variable]):
        super().__init__()
        self.children = [dir]
        self.size = self.default_size + dir.size
        if isinstance(dir, Dir):
            self.code = f"moveNodeMax({dir.code})"
            self.childTypes = [Types.DIRECTION]
        elif isinstance(dir, Variable):
            self.code = f"moveNodeMax({dir.code}.direction)"
            self.childTypes = [Types.VARIABLE]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        if isinstance(children[0], Dir):
            instance.values = task.transform_values(filter, instance)
        elif isinstance(children[0], Variable):
            instance.values = task.var_transform_values(filter, instance)
        return instance


class RotateNode(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [[Types.ROTATION_ANGLE]]
    default_size = 1

    def __init__(self, rotation_angle: Rotation_Angle):
        super().__init__()
        self.children = [rotation_angle]
        self.size = self.default_size + rotation_angle.size
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
    childTypes = [[Types.COLOR]]
    default_size = 1

    def __init__(self, color: Color):
        super().__init__()
        self.children = [color]
        self.size = self.default_size + color.size
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
    childTypes = [[Types.COLOR, Types.OVERLAP]]
    default_size = 1

    def __init__(self, color: Color, overlap: Overlap):
        super().__init__()
        self.children = [color, overlap]
        self.size = self.default_size + color.size + overlap.size
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
    childTypes = [[Types.COLOR]]
    default_size = 1

    def __init__(self, color: Color):
        super().__init__()
        self.children = [color]
        self.size = self.default_size + color.size
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
    childTypes = [[Types.VARIABLE]]
    default_size = 1

    def __init__(self, mirror_axis: Variable):
        super().__init__()
        self.children = [mirror_axis]
        self.size = self.default_size + mirror_axis.size
        if isinstance(mirror_axis, Variable):
            self.code = f"mirror({mirror_axis.code}.mirror_axis)"
            self.childTypes = [Types.VARIABLE]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        if isinstance(children[0], Variable):
            instance.values = task.var_transform_values(filter, instance)
        return instance


class Flip(Transforms):
    arity = 1
    nodeType = Types.TRANSFORMS
    childTypes = [[Types.SYMMETRY_AXIS]]
    default_size = 1

    def __init__(self, mirror_direction: Symmetry_Axis):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [mirror_direction]
        self.size = self.default_size + mirror_direction.size
        self.code = f"flip({mirror_direction.code})"
        self.arity = 1
        self.childTypes = [Types.SYMMETRY_AXIS]

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance


class Insert(Transforms):
    arity = 3
    nodeType = Types.TRANSFORMS
    childTypes = [
        [Types.OBJECT_ID, Types.IMAGE_POINTS, Types.RELATIVE_POSITION]] #[Types.OBJECT_ID, Types.VARIABLE, Types.RELATIVE_POSITION] #todo
    default_size = 1

    def __init__(self, object_id: ObjectId, image_points: ImagePoints, relative_pos: RelativePosition):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = [object_id, image_points, relative_pos]
        self.childTypes = [Types.OBJECT_ID,
                        Types.IMAGE_POINTS, Types.RELATIVE_POSITION]
        self.size = self.default_size + \
            sum(child.size for child in self.children)
        self.arity = 3
        self.code = f"insert({object_id.code, image_points.code, relative_pos.code})"

    @classmethod
    def apply(cls, task, children, filter):
        instance = cls(children[0], children[1], children[2])
        values = task.transform_values(filter, instance)
        instance.values = values
        return instance
