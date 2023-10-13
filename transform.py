from enum import Enum
from typing import Union, List, Dict, Iterator, Any

class Types(Enum):
    TRANSFORMS = "Transforms"
    TRANSFORM_OPS = "TransformOps"
    COLOR = "Color"
    DIRECTION = "Direction"
    OVERLAP = "Overlap"
    ROTATION_DIRECTION = "Rotation_Direction"
    FILLCOLOR = "FillColor"
    MIRROR_AXIS = "Mirror_Axis"
    MIRROR_DIRECTION = "Mirror_Direction"
    RELATIVE_POS = "Relative_Pos"
    IMAGEPOINTS = "ImagePoints"

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
        self.nodeType = Types.COLOR
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []  # Explicitly set children to an empty list

    @classmethod
    @property
    def arity(cls):
        return 0

    def apply(self, children=None):
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
    def __init__(self, value):
        super().__init__(Types.DIRECTION)
        self.nodeType = Types.DIRECTION
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = [] 

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Overlap(TransformASTNode, Enum):
    TRUE = True
    FALSE = False
    def __init__(self, value):
        super().__init__(Types.OVERLAP)
        self.nodeType = Types.OVERLAP
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = [] 

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Rotation_Direction(TransformASTNode, Enum):
    CW = "CW"
    CCW = "CCW"
    CW2 = "CW2"
    def __init__(self, value):
        super().__init__(Types.ROTATION_DIRECTION)
        self.nodeType = Types.ROTATION_DIRECTION
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = [] 

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class FillColor(TransformASTNode, Enum):
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
    def __init__(self, value):
        super().__init__(Types.FILLCOLOR)
        self.nodeType = Types.FILLCOLOR
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = [] 

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Mirror_Direction(TransformASTNode, Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    DIAGONAL_LEFT = "DIAGONAL_LEFT"
    DIAGONAL_RIGHT = "DIAGONAL_RIGHT"
    def __init__(self, value):
        super().__init__(Types.MIRROR_DIRECTION)
        self.nodeType = Types.MIRROR_DIRECTION
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = [] 

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class Mirror_Axis(TransformASTNode, Enum):
    X_AXIS = "X_AXIS"
    Y_AXIS = "Y_AXIS"
    def __init__(self, value):
        super().__init__(Types.MIRROR_AXIS)
        self.nodeType = Types.MIRROR_AXIS
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = [] 

    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, children=None):
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
        super().__init__(Types.IMAGEPOINTS)
        self.nodeType = Types.IMAGEPOINTS
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []

    @classmethod
    @property
    def arity(cls):
        return 0

    def apply(self, children=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class RelativePosition(TransformASTNode, Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    MIDDLE = "MIDDLE"
    def __init__(self, value):
        super().__init__(Types.RELATIVE_POS)
        self.nodeType = Types.RELATIVE_POS
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = [] 
    
    @classmethod
    @property
    def arity(cls):
        return 0
    
    def apply(self, children=None):
        return self
    
    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

class TransformOps(TransformASTNode):
    def __init__(self):
        super().__init__(Types.TRANSFORM_OPS)

class Transforms(TransformASTNode):
    nodeType = Types.TRANSFORMS
    def __init__(self, transform_list: List[TransformASTNode]):
        super().__init__()
        self.nodeType = Types.TRANSFORMS
        self.children = transform_list
        self.size = sum(t.size for t in transform_list)  # 0 for Transforms node + total of all child nodes
        self.code = "[" + ", ".join(t.code for t in transform_list) + "]"
    
    @classmethod
    def apply(cls, children):
        #super().apply(children)
        return cls(children)

class UpdateColor(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    childTypes = [Types.COLOR]
    def __init__(self, color: Color):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [color]
        self.size = 1 + color.size
        self.code = f"updateColor({color.code})"

    @classmethod
    def apply(cls, children):
        return cls(children[0])

class MoveNode(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, direction: Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [direction]
        self.size = 1 + direction.size
        self.code = f"moveNode({direction.code})"
        self.childTypes = [Types.DIRECTION]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0])

class ExtendNode(TransformOps):
    arity = 2
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, direction: Direction, overlap: Overlap):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [direction, overlap]
        self.size = 1 + direction.size + overlap.size
        self.code = f"extendNode({direction.code}, {overlap.code})"
        self.childTypes = [Types.DIRECTION, Types.OVERLAP]

    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class MoveNodeMax(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, direction: Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [direction]
        self.size = 1 + direction.size
        self.code = f"moveNodeMax({direction.code})"
        self.childTypes = [Types.DIRECTION]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0])

class RotateNode(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, rotation_direction: Rotation_Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [rotation_direction]
        self.size = 1 + rotation_direction.size
        self.code = f"rotateNode({rotation_direction.code})"
        self.childTypes = [Types.ROTATION_DIRECTION]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0])

class AddBorder(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, fill_color: FillColor):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [fill_color]
        self.size = 1 + fill_color.size
        self.code = f"addBorder({fill_color.code})"
        self.childTypes = [Types.FILLCOLOR]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0])

class FillRectangle(TransformOps):
    arity = 2
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, fill_color: FillColor, overlap: Overlap):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [fill_color, overlap]
        self.size = 1 + fill_color.size + overlap.size
        self.code = f"fillRectangle({fill_color.code}, {overlap.code})"
        self.childTypes = [Types.FILLCOLOR, Types.OVERLAP]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class HollowRectangle(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, fill_color: FillColor):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [fill_color]
        self.size = 1 + fill_color.size
        self.code = f"hollowRectangle({fill_color.code})"
        self.childTypes = [Types.FILLCOLOR]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0])

class Insert(TransformOps):
    arity = 2
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, point: ImagePoints, relative_pos: RelativePosition):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [point, relative_pos]
        self.size = 1 + point.size + relative_pos.size
        self.code = f"Insert({point.code, relative_pos.code})"
        self.childTypes = [Types.IMAGEPOINTS, Types.RELATIVE_POS]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class Mirror(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, mirror_axis: Mirror_Axis):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [mirror_axis]
        self.size = 1 + mirror_axis.size
        self.code = f"mirror({mirror_axis.code})"
        self.childTypes = [Types.MIRROR_AXIS]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0])

class Flip(TransformOps):
    arity = 1
    nodeType = Types.TRANSFORM_OPS
    def __init__(self, mirror_direction: Mirror_Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM_OPS
        self.children = [mirror_direction]
        self.size = 1 + mirror_direction.size
        self.code = f"flip({mirror_direction.code})"
        self.arity = 1
        self.childTypes = [Types.MIRROR_DIRECTION]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0])

import unittest

class TestGrammarRepresentation(unittest.TestCase):
    
    def test_color_enum(self):
        color_instance = Color.C0
        self.assertEqual(color_instance.nodeType, Types.COLOR)
        self.assertEqual(color_instance.code, "Color.C0")
        self.assertEqual(color_instance.size, 1)
        self.assertEqual(color_instance.children, [])
        
    def test_direction_enum(self):
        direction_instance = Direction.UP
        self.assertEqual(direction_instance.nodeType, Types.DIRECTION)
        self.assertEqual(direction_instance.code, "Direction.UP")
        self.assertEqual(direction_instance.size, 1)
        self.assertEqual(direction_instance.children, [])
        
    def test_transforms(self):
        transforms_instance = Transforms([UpdateColor(Color.C1), UpdateColor(Color.C2)])
        self.assertEqual(transforms_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(transforms_instance.code, "[updateColor(Color.C1), updateColor(Color.C2)]")
        self.assertEqual(transforms_instance.size, 4)  # 1 for transforms + 2 for each updateColor
        self.assertEqual(len(transforms_instance.children), 2)

    def test_move_node(self):
        move_node_instance = MoveNode(Direction.DOWN)
        self.assertEqual(move_node_instance.nodeType, Types.TRANSFORM_OPS)
        self.assertEqual(move_node_instance.code, "moveNode(Direction.DOWN)")
        self.assertEqual(move_node_instance.size, 2)
        self.assertEqual(move_node_instance.children, [Direction.DOWN])
        
    def test_extend_node(self):
        extend_node_instance = ExtendNode(Direction.LEFT, Overlap.TRUE)
        self.assertEqual(extend_node_instance.nodeType, Types.TRANSFORM_OPS)
        self.assertEqual(extend_node_instance.code, "extendNode(Direction.LEFT, Overlap.TRUE)")
        self.assertEqual(extend_node_instance.size, 3)
        self.assertEqual(extend_node_instance.children, [Direction.LEFT, Overlap.TRUE])
        
    def test_add_border(self):
        add_border_instance = AddBorder(FillColor.C3)
        self.assertEqual(add_border_instance.nodeType, Types.TRANSFORM_OPS)
        self.assertEqual(add_border_instance.code, "addBorder(FillColor.C3)")
        self.assertEqual(add_border_instance.size, 2)
        self.assertEqual(add_border_instance.children, [FillColor.C3])
        
    def test_mirror(self):
        mirror_instance = Mirror(Mirror_Axis.X_AXIS)
        self.assertEqual(mirror_instance.nodeType, Types.TRANSFORM_OPS)
        self.assertEqual(mirror_instance.code, "mirror(Mirror_Axis.X_AXIS)")
        self.assertEqual(mirror_instance.size, 2)
        self.assertEqual(mirror_instance.children, [Mirror_Axis.X_AXIS])

    def test_sequence_of_multiple_transforms(self):
        transforms_list = [UpdateColor(Color.C3), MoveNode(Direction.UP)]
        sequence = Transforms(transforms_list)
        self.assertEqual(sequence.nodeType, Types.TRANSFORMS)
        self.assertEqual(sequence.code, "[updateColor(Color.C3), moveNode(Direction.UP)]")
        self.assertEqual(sequence.size, 4)  # 2 for each transformation
        self.assertEqual(len(sequence.children), 2)

    def test_complex_transform_sequence(self):
        transforms_list = [
        UpdateColor(Color.C1),
        MoveNode(Direction.UP),
        RotateNode(Rotation_Direction.CCW),
        Mirror(Mirror_Axis.Y_AXIS),
        AddBorder(FillColor.C7)
        ]
        sequence = Transforms(transforms_list)
        self.assertEqual(sequence.nodeType, Types.TRANSFORMS)
        self.assertEqual(sequence.code, "[updateColor(Color.C1), moveNode(Direction.UP), rotateNode(Rotation_Direction.CCW), mirror(Mirror_Axis.Y_AXIS), addBorder(FillColor.C7)]")
        self.assertEqual(sequence.size, 10)  # 2 for each transformation
        self.assertEqual(len(sequence.children), 5)
          
    def test_get_all_values(self):
        all_colors = Color.get_all_values()
        self.assertIsInstance(all_colors, list)
        self.assertEqual(len(all_colors), 12)  # because there are 11 colors defined in the Color enum
        for color in all_colors:
            self.assertIsInstance(color, Color)
        self.assertIn(Color.C0, all_colors)
        self.assertIn(Color.MOST, all_colors)

if __name__ == "__main__":
    unittest.main()