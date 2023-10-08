from typing import List, Union
from enum import Enum

class Types(Enum):
    START = "Start"
    TRANSFORMS = "Transforms"
    TRANSFORM = "Transform"
    COLOR = "Color"
    DIRECTION = "Direction"
    OVERLAP = "Overlap"
    ROTATION_DIRECTION = "Rotation_Direction"
    FILLCOLOR = "FillColor"
    MIRROR_AXIS = "Mirror_Axis"
    MIRROR_DIRECTION = "Mirror_Direction"
    RELATIVE_POS = "Relative_Pos"

class ASTNode:
    def __init__(self, node_type: Types):
        self.nodeType: NodeType = node_type
        self.code: str = self.__class__.__name__
        self.size: int = 1
        self.children: List[ASTNode] = []

class Color(ASTNode, Enum):
    C0 =  "0"
    C1 =  "1"
    C2 =  "2"
    C3 =  "3"
    C4 =  "4"
    C5 =  "5"
    C6 =  "6"
    C7 =  "7"
    C8 =  "8"
    C9 =  "9"
    LEAST = "least"
    MOST =  "most"

    def __init__(self, value):
        super().__init__(Types.COLOR)
        self.code = value

class Direction(ASTNode, Enum):
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
        self.code = value

class Overlap(ASTNode, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"

    def __init__(self, value):
        super().__init__(Types.OVERLAP)
        self.code = value

class Rotation_Direction(ASTNode, Enum):
    CW = "CW"
    CCW = "CCW"
    CW2 = "CW2"

    def __init__(self, value):
        super().__init__(Types.ROTATION_DIRECTION)
        self.code = value

class FillColor(ASTNode, Enum):
    C0 =  "0"
    C1 =  "1"
    C2 =  "2"
    C3 =  "3"
    C4 =  "4"
    C5 =  "5"
    C6 =  "6"
    C7 =  "7"
    C8 =  "8"
    C9 =  "9"
    def __init__(self, value):
        super().__init__(Types.FILLCOLOR)
        self.code = value

class Mirror_Direction(ASTNode, Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    DIAGONAL_LEFT = "DIAGONAL_LEFT"
    DIAGONAL_RIGHT = "DIAGONAL_RIGHT"

    def __init__(self, value):
        super().__init__(Types.MIRROR_DIRECTION)
        self.code = value

class Mirror_Axis(ASTNode, Enum):
    X_AXIS = "X_AXIS"
    Y_AXIS = "Y_AXIS"

    def __init__(self, value):
        super().__init__(Types.MIRROR_AXIS)
        self.code = value

class RelativePosition(ASTNode, Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    MIDDLE = "MIDDLE"

    def __init__(self, value):
        super().__init__(Types.RELATIVE_POS)
        self.code = value

class Transform(ASTNode):
    def __init__(self):
        super().__init__(Types.TRANSFORM)

class Transforms(ASTNode):
    def __init__(self, transform_list: Union[Transform, List[Transform]]):
        super().__init__(Types.TRANSFORMS)
        if not isinstance(transform_list, list):
            transform_list = [transform_list]
        self.children = transform_list
        self.size = sum(t.size for t in transform_list)  # 1 for Transforms node + total of all child nodes
        self.code = "[" + ", ".join(t.code for t in transform_list) + "]"

class UpdateColor(Transform):
    def __init__(self, color: Color):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [color]
        self.size = 1 + color.size
        self.code = f"updateColor({color.code})"

class MoveNode(Transform):
    def __init__(self, direction: Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [direction]
        self.size = 1 + direction.size
        self.code = f"moveNode({direction.code})"

class ExtendNode(Transform):
    def __init__(self, direction: Direction, overlap: Overlap):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [direction, overlap]
        self.size = 1 + direction.size + overlap.size
        self.code = f"extendNode({direction.code}, {overlap.code})"

class MoveNodeMax(Transform):
    def __init__(self, direction: Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [direction]
        self.size = 1 + direction.size
        self.code = f"moveNodeMax({direction.code})"

class RotateNode(Transform):
    def __init__(self, rotation_direction: Rotation_Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [rotation_direction]
        self.size = 1 + rotation_direction.size
        self.code = f"rotateNode({rotation_direction.code})"

class AddBorder(Transform):
    def __init__(self, fill_color: FillColor):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [fill_color]
        self.size = 1 + fill_color.size
        self.code = f"addBorder({fill_color.code})"

class FillRectangle(Transform):
    def __init__(self, fill_color: FillColor, overlap: Overlap):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [fill_color, overlap]
        self.size = 1 + fill_color.size + overlap.size
        self.code = f"fillRectangle({fill_color.code}, {overlap.code})"

class HollowRectangle(Transform):
    def __init__(self, fill_color: FillColor):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [fill_color]
        self.size = 1 + fill_color.size
        self.code = f"hollowRectangle({fill_color.code})"

class Insert(Transform):
    def __init__(self, relative_pos: RelativePosition):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [relative_pos]
        self.size = 1 + relative_pos.size
        self.code = f"Insert({relative_pos.code})"

class Mirror(Transform):
    def __init__(self, mirror_axis: Mirror_Axis):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [mirror_axis]
        self.size = 1 + mirror_axis.size
        self.code = f"mirror({mirror_axis.code})"

class Flip(Transform):
    def __init__(self, mirror_direction: Mirror_Direction):
        super().__init__()
        self.nodeType = Types.TRANSFORM
        self.children = [mirror_direction]
        self.terms = 1 + mirror_direction.size
        self.code = f"flip({mirror_direction.code})"

# Tests below
import unittest

class TestGrammarRepresentation(unittest.TestCase):
    
    def test_color_enum(self):
        color_instance = Color.C0
        self.assertEqual(color_instance.nodeType, Types.COLOR)
        self.assertEqual(color_instance.code, "0")
        self.assertEqual(color_instance.size, 1)
        self.assertEqual(color_instance.children, [])
        
    def test_direction_enum(self):
        direction_instance = Direction.UP
        self.assertEqual(direction_instance.nodeType, Types.DIRECTION)
        self.assertEqual(direction_instance.code, "UP")
        self.assertEqual(direction_instance.size, 1)
        self.assertEqual(direction_instance.children, [])
        
    def test_transforms(self):
        transforms_instance = Transforms([UpdateColor(Color.C1), UpdateColor(Color.C2)])
        self.assertEqual(transforms_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(transforms_instance.code, "[updateColor(1), updateColor(2)]")
        self.assertEqual(transforms_instance.size, 4)  # 1 for transforms + 2 for each updateColor
        self.assertEqual(len(transforms_instance.children), 2)

    def test_single_transforms(self):
        transforms_instance = Transforms(UpdateColor(Color.C1))
        self.assertEqual(transforms_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(transforms_instance.code, "[updateColor(1)]")
        self.assertEqual(transforms_instance.size, 2)
        self.assertEqual(len(transforms_instance.children), 1)

    def test_move_node(self):
        move_node_instance = MoveNode(Direction.DOWN)
        self.assertEqual(move_node_instance.nodeType, Types.TRANSFORM)
        self.assertEqual(move_node_instance.code, "moveNode(DOWN)")
        self.assertEqual(move_node_instance.size, 2)
        self.assertEqual(move_node_instance.children, [Direction.DOWN])
        
    def test_extend_node(self):
        extend_node_instance = ExtendNode(Direction.LEFT, Overlap.TRUE)
        self.assertEqual(extend_node_instance.nodeType, Types.TRANSFORM)
        self.assertEqual(extend_node_instance.code, "extendNode(LEFT, TRUE)")
        self.assertEqual(extend_node_instance.size, 3)
        self.assertEqual(extend_node_instance.children, [Direction.LEFT, Overlap.TRUE])
        
    def test_add_border(self):
        add_border_instance = AddBorder(FillColor.C3)
        self.assertEqual(add_border_instance.nodeType, Types.TRANSFORM)
        self.assertEqual(add_border_instance.code, "addBorder(3)")
        self.assertEqual(add_border_instance.size, 2)
        self.assertEqual(add_border_instance.children, [FillColor.C3])
        
    def test_mirror(self):
        mirror_instance = Mirror(Mirror_Axis.X_AXIS)
        self.assertEqual(mirror_instance.nodeType, Types.TRANSFORM)
        self.assertEqual(mirror_instance.code, "mirror(X_AXIS)")
        self.assertEqual(mirror_instance.size, 2)
        self.assertEqual(mirror_instance.children, [Mirror_Axis.X_AXIS])

    def test_flip(self):
        flip_instance = Flip(Mirror_Direction.VERTICAL)
        self.assertEqual(flip_instance.nodeType, Types.TRANSFORM)
        self.assertEqual(flip_instance.code, "flip(VERTICAL)")
        self.assertEqual(flip_instance.children, [Mirror_Direction.VERTICAL])

    def test_sequence_of_single_transforms(self):
        # Testing a sequence with single transformation
        sequence = Transforms(UpdateColor(Color.C3))
        self.assertEqual(sequence.nodeType, Types.TRANSFORMS)
        self.assertEqual(sequence.code, "[updateColor(3)]")
        self.assertEqual(sequence.size, 2)  # 1 for transform + 1 for color
        self.assertEqual(len(sequence.children), 1)

    def test_sequence_of_multiple_transforms(self):
        # Testing a sequence with multiple transformations
        sequence = Transforms([UpdateColor(Color.C3), MoveNode(Direction.UP)])
        self.assertEqual(sequence.nodeType, Types.TRANSFORMS)
        self.assertEqual(sequence.code, "[updateColor(3), moveNode(UP)]")
        self.assertEqual(sequence.size, 4)  # 2 for each transformation
        self.assertEqual(len(sequence.children), 2)

    def test_nested_transform_sequence(self):
        # Testing nested sequences of transformations
        inner_sequence = Transforms([UpdateColor(Color.C5), MoveNode(Direction.LEFT)])
        outer_sequence = Transforms([inner_sequence, MoveNode(Direction.DOWN)])
        self.assertEqual(outer_sequence.nodeType, Types.TRANSFORMS)
        self.assertEqual(outer_sequence.code, "[[updateColor(5), moveNode(LEFT)], moveNode(DOWN)]")
        self.assertEqual(outer_sequence.size, 6)
        self.assertEqual(len(outer_sequence.children), 2)

    def test_complex_transform_sequence(self):
        # Testing a more complex sequence of transformations
        sequence = Transforms([
            UpdateColor(Color.C1),
            MoveNode(Direction.UP),
            RotateNode(Rotation_Direction.CCW),
            Mirror(Mirror_Axis.Y_AXIS),
            AddBorder(FillColor.C7)
        ])
        self.assertEqual(sequence.nodeType, Types.TRANSFORMS)
        self.assertEqual(sequence.code, "[updateColor(1), moveNode(UP), rotateNode(CCW), mirror(Y_AXIS), addBorder(7)]")
        self.assertEqual(sequence.size, 10)  # 2 for each transformation
        self.assertEqual(len(sequence.children), 5)
    
if __name__ == "__main__":
    unittest.main()