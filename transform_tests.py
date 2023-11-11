import unittest
from task import *
from transform import *
from filters import FilterByColor, Not
class TestGrammarRepresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taskNumber = "bb43febb"
        cls.task = Task("dataset/" + cls.taskNumber + ".json")
        cls.task.abstraction = "nbccg"
        cls.task.input_abstracted_graphs_original[cls.task.abstraction] = [
            getattr(input, Image.abstraction_ops[cls.task.abstraction])() for
            input in cls.task.train_input
        ]
        cls.task.get_static_object_attributes(cls.task.abstraction)
        cls.filter = Not(FilterByColor(FColor.C0))

    def test_color_enum(self):
        color_instance = Color.C0
        self.assertEqual(color_instance.nodeType, Types.COLOR)
        self.assertEqual(color_instance.code, "Color.C0")
        self.assertEqual(color_instance.size, 1)
        self.assertEqual(color_instance.children, [])
        self.assertEqual(color_instance.values, [])
        
    def test_direction_enum(self):
        direction_instance = Direction.UP
        self.assertEqual(direction_instance.nodeType, Types.DIRECTION)
        self.assertEqual(direction_instance.code, "Direction.UP")
        self.assertEqual(direction_instance.size, 1)
        self.assertEqual(direction_instance.children, [])
        self.assertEqual(direction_instance.values, [])

    def test_move_node(self):
        move_node_instance = Transforms(MoveNode(Direction.DOWN), NoOp())
        self.assertEqual(move_node_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(move_node_instance.code, "[moveNode(Direction.DOWN), NoOp]")
        self.assertEqual(move_node_instance.size, 3)
        self.assertEqual(len(move_node_instance.children), 2)
        
    def test_extend_node(self):
        extend_node_instance = ExtendNode(Direction.LEFT, Overlap.TRUE)
        new_instance = extend_node_instance.apply(self.task, extend_node_instance.children, self.filter)
        self.assertEqual(new_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(new_instance.code, "extendNode(Direction.LEFT, Overlap.TRUE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(new_instance.children, [Direction.LEFT, Overlap.TRUE])
        self.assertEqual(new_instance.values, [[5, 5, 5, 5, 5, 5, 0, 0, 0, 0], 
                                               [5, 5, 5, 5, 5, 5, 0, 0, 0, 0], 
                                               [5, 5, 5, 5, 5, 5, 0, 0, 0, 0], 
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 
                                               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 
                                               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 
                                               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 
                                               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 
                                               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]])

    def test_update_color(self):
        instance = UpdateColor(Color.C1)
        new_instance = instance.apply(self.task, instance.children, self.filter)
        self.assertEqual(new_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(new_instance.code, "updateColor(Color.C1)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(new_instance.children, [Color.C1])
        self.assertEqual(new_instance.values, [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
                                               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
                                               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                                               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                                               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                                               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                                               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                                               [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

    def test_add_border(self):
        add_border_instance = AddBorder(Color.C3)
        self.assertEqual(add_border_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(add_border_instance.code, "addBorder(Color.C3)")
        self.assertEqual(add_border_instance.size, 2)
        self.assertEqual(add_border_instance.children, [Color.C3])
        
    def test_mirror(self):
        mirror_instance = Mirror(Mirror_Axis.X_AXIS)
        self.assertEqual(mirror_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(mirror_instance.code, "mirror(Mirror_Axis(axis_point=(1, None)))")
        self.assertEqual(mirror_instance.size, 2)

    def test_get_all_values(self):
        all_colors = Color.get_all_values()
        self.assertIsInstance(all_colors, list)
        self.assertEqual(len(all_colors), 12)  # because there are 11 colors defined in the Color enum
        for color in all_colors:
            self.assertIsInstance(color, Color)
        self.assertIn(Color.C0, all_colors)
        self.assertIn(Color.MOST, all_colors)

    def test_noop_properties(self):
        noop = NoOp()
        self.assertEqual(noop.code, "NoOp")
        self.assertEqual(noop.size, 1)
        self.assertEqual(noop.nodeType, Types.TRANSFORMS)
        self.assertEqual(NoOp.arity, 0)

    def test_transforms(self):
        move_node_instance = Transforms(MoveNode(Direction.DOWN), MoveNode(Direction.UP))
        self.assertEqual(move_node_instance.nodeType, Types.TRANSFORMS)
        self.assertEqual(move_node_instance.code, "[moveNode(Direction.DOWN), moveNode(Direction.UP)]")
        self.assertEqual(move_node_instance.size, 4)  # 1 for transforms + 2 for each updateColor
        self.assertEqual(len(move_node_instance.children), 2)

    def test_sequence_of_multiple_transforms(self):
        transforms_list = Transforms(UpdateColor(Color.C3), Transforms(MoveNode(Direction.UP), NoOp()))
        self.assertEqual(transforms_list.nodeType, Types.TRANSFORMS)
        self.assertEqual(transforms_list.code, "[updateColor(Color.C3), [moveNode(Direction.UP), NoOp]]")
        self.assertEqual(transforms_list.size, 5)  # 2 for each transformation
        self.assertEqual(len(transforms_list.children), 2)

    def test_complex_transform_sequence(self):
        transforms_list = Transforms(UpdateColor(Color.C1),
                             Transforms(MoveNode(Direction.UP),
                                        Transforms(RotateNode(Rotation_Angle.CCW),
                                                   Transforms(Mirror(Mirror_Axis.Y_AXIS),
                                                              Transforms(AddBorder(Color.C7), NoOp())
                                                             )
                                                  )
                                       )
                            )
        self.assertEqual(transforms_list.nodeType, Types.TRANSFORMS)
        self.assertEqual(transforms_list.code, "[updateColor(Color.C1), [moveNode(Direction.UP), [rotateNode(Rotation_Angle.CCW), [mirror(Mirror_Axis.Y_AXIS), [addBorder(Color.C7), NoOp]]]]]")
        self.assertEqual(transforms_list.size, 11)  # 2 for each transformation
        self.assertEqual(len(transforms_list.children), 2)

if __name__ == "__main__":
    unittest.main()