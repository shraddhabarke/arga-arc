import arga_ast_generator as ast
from task import Task
import transform as tf
import filters as f
import typing as t
from enum import Enum

from utils import TASK_IDS, TASK_IDS_TYPE
from config import CONFIG
from image import Image

TASKS = [
    {"task_id": "08ed6ac7", "abstraction": "nbccg"},
    {"task_id": "1e0a9b12", "abstraction": None},
    {"task_id": "25ff71a9", "abstraction": "nbccg"},
    {"task_id": "3906de3d", "abstraction": "nbvcg"},
    {"task_id": "4258a5f9", "abstraction": "nbccg"},
    {"task_id": "50cb2852", "abstraction": "nbccg"},
    {"task_id": "543a7ed5", "abstraction": "mcccg"},
    {"task_id": "6455b5f5", "abstraction": "ccg"},
    {"task_id": "67385a82", "abstraction": "nbccg"},
    {"task_id": "694f12f3", "abstraction": "nbccg"},
    {"task_id": "6e82a1ae", "abstraction": "nbccg"},
    {"task_id": "7f4411dc", "abstraction": "lrg"},
    {"task_id": "a79310a0", "abstraction": "nbccg"},
    {"task_id": "aedd82e4", "abstraction": "nbccg"},
    {"task_id": "b1948b0a", "abstraction": "nbccg"},
    {"task_id": "b27ca6d3", "abstraction": "nbccg"},
    {"task_id": "bb43febb", "abstraction": "nbccg"},
    {"task_id": "c8f0f002", "abstraction": "nbccg"},
    {"task_id": "d2abd087", "abstraction": "nbccg"},
    {"task_id": "dc1df850", "abstraction": "nbccg"},
    {"task_id": "ea32f347", "abstraction": "nbccg"},
    {"task_id": "6d75e8bb", "abstraction": "nbccg"},
    {"task_id": "00d62c1b", "abstraction": "ccgbr"},
    {"task_id": "9565186b", "abstraction": "nbccg"},
    {"task_id": "810b9b61", "abstraction": "ccgbr"},
    {"task_id": "a5313dff", "abstraction": "ccgbr"},
    {"task_id": "aabf363d", "abstraction": "nbccg"},
    {"task_id": "d5d6de2d", "abstraction": "ccg"},
    {"task_id": "67a3c6ac", "abstraction": "na"},
    {"task_id": "3c9b0459", "abstraction": "na"},
    {"task_id": "9dfd6313", "abstraction": "na"},
    {"task_id": "ed36ccf7", "abstraction": "na"},
    {"task_id": "ddf7fa4f", "abstraction": "nbccg"},
    {"task_id": "05f2a901", "abstraction": "nbccg"},
    {"task_id": "d43fd935", "abstraction": "nbccg"},
    {"task_id": "f8a8fe49", "abstraction": "nbccg"},
    {"task_id": "ae3edfdc", "abstraction": "nbccg"},
]

TASK_IDS = [task["task_id"] for task in TASKS]
TASK_ABSTRACTIONS = [task["abstraction"] for task in TASKS]


def main():
    # abstraction = "nbccg"
    for idx, task in enumerate(TASKS):
        task_id = task["task_id"]
        abstraction = task["abstraction"] if task["abstraction"] else "nbccg"
        try:
            programs = parse_programs(task_id, abstraction)
            num_correct = num_correct_programs(task_id, abstraction, programs)
        except Exception as e:
            print(f"{task_id}: {e}")
            continue
        print(f"{task_id}: correct/total {num_correct}/{len(programs)}")


def parse_programs(
    task_id: TASK_IDS_TYPE, abstraction: str, directory="dsl/gens/gens_20231120"
) -> t.List[ast.DoOperation]:
    file = (CONFIG.ROOT_DIR / directory / f"{task_id}_correct.txt").resolve().absolute()
    with open(file, "r") as f:
        program = f.read()

    ast_program = ast.parse(program)
    programs = ast_program.children
    return programs


def get_task(task_id: TASK_IDS_TYPE, abstraction: str) -> Task:
    path = CONFIG.ROOT_DIR / "ARC" / f"{task_id}.json"
    task = Task(str(path.resolve().absolute()))
    task.abstraction = abstraction
    task.input_abstracted_graphs_original[abstraction] = [
        getattr(input, Image.abstraction_ops[abstraction])()
        for input in task.train_input
    ]
    task.get_static_object_attributes(abstraction)
    f.setup_size_and_degree_based_on_task(task)
    return task


def num_correct_programs(
    task_id: TASK_IDS_TYPE, abstraction: str, programs: t.List[ast.DoOperation]
) -> int:
    task = get_task(task_id, abstraction)
    num_correct = 0
    for idx, program in enumerate(programs):
        executable_program = convert_ast_to_executable(program)
        test_results = task.test_program(executable_program, abstraction)
        if test_results:
            num_correct += 1
    return num_correct


# region CONVERT AST TO EXECUTABLE


def convert_ast_to_executable(
    program: ast.DoOperation,
) -> t.List[t.Tuple[f.FilterASTNode, t.List[tf.TransformASTNode]]]:
    rules = [
        rule if isinstance(rule, ast._Rule) else rule.rules[0]
        for rule in program.rule_list
    ]
    return [convert_rule_to_executable(rule) for rule in rules]


def convert_rule_to_executable(
    rule: ast._Rule,
) -> t.Tuple[f.FilterASTNode, t.List[tf.TransformASTNode]]:
    return (
        convert_filter_op_to_executable(rule.filter_op),
        convert_transforms_to_executable(rule.transforms),
    )


def convert_filter_op_to_executable(filter_op: ast._Filter_Op) -> f.FilterASTNode:
    if isinstance(filter_op, ast.FilterByNeighborDegree):
        return f.FilterByNeighborDegree(
            degree=convert_filter_op_to_executable(filter_op.degree)
        )
    elif isinstance(filter_op, ast.FilterByNeighborColor):
        return f.FilterByNeighborColor(
            color=convert_color_to_filter_ast_node(filter_op.color)
        )
    elif isinstance(filter_op, ast.FilterByNeighborSize):
        return f.FilterByNeighborSize(
            size=convert_filter_op_to_executable(filter_op.size)
        )
    elif isinstance(filter_op, ast.FilterByDegree):
        return f.FilterByDegree(
            degree=convert_filter_op_to_executable(filter_op.degree)
        )
    elif isinstance(filter_op, ast.FilterByColor):
        return f.FilterByColor(color=convert_color_to_filter_ast_node(filter_op.color))
    elif isinstance(filter_op, ast.FilterBySize):
        return f.FilterBySize(size=convert_filter_op_to_executable(filter_op.size))
    elif isinstance(filter_op, ast.Or):
        assert len(filter_op.filters) >= 2
        first_two_filters = filter_op.filters[:2]
        ans = f.Or(
            convert_filter_op_to_executable(first_two_filters[0]),
            convert_filter_op_to_executable(first_two_filters[1]),
        )
        for filter in filter_op.filters[2:]:
            ans = f.Or(ans, convert_filter_op_to_executable(filter))
        return ans
    elif isinstance(filter_op, ast.And):
        assert len(filter_op.filters) >= 2
        first_two_filters = filter_op.filters[:2]
        ans = f.And(
            convert_filter_op_to_executable(first_two_filters[0]),
            convert_filter_op_to_executable(first_two_filters[1]),
        )
        for filter in filter_op.filters[2:]:
            ans = f.And(ans, convert_filter_op_to_executable(filter))
        return ans
    elif isinstance(filter_op, ast.Not):
        return f.Not(convert_filter_op_to_executable(filter_op.not_filter))
    elif isinstance(filter_op, ast.Color):
        return convert_color_to_filter_ast_node(filter_op)
    elif isinstance(filter_op, ast.Size):
        return convert_value_to_size_filter_ast_node(filter_op.value)
    elif isinstance(filter_op, ast.Degree):
        return convert_value_to_degree_filter_ast_node(filter_op.value)


def convert_value_to_size_filter_ast_node(value: t.Union[str, int]) -> f.Size:
    all_values = f.Size.get_all_values()
    ans = [v for v in all_values if str(v.value) == str(value)]

    assert len(ans) <= 1

    if len(ans) == 1:
        return ans[0]

    try:
        value = int(value)
    except ValueError:
        raise ValueError(f"Size value {value} is not a valid size.")

    # add new enum value if it doesn't exist
    next_enum_members = all_values
    _temp_enum = Enum("SizeEnum", {f"{value}": value})
    for name, member in _temp_enum.__members__.items():
        setattr(f.Size, name, f.SizeValue(member))
        next_enum_members.append(f.Size(member))
    f.Size._enum_members = next_enum_members

    return convert_value_to_size_filter_ast_node(value)


def convert_value_to_degree_filter_ast_node(value: t.Union[str, int]) -> f.Degree:
    all_values = f.Degree.get_all_values()
    ans = [v for v in all_values if str(v.value) == str(value)]

    assert len(ans) <= 1

    if len(ans) == 1:
        return ans[0]

    try:
        value = int(value)
    except ValueError:
        raise ValueError(f"Degree value {value} is not a valid degree.")

    # add new enum value if it doesn't exist
    next_enum_members = all_values
    _temp_enum = Enum("DegreeEnum", {f"{value}": value})
    for name, member in _temp_enum.__members__.items():
        setattr(f.Degree, name, f.DegreeValue(member))
        next_enum_members.append(f.Degree(member))
    f.Degree._enum_members = next_enum_members

    return convert_value_to_degree_filter_ast_node(value)


def convert_color_to_filter_ast_node(color: ast.Color) -> f.FColor:
    if color.value == "O":
        return f.FColor.black
    elif color.value == "B":
        return f.FColor.blue
    elif color.value == "R":
        return f.FColor.red
    elif color.value == "G":
        return f.FColor.green
    elif color.value == "Y":
        return f.FColor.yellow
    elif color.value == "X":
        return f.FColor.grey
    elif color.value == "F":
        return f.FColor.fuchsia
    elif color.value == "A":
        return f.FColor.orange
    elif color.value == "C":
        return f.FColor.cyan
    elif color.value == "W":
        return f.FColor.brown


def convert_transforms_to_executable(
    transforms: ast.Transforms,
) -> t.List[tf.TransformASTNode]:
    return [
        convert_transform_to_executable(transform)
        for transform in transforms.transforms
    ]


def convert_transform_to_executable(transform: ast.Transform) -> tf.TransformASTNode:
    if isinstance(transform, ast.Mirror):
        raise NotImplementedError(
            "TODO: implement mirror transform. I'm not 100% sure how MirrorAxis works"
        )
    elif isinstance(transform, ast.Flip):
        return tf.Flip(axis_point=convert_transform_to_executable(transform.axis_point))
    elif isinstance(transform, ast.HollowRectangle):
        return tf.HollowRectangle(
            color=convert_transform_to_executable(transform.color),
            # overlap=convert_overlap_to_transform_ast_node(transform.overlap),
        )
    elif isinstance(transform, ast.FillRectangle):
        return tf.FillRectangle(
            color=convert_transform_to_executable(transform.color),
            overlap=tf.Overlap.TRUE if transform.overlap else tf.Overlap.FALSE,
        )
    elif isinstance(transform, ast.AddBorder):
        return tf.AddBorder(color=convert_transform_to_executable(transform.color))
    elif isinstance(transform, ast.RotateNode):
        return tf.RotateNode(
            rotation_angle=convert_transform_to_executable(transform.rotation_angle)
        )
    elif isinstance(transform, ast.MoveNodeMax):
        return tf.MoveNodeMax(dir=convert_transform_to_executable(transform.direction))
    elif isinstance(transform, ast.MoveNode):
        return tf.MoveNode(dir=convert_transform_to_executable(transform.direction))
    elif isinstance(transform, ast.ExtendNode):
        return tf.ExtendNode(
            dir=convert_transform_to_executable(transform.direction),
            overlap=convert_overlap_to_transform_ast_node(transform.overlap),
        )
    elif isinstance(transform, ast.UpdateColor):
        return tf.UpdateColor(color=convert_transform_to_executable(transform.color))
    elif isinstance(transform, ast.Symmetry_Axis):
        return convert_symmetry_axis_to_transform_ast_node(transform)
    elif isinstance(transform, ast.Rotation_Angle):
        return convert_rotation_angle_to_transform_ast_node(transform)
    elif isinstance(transform, ast.Color):
        return convert_color_to_transform_ast_node(transform)
    elif isinstance(transform, ast.Direction):
        return convert_direction_to_transform_ast_node(transform)
    else:
        raise ValueError(
            f"Transform {transform} of type {transform.__class__.__name__} is not a valid transform."
        )


def convert_overlap_to_transform_ast_node(overlap: bool) -> tf.Overlap:
    return tf.Overlap.TRUE if overlap else tf.Overlap.FALSE


def convert_symmetry_axis_to_transform_ast_node(
    symmetry_axis: ast.Symmetry_Axis,
) -> tf.Symmetry_Axis:
    if symmetry_axis.value == "H":
        return tf.Symmetry_Axis.HORIZONTAL
    elif symmetry_axis.value == "V":
        return tf.Symmetry_Axis.VERTICAL
    # TODO: i'm confused about which is diagonal and which is anti-diagonal
    # don't worry about this, it's from an old version of the grammar
    elif symmetry_axis.value == "D":
        return tf.Symmetry_Axis.DIAGONAL_LEFT
    elif symmetry_axis.value == "AD":
        return tf.Symmetry_Axis.DIAGONAL_RIGHT
    else:
        raise ValueError(
            f"Symmetry axis value {symmetry_axis.value} is not a valid symmetry axis."
        )


def convert_rotation_angle_to_transform_ast_node(
    rotation_angle: ast.Rotation_Angle,
) -> tf.Rotation_Angle:
    if rotation_angle.value == "90":
        return tf.Rotation_Angle.CCW
    elif rotation_angle.value == "180":
        return tf.Rotation_Angle.CW2
    elif rotation_angle.value == "270":
        return tf.Rotation_Angle.CW
    else:
        raise ValueError(
            f"Rotation angle value {rotation_angle.value} is not a valid rotation angle."
        )


def convert_color_to_transform_ast_node(color: ast.Color) -> tf.Color:
    if color.value == "O":
        return tf.Color.black
    elif color.value == "B":
        return tf.Color.blue
    elif color.value == "R":
        return tf.Color.red
    elif color.value == "G":
        return tf.Color.green
    elif color.value == "Y":
        return tf.Color.yellow
    elif color.value == "X":
        return tf.Color.grey
    elif color.value == "F":
        return tf.Color.fuchsia
    elif color.value == "A":
        return tf.Color.orange
    elif color.value == "C":
        return tf.Color.cyan
    elif color.value == "W":
        return tf.Color.brown
    else:
        raise ValueError(f"Color value {color.value} is not a valid color.")


def convert_direction_to_transform_ast_node(direction: ast.Direction) -> tf.Dir:
    if direction.value == "U":
        return tf.Dir.UP
    elif direction.value == "D":
        return tf.Dir.DOWN
    elif direction.value == "L":
        return tf.Dir.LEFT
    elif direction.value == "R":
        return tf.Dir.RIGHT
    elif direction.value == "UL":
        return tf.Dir.UP_LEFT
    elif direction.value == "UR":
        return tf.Dir.UP_RIGHT
    elif direction.value == "DL":
        return tf.Dir.DOWN_LEFT
    elif direction.value == "DR":
        return tf.Dir.DOWN_RIGHT
    else:
        raise ValueError(f"Direction value {direction.value} is not a valid direction.")


# endregion CONVERT AST TO EXECUTABLE

if __name__ == "__main__":
    main()
