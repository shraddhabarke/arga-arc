from lark import Lark, ast_utils, Transformer, v_args
from typing import List, Any, Optional, Tuple
from lark import Tree
import sys
from dataclasses import dataclass, fields, is_dataclass

this_module = sys.modules[__name__]
class _Ast(ast_utils.Ast):
    pass

@dataclass
class Program(_Ast):
    rule_list: 'RuleList'

@dataclass
class RuleList(_Ast):
    rules: List['_Rule']

@dataclass
class _Rule(_Ast):
    filter_op: '_Filter_Op'
    transforms: 'Transforms'

@dataclass
class Transforms(_Ast, ast_utils.AsList):
    transforms: List['Transform']

@dataclass
class _Filter_Op(_Ast):
    pass

class Transform(_Ast):
    pass

@dataclass
class Color(_Ast):
    value: str

@dataclass
class Direction(_Ast):
    value: str

@dataclass
class Size(_Ast):
    value: str

@dataclass
class Degree(_Ast):
    value: str

@dataclass
class Symmetry_Axis(_Ast):
    value: str

@dataclass
class Rotation_Angle(_Ast):
    value: str

@dataclass
class UpdateColor(Transform):
    color: Color

@dataclass
class MoveNode(Transform):
    direction: Direction

@dataclass
class ExtendNode(Transform):
    direction: Direction
    overlap: Optional[bool] = False

@dataclass
class MoveNodeMax(Transform):
    direction: Direction

@dataclass
class AddBorder(Transform):
    color: Color

@dataclass
class HollowRectangle(Transform):
    color: Color

@dataclass
class FillRectangle(Transform):
    color: Color
    overlap: bool

@dataclass 
class Flip(Transform):
    symmetry_axis: Symmetry_Axis

@dataclass
class RotateNode(Transform):
    rotation_angle: Rotation_Angle

@dataclass
class Mirror(Transform):
    axis_point: Tuple[Optional[int], Optional[int]]

@dataclass
class Not(_Filter_Op):
    not_filter: _Filter_Op

@dataclass
class And:
    filters: List[_Filter_Op]

@dataclass
class Or:
    filters: List[_Filter_Op]

@dataclass
class FilterByColor(_Filter_Op):
    color: Color

@dataclass
class FilterByNeighborColor(_Filter_Op):
    color: Color

@dataclass
class FilterBySize(_Filter_Op):
    size: Size

@dataclass
class FilterByNeighborSize(_Filter_Op):
    size: Size

@dataclass
class FilterByDegree(_Filter_Op):
    degree: Degree

@dataclass
class FilterByNeighborDegree(_Filter_Op):
    degree: Degree

class ToAst(Transformer):
    def program(self, rule_list):
        return Program(rule_list=rule_list)

    def rule_list(self, *rules):
        return RuleList(rules=list(rules))

    @v_args(inline=True)
    def rule(self, filter_op, transforms):
        return _Rule(filter_op=filter_op, transforms=transforms)

    @v_args(inline=True)
    def color(self, color_token):
        return Color(value=str(color_token))

    @v_args(inline=True)
    def direction(self, direction_token):
        return Direction(value=str(direction_token))

    @v_args(inline=True)
    def size(self, size_token):
        return Size(value=str(size_token))

    @v_args(inline=True)
    def symmetry_axis(self, axis_token):
        return Symmetry_Axis(value=str(axis_token))

    @v_args(inline=True)
    def rotation_angle(self, angle_token):
        return Rotation_Angle(value=str(angle_token))

    def mirror_params(self, params):
        axis1, axis2 = params
        x = None if axis1 == "null" else int(axis1)
        y = None if axis2 == "null" else int(axis2)
        return Mirror(axis_point=(x, y))
    
    @v_args(inline=True)
    def bool_expr(self, value):
        return bool(value)

    def transforms(self, transforms):
        return Transforms(transforms=transforms)

    def transform(self, operator):
        if operator[0] == "update_color":
            return UpdateColor(color=operator[1])
        elif operator[0] == "move_node":
            return MoveNode(direction=operator[1])
        elif operator[0] == "extend_node":
            overlap = False
            if len(operator) == 3:
                overlap = self.bool_expr(operator[2])
            return ExtendNode(direction=operator[1], overlap=overlap)
        elif operator[0] == "move_node_max":
            return MoveNodeMax(direction=operator[1])
        elif operator[0] == "add_border":
            return AddBorder(color=operator[1])
        elif operator[0] == "fill_rectangle":
            overlap = self.bool_expr(operator[2]) 
            return FillRectangle(color=operator[1], overlap=overlap)
        elif operator[0] == "hollow_rectangle":
            return HollowRectangle(color=operator[1])
        elif operator[0] == "rotate_node":
            return RotateNode(rotation_angle=operator[1])
        elif operator[0] == "mirror":
            return Mirror(axis_point=operator[1])
        elif operator[0] == "flip":
            return Flip(symmetry_axis=operator[1])
        else:
            raise ValueError(f"Unknown operation: {operator}")

    def filter_op(self, operator):
        if operator[0] == "filter_by_color":
            return FilterByColor(color=operator[1])
        elif operator[0] == "filter_by_neighbor_color":
            return FilterByNeighborColor(color=operator[1])
        elif operator[0] == "filter_by_size":
            return FilterBySize(size=operator[1])
        elif operator[0] == "filter_by_neighbor_size":
            return FilterByNeighborSize(size=operator[1])
        elif operator[0] == "filter_by_degree":
            return FilterByDegree(degree=operator[1])
        elif operator[0] == "filter_by_neighbor_degree":
            return FilterByNeighborDegree(degree=operator[1])
        elif operator[0] == "not":
            return Not(not_filter=operator[1])
        elif operator[0] == "and":
            return And(filters=operator[1:])
        elif operator[0] == "or":
            return Or(filters=operator[1:])
        else:
            raise ValueError(f"Unknown operation: {operator}")

def print_ast_class_names(node, indent=0):
    indent_str = '    ' * indent
    if is_dataclass(node):
        # Print class name
        print(f"{indent_str}{node.__class__.__name__}", end='')
        if hasattr(node, 'value'):
            print(f"(value='{node.value}')", end='')
        for field in fields(node):
            field_value = getattr(node, field.name)
            if isinstance(field_value, _Ast) or isinstance(field_value, list):
                print_ast_class_names(field_value, indent + 1)
    elif isinstance(node, list):
        for item in node:
            print_ast_class_names(item, indent + 1)

if __name__ == '__main__':
    with open("arga_dsl.lark", "r") as f:
        arga_dsl_grammar = f.read()
    ast_parser = Lark(arga_dsl_grammar, start="program", parser="lalr", transformer=ToAst())
    with open("08ed6ac7.dsl", "r") as f:
        program = f.read()
    ast_program = ast_parser.parse(program)
    print(ast_program)