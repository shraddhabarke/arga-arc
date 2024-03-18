from lark import Lark, ast_utils, Transformer, v_args, ParseTree
from typing import List, Any, Optional, Tuple
from lark import Tree
import sys
from dataclasses import dataclass, fields, is_dataclass
import typing as t
# from config import CONFIG
import os
import dsl.v0_3.parser as dsl_parser
from enum import Enum

this_module = sys.modules[__name__]

# class _Ast(ast_utils.Ast):
#     pass

# @dataclass
# class Library(_Ast):
#     programs: List["Program"]


# @dataclass
# class Program(_Ast):
#     rules: List["_Rule"]


# @dataclass
# class DoOperation(_Ast):
#     rule_list: "RuleList"


# @dataclass
# class RuleList(_Ast):
#     rules: List["_Rule"]


# @dataclass
# class _Rule(_Ast):
#     filter_op: "_Filter_Op"
#     transforms: "Transforms"


# @dataclass
# class Transforms(_Ast, ast_utils.AsList):
#     transforms: List["Transform"]


# @dataclass
# class _Filter_Op(_Ast):
#     pass


# class Transform(_Ast):
#     pass


# @dataclass
# class Color(_Ast):
#     value: str


# @dataclass
# class Direction(_Ast):
#     value: str


# @dataclass
# class Size(_Ast):
#     value: str


# @dataclass
# class Degree(_Ast):
#     value: str


# @dataclass
# class Symmetry_Axis(_Ast):
#     value: str


# @dataclass
# class Rotation_Angle(_Ast):
#     value: str


# @dataclass
# class UpdateColor(Transform):
#     color: Color


# @dataclass
# class MoveNode(Transform):
#     direction: Direction


# @dataclass
# class ExtendNode(Transform):
#     direction: Direction
#     overlap: Optional[bool] = False


# @dataclass
# class MoveNodeMax(Transform):
#     direction: Direction


# @dataclass
# class AddBorder(Transform):
#     color: Color


# @dataclass
# class HollowRectangle(Transform):
#     color: Color


# @dataclass
# class FillRectangle(Transform):
#     color: Color
#     overlap: bool


# @dataclass
# class Flip(Transform):
#     symmetry_axis: Symmetry_Axis


# @dataclass
# class RotateNode(Transform):
#     rotation_angle: Rotation_Angle


# @dataclass
# class Mirror(Transform):
#     axis_point: Tuple[Optional[int], Optional[int]]


# @dataclass
# class Not(_Filter_Op):
#     not_filter: _Filter_Op


# @dataclass
# class And:
#     filters: List[_Filter_Op]


# @dataclass
# class Or:
#     filters: List[_Filter_Op]


# @dataclass
# class FilterByColor(_Filter_Op):
#     color: Color


# @dataclass
# class FilterByNeighborColor(_Filter_Op):
#     color: Color


# @dataclass
# class FilterBySize(_Filter_Op):
#     size: Size


# @dataclass
# class FilterByNeighborSize(_Filter_Op):
#     size: Size


# @dataclass
# class FilterByDegree(_Filter_Op):
#     degree: Degree


# @dataclass
# class FilterByNeighborDegree(_Filter_Op):
#     degree: Degree


# class ToAst(Transformer):
#     def do_operation(self, rule_list) -> DoOperation:
#         return DoOperation(rule_list=rule_list)

#     @v_args(inline=True)
#     def rule_list(self, *rules) -> RuleList:
#         return RuleList(rules=list(rules))

#     def rule(self, args) -> _Rule:
#         filter_op, transforms = args[0], args[1] if len(args) > 1 else []
#         return _Rule(filter_op=filter_op, transforms=transforms)

#     def transforms(self, transforms) -> Transforms:
#         return Transforms(transforms=transforms)

#     @v_args(inline=True)
#     def color(self, color_token) -> Color:
#         return Color(value=str(color_token))

#     @v_args(inline=True)
#     def direction(self, direction_token) -> Direction:
#         return Direction(value=str(direction_token))

#     @v_args(inline=True)
#     def size(self, size_token) -> Size:
#         return Size(value=str(size_token))

#     @v_args(inline=True)
#     def degree(self, degree_token) -> Degree:
#         return Degree(value=str(degree_token))

#     @v_args(inline=True)
#     def symmetry_axis(self, axis_token) -> Symmetry_Axis:
#         return Symmetry_Axis(value=str(axis_token))

#     @v_args(inline=True)
#     def rotation_angle(self, angle_token) -> Rotation_Angle:
#         return Rotation_Angle(value=str(angle_token))

#     def mirror_params(self, params) -> Mirror:
#         axis1, axis2 = params
#         x = None if axis1 == "null" else int(axis1)
#         y = None if axis2 == "null" else int(axis2)
#         return Mirror(axis_point=(x, y))

#     @v_args(inline=True)
#     def bool_expr(self, value) -> bool:
#         return bool(value)

#     def transform_op(
#         self, operator
#     ) -> (
#         UpdateColor
#         | MoveNode
#         | ExtendNode
#         | MoveNodeMax
#         | AddBorder
#         | FillRectangle
#         | HollowRectangle
#         | RotateNode
#         | Mirror
#         | Flip
#     ):
#         if operator[0] == "update_color":
#             return UpdateColor(color=self.color(operator[1]))
#         elif operator[0] == "move_node":
#             return MoveNode(direction=self.direction(operator[1]))
#         elif operator[0] == "extend_node":
#             overlap = False
#             if len(operator) == 3:
#                 overlap = self.bool_expr(operator[2])
#             return ExtendNode(direction=self.direction(operator[1]), overlap=overlap)
#         elif operator[0] == "move_node_max":
#             return MoveNodeMax(direction=self.direction(operator[1]))
#         elif operator[0] == "add_border":
#             return AddBorder(color=self.color(operator[1]))
#         elif operator[0] == "fill_rectangle":
#             overlap = self.bool_expr(operator[2])
#             return FillRectangle(color=self.color(operator[1]), overlap=operator[2])
#         elif operator[0] == "hollow_rectangle":
#             return HollowRectangle(color=self.color(operator[1]))
#         elif operator[0] == "rotate_node":
#             return RotateNode(rotation_angle=self.rotation_angle(operator[1]))
#         elif operator[0] == "mirror":
#             # TODO: not sure how to fix this transformation
#             return Mirror(axis_point=operator[1])
#         elif operator[0] == "flip":
#             return Flip(symmetry_axis=self.symmetry_axis(operator[1]))
#         else:
#             raise ValueError(f"Unknown operation: {operator}")

#     def filter_op(
#         self, operator
#     ) -> (
#         FilterByColor
#         | FilterByNeighborColor
#         | FilterBySize
#         | FilterByNeighborSize
#         | FilterByDegree
#         | FilterByNeighborDegree
#         | Not
#         | And
#         | Or
#     ):
#         if operator[0] == "filter_by_color":
#             return FilterByColor(color=self.color(operator[1]))
#         elif operator[0] == "filter_by_neighbor_color":
#             return FilterByNeighborColor(color=self.color(operator[1]))
#         elif operator[0] == "filter_by_size":
#             return FilterBySize(size=self.size(operator[1]))
#         elif operator[0] == "filter_by_neighbor_size":
#             return FilterByNeighborSize(size=self.size(operator[1]))
#         elif operator[0] == "filter_by_degree":
#             return FilterByDegree(degree=self.degree(operator[1]))
#         elif operator[0] == "filter_by_neighbor_degree":
#             return FilterByNeighborDegree(degree=self.degree(operator[1]))
#         elif operator[0] == "not":
#             return Not(not_filter=operator[1])
#         elif operator[0] == "and":
#             return And(filters=operator[1:])
#         elif operator[0] == "or":
#             return Or(filters=operator[1:])
#         else:
#             raise ValueError(f"Unknown operation: {operator}")

@dataclass
class _Ast(ast_utils.Ast):
    pass

# @dataclass
# class _Xform(_Ast):
#     children: List[_Ast]

# class XformList(_Ast):
#     children: List[_Ast]

@dataclass
class Color(_Ast):
    value: str

@dataclass
class Size(_Ast):
    value: int | str

@dataclass
class Direction(_Ast):
    value: str

@dataclass
class Overlap(_Ast):
    value: bool

### Filter relations

@dataclass
class _FilterRelation(_Ast):
    pass

@dataclass
class IsAnyNeighbor(_FilterRelation):
    pass

@dataclass 
class IsDirectNeighbor(_FilterRelation):
    pass

@dataclass
class IsDiagonalNeighbor(_FilterRelation):
    pass

### Filter expressions


@dataclass
class _FilterExpr(_Ast):
    pass

# @dataclass
# class Filter(_Ast):
#     filter_expr: _FilterExpr

@dataclass
class And(_FilterExpr):
    left: _FilterExpr
    right: _FilterExpr

@dataclass
class Or(_Ast):
    # children: Tuple[_FilterExpr, _FilterExpr]
    left: _FilterExpr
    right: _FilterExpr

@dataclass
class Not(_Ast):
    child: _FilterExpr

@dataclass
class VarAnd(_Ast):
    relation: _FilterRelation
    filter: _FilterExpr

### Filter primitives

@dataclass
class FilterByColor(_Ast):
    color: Color

@dataclass
class FilterByNeighborSize(_Ast):
    size: str

### Transforms

@dataclass
class ExtendNode(_Ast):
    direction: Direction
    overlap: Overlap

@dataclass
class Rule(_Ast):
    filter: _FilterExpr
    xforms: List[_Ast]

@dataclass
class Program(_Ast):
    rules: List[Rule]

@dataclass
class Library(_Ast):
    programs: List[Program]


class ToAst(Transformer):
    # def __default_token__(self, token):
    #     return token.value
    
    def VAR(self, token):
        return token.value
    
    def OVERLAP(self, token):
        return Overlap(bool(token.value))
    
    def COLOR(self, token):
        return Color(token.value)

    def SIZE(self, token):
        try:
            return Size(int(token.value))
        except ValueError:
            return Size(token.value)
    
    def filter(self, children):
        if len(children) == 1:
            return children[0]
        else:
            None
    
    def filter_expr(self, children):
        # if this is a single primitive
        if len(children) == 1:
            return children[0]
        else:
            match children[0]:
                case "and":
                    return And(children[1], children[2])
                case "or":
                    return Or((children[1], children[2]))
                case "not":
                    return Not(children[1])
                case "varand":
                    return VarAnd(
                        relation=children[1],
                        filter=children[2]
                    )
                case _:
                    # return children
                    raise ValueError(f"Unknown filter expression: {children[0]}")

        return children

    @v_args(tree=True)
    def filter_prim(self, tree):
        children = tree.children
        match children[0]:
            case "filter_by_color":
                return FilterByColor(
                    color=children[1]
                )
            case "filter_by_neighbor_size":
                return FilterByNeighborSize(
                    size=children[1]
                ) 
            case _:
                # return tree
                raise ValueError(f"Unknown filter primitive: {children[0]}")
    
    def filter_relation(self, children):
        match children[0]:
            case "is_direct_neighbor":
                return IsDirectNeighbor()
            case _:
                raise ValueError(f"Unknown filter relation: {children[0]}")
    
    def xform_list(self, children):
        return children

    def xform(self, children):
        match children[0]:
            case "extend_node":
                return ExtendNode(
                    direction=self.direction(children[1]), 
                    overlap=children[2]
                )

def print_ast_class_names(node, indent=0):
    indent_str = "    " * indent
    if is_dataclass(node):
        # Print class name
        print(f"{indent_str}{node.__class__.__name__}", end="")
        if hasattr(node, "value"):
            print(f"(value='{node.value}')", end="")
        for field in fields(node):
            field_value = getattr(node, field.name)
            if isinstance(field_value, _Ast) or isinstance(field_value, list):
                print_ast_class_names(field_value, indent + 1)
    elif isinstance(node, list):
        for item in node:
            print_ast_class_names(item, indent + 1)


# GRAMMAR: t.Optional[Lark] = None


# def __ensure_grammar() -> Lark:
#     global GRAMMAR
#     grammar_file = CONFIG.ROOT_DIR / "dsl" / "dsl.lark"
#     if GRAMMAR is None:
#         with open(grammar_file, "r") as f:
#             arga_dsl_grammar = f.read()
#         GRAMMAR = Lark(
#             arga_dsl_grammar, start="start", parser="lalr", transformer=ToAst()
#         )
#     return GRAMMAR


# def parse(program: str) -> ParseTree:
#     return __ensure_grammar().parse(program)

def test_file(filename, parser, xformer):
    with open(filename, "r") as f:
        lib = "(" + f.read() + ")"
    print(f"Testing {filename}...")
    t = parser.lib_parse_tree(lib)
    ast = xformer.transform(t)
    # print_ast_class_names(ast)
    print(ast)

if __name__ == "__main__":
    parser = dsl_parser.Parser.new()

    # grammar_file = "dsl/v0_3/dsl.lark"
    # with open(grammar_file, "r") as f:
    #     arga_dsl_grammar = f.read()
    # parser = Lark(
    #     arga_dsl_grammar, 
    #     start="library", 
    #     parser="lalr", 
    #     transformer=ToAst()
    # )

    xformer = ast_utils.create_transformer(this_module, ToAst())
    # xformer = ToAst()

    test_file("dsl/v0_3/reference/d43fd935.dsl", parser, xformer)

    # test_dir = "dsl/v0_3/reference"
    # test_dirs = [
    #     "dsl/v0_3/reference",
    #     "dsl/v0_3/examples",
    # ]
    # for test_dir in test_dirs:
    #     print(f"Testing directory {test_dir}...")
    #     for filename in os.listdir(test_dir):
    #         if filename.endswith(".dsl"):
    #             # open the file as a library
    #             with open(os.path.join(test_dir, filename), "r") as f:
    #                 lib = "(" + f.read() + ")"
    #             print(f"Testing {filename}...")
    #             # try:
    #             t = parser.lib_parse_tree(lib)
    #             # t = parser.parse(lib)
    #             # ast = xformer.transform(t)
    #             ast = ToAst().transform(t)
    #             print(ast)
    #             # except Exception as e:
    #             #     print(f"Error parsing {filename}: {e}")
    #             #     exit(1)
    # print("All tests passed!")