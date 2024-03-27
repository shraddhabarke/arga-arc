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
from pprint import pprint

this_module = sys.modules[__name__]

@dataclass
class _Ast(ast_utils.Ast):
    pass

### Var types

# @dataclass
# class Var():
#     pass

@dataclass
class VarUpdateColor(_Ast):
    pass

@dataclass
class VarMoveNode(_Ast):
    pass

@dataclass
class VarExtendNode(_Ast):
    pass

@dataclass
class VarMoveNodeMax(_Ast):
    pass

@dataclass
class VarMirror(_Ast):
    pass

@dataclass
class VarInsert(_Ast):
    pass

@dataclass
class ObjectId(_Ast):
    value: int

### Base types

@dataclass
class Size(_Ast):
    value: int | str

@dataclass
class Degree(_Ast):
    value: int | str

@dataclass
class Height(_Ast):
    value: int | str

@dataclass
class Width(_Ast):
    value: int | str

@dataclass
class Column(_Ast):
    value: int | str

@dataclass
class Shape(_Ast):
    value: str

@dataclass
class Direction(_Ast):
    value: str

@dataclass
class SymmetryAxis(_Ast):
    value: str

@dataclass
class RotationAngle(_Ast):
    value: int

@dataclass
class Color(_Ast):
    value: str

@dataclass
class Overlap(_Ast):
    value: bool

@dataclass
class ImagePoints(_Ast):
    value: str

@dataclass
class RelativePosition(_Ast):
    value: str

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
class Or(_FilterExpr):
    # children: Tuple[_FilterExpr, _FilterExpr]
    left: _FilterExpr
    right: _FilterExpr

@dataclass
class Not(_FilterExpr):
    child: _FilterExpr

@dataclass
class VarAnd(_FilterExpr):
    relation: _FilterRelation
    filter: _FilterExpr

### Filter primitives

@dataclass
class FilterByColor(_Ast):
    color: Color

@dataclass
class FilterBySize(_Ast):
    size: Size

@dataclass
class FilterByHeight(_Ast):
    height: Height

@dataclass
class FilterByWidth(_Ast):
    width: Width

@dataclass
class FilterByDegree(_Ast):
    degree: Degree

@dataclass
class FilterByShape(_Ast):
    shape: Shape

@dataclass
class FilterByColumns(_Ast):
    columns: Column

@dataclass
class FilterByNeighborSize(_Ast):
    size: Size

@dataclass
class FilterByNeighborColor(_Ast):
    color: Color

@dataclass
class FilterByNeighborDegree(_Ast):
    degree: Degree

@dataclass
class FilterByNeighborSize(_Ast):
    size: str

### Transforms

@dataclass
class _Transform(_Ast):
    pass

@dataclass
class UpdateColor(_Transform):
    color: Color | VarUpdateColor

@dataclass
class MoveNode(_Transform):
    direction: Direction | VarMoveNode

@dataclass
class ExtendNode(_Transform):
    direction: Direction | VarExtendNode
    overlap: Overlap

@dataclass
class MoveNodeMax(_Transform):
    direction: Direction | VarMoveNodeMax

@dataclass
class RotateNode(_Transform):
    angle: RotationAngle

@dataclass
class AddBorder(_Transform):
    color: Color

@dataclass
class FillRectangle(_Transform):
    color: Color
    overlap: Overlap

@dataclass
class HollowRectangle(_Transform):
    color: Color

@dataclass
class Mirror(_Transform):
    axis: VarMirror | SymmetryAxis

@dataclass
class Flip(_Transform):
    axis: SymmetryAxis

@dataclass
class Insert(_Transform):
    source: VarInsert | ObjectId
    image_points: ImagePoints
    relative_position: RelativePosition

@dataclass
class NoOp(_Transform):
    pass

###

@dataclass
class Rule(_Ast):
    filter: _FilterExpr
    transforms: List[_Transform]

@dataclass
class Program(_Ast, ast_utils.AsList):
    rules: List[Rule]

@dataclass
class Library(_Ast, ast_utils.AsList):
    programs: List[Program]


class ToAst(Transformer):
    # def VAR(self, token):
    #     return Var()

    # Var types

    def VAR_UPDATE_COLOR(self, token):
        return VarUpdateColor()
    
    def VAR_MOVE_NODE(self, token):
        return VarMoveNode()
    
    def VAR_EXTEND_NODE(self, token):
        return VarExtendNode()
    
    def VAR_MOVE_NODE_MAX(self, token):
        return VarMoveNodeMax()
    
    def VAR_MIRROR(self, token):
        return VarMirror()
    
    def VAR_INSERT(self, token):
        return VarInsert()

    def OBJECT_ID(self, token):
        return ObjectId(int(token.value))
    
    # Base types

    def OVERLAP(self, token):
        return Overlap(bool(token.value))
    
    def COLOR(self, token):
        return Color(token.value)

    def DIRECTION(self, token):
        return Direction(token.value)

    def SIZE(self, token):
        try:
            return Size(int(token.value))
        except ValueError:
            return Size(token.value)

    def ROT_ANGLE(self, token):
        return RotationAngle(int(token.value))

    def SHAPE(self, token):
        return Shape(token.value)

    def DEGREE(self, token):
        try:
            return Degree(int(token.value))
        except ValueError:
            return Degree(token.value)
    
    def HEIGHT(self, token):
        try:
            return Height(int(token.value))
        except ValueError:
            return Height(token.value)
    
    def COLUMN(self, token):
        try:
            return Column(int(token.value))
        except ValueError:
            return Column(token.value)

    def IMAGE_POINTS(self, token):
        return ImagePoints(token.value)
    
    def RELATIVE_POSITION(self, token):
        return RelativePosition(token.value)

    # def library(self, children):
    #     return Library(children)
    
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
                    return Or(children[1], children[2])
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
            case "filter_by_size":
                return FilterBySize(
                    size=children[1]
                )
            case "filter_by_height":
                return FilterByHeight(
                    height=children[1]
                )
            case "filter_by_width":
                return FilterByWidth(
                    width=children[1]
                )
            case "filter_by_degree":
                return FilterByDegree(
                    degree=children[1]
                )
            case "filter_by_shape":
                return FilterByShape(
                    shape=children[1]
                )
            case "filter_by_columns":
                return FilterByColumns(
                    columns=children[1]
                )
            case "filter_by_neighbor_size":
                return FilterByNeighborSize(
                    size=children[1]
                )
            case "filter_by_neighbor_color":
                return FilterByNeighborColor(
                    color=children[1]
                )
            case "filter_by_neighbor_degree":
                return FilterByNeighborDegree(
                    degree=children[1]
                )
            case _:
                # return tree
                raise ValueError(f"Unknown filter primitive: {children[0]}")
    
    def filter_relation(self, children):
        match children[0]:
            case "is_any_neighbor":
                return IsAnyNeighbor()
            case "is_direct_neighbor":
                return IsDirectNeighbor()
            case "is_diagonal_neighbor":
                return IsDiagonalNeighbor()
            case _:
                raise ValueError(f"Unknown filter relation: {children[0]}")
    
    def xform_list(self, children):
        return children

    def xform(self, children):
        match children[0]:
            case "update_color":
                return UpdateColor(
                    color=children[1]
                )
            case "move_node":
                return MoveNode(
                    direction=children[1]
                )

            case "extend_node":
                return ExtendNode(
                    direction=children[1], 
                    overlap=children[2]
                )
            case "move_node_max":
                return MoveNodeMax(
                    direction=children[1]
                )
            case "rotate_node":
                return RotateNode(
                    angle=children[1]
                )
            case "add_border":
                return AddBorder(
                    color=children[1]
                )
            case "fill_rectangle":
                return FillRectangle(
                    color=children[1],
                    overlap=children[2]
                )
            case "hollow_rectangle":
                return HollowRectangle(
                    color=children[1]
                )
            case "mirror":
                return Mirror(
                    axis=children[1]
                )
            case "flip":
                return Flip(
                    axis=children[1]
                )
            case "insert":
                return Insert(
                    source=children[1],
                    image_points=children[2],
                    relative_position=children[3]
                )
            case "noop":
                return NoOp()        

            case _:
                raise ValueError(f"Unknown transform: {children[0]}")
    

def print_ast_class_names(node, indent=0):
    indent_str = "    " * indent
    if is_dataclass(node):
        # Print class name
        print(f"{indent_str}{node.__class__.__name__}", end="\n")
        if hasattr(node, "value"):
            print(f"(value='{node.value}')", end="")
        for field in fields(node):
            field_value = getattr(node, field.name)
            if isinstance(field_value, _Ast) or isinstance(field_value, list):
                print_ast_class_names(field_value, indent + 1)
    elif isinstance(node, list):
        for item in node:
            print_ast_class_names(item, indent + 1)


def test_file(filename, parser, xformer):
    with open(filename, "r") as f:
        lib = "(" + f.read() + ")"
    print(f"Testing {filename}...")
    t = parser.lib_parse_tree(lib)
    # print(t.pretty())
    ast = xformer.transform(t)
    pprint(ast)

def test_gpt_gens():
    parser = dsl_parser.Parser.new()
    xformer = ast_utils.create_transformer(this_module, ToAst())

    # gens_dir = "/Users/emmanuel/repos/arc_stuff/arga-arc/models/logs/gens_20240318T224833"
    # gens_dir = "models/logs/gens_20240318T224833"
    # gens_dir = "models/logs/gens_20240318T230602"
    # gens_dir = "models/logs/gens_20240318T231857"
    gens_dir = "dsl/v0_3/generations"
    # iterate over the directories in gens_dir
    for dir in os.listdir(gens_dir):
        dir_path = os.path.join(gens_dir, dir)
        if os.path.isdir(dir_path):
            print(f"Testing directory {dir_path}...")
            for filename in os.listdir(dir_path):
                if filename.endswith("_valid.txt"):
                    file_path = os.path.join(dir_path, filename)
                    test_file(file_path, parser, xformer)


def test_reference_programs():
    parser = dsl_parser.Parser.new()
    xformer = ast_utils.create_transformer(this_module, ToAst())

    test_dirs = [
        "dsl/v0_3/reference",
        "dsl/v0_3/examples",
    ]
    for test_dir in test_dirs:
        print(f"Testing directory {test_dir}...")
        for filename in os.listdir(test_dir):
            if filename.endswith(".dsl"):
                file_path = os.path.join(test_dir, filename)
                # open the file as a library
                # print(f"Testing {file_path}...")
                test_file(file_path, parser, xformer)

if __name__ == "__main__":
    # test_gpt_gens()
    test_reference_programs()