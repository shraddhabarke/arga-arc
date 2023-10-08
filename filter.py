from enum import Enum
from typing import Union, List
from task import *

taskNumber = "bb43febb"
task = Task("dataset/" + taskNumber + ".json")
task.abstraction = "nbccg"
task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                               input in task.train_input]
task.get_static_object_attributes(task.abstraction)
task_sizes = [w for w in task.object_sizes[task.abstraction]]
task_degree = [d for d in task.object_degrees[task.abstraction]]
print(task_degree)
class FilterTypes(Enum):
    FILTERS = "Filters"
    FILTER_OPS = "Filter_Ops"
    COLOR = "Color"
    SIZE = "Size"
    EXCLUDE = "Exclude"
    DEGREE = "Degree"

class FilterASTNode:
    def __init__(self, node_type: FilterTypes):
        self.nodeType: FilterTypes = node_type
        self.code: str = self.__class__.__name__
        self.size: int = 1
        self.children: List[FilterASTNode] = []

print(task_sizes)
class Color(FilterASTNode, Enum):
    C0 = "0"
    C1 = "1"
    C2 = "2"
    C3 = "3"
    C4 = "4"
    C5 = "5"
    C6 = "6"
    C7 = "7"
    C8 = "8"
    C9 = "9"
    LEAST = "least"
    MOST = "most"

    def __init__(self, value):
        super().__init__(FilterTypes.COLOR)
        self.code = value

class SizeValue:
    def __init__(self, value):
        self.nodeType = FilterTypes.SIZE
        self.code = value
        self.size = 1
        self.children = []

_additional = {f'S{item}': str(item) for item in task_sizes}
SizeEnum = Enum("SizeEnum", {'MIN': "min", 'MAX': "max", 'ODD': "odd", **_additional})
class Size:
    def __new__(cls, enum_value):
        instance = SizeValue(enum_value.value)
        return instance
for name, member in SizeEnum.__members__.items():
    setattr(Size, name, Size(member))

class Exclude(FilterASTNode, Enum):
    TRUE = "True"
    FALSE = "False"

    def __init__(self, value):
        super().__init__(FilterTypes.EXCLUDE)
        self.code = value

class DegreeValue:
    def __init__(self, value):
        self.nodeType = FilterTypes.DEGREE
        self.code = value
        self.size = 1
        self.children = []

_additional = {f'D{item}': str(item) for item in task_degree}
DegreeEnum = Enum("SizeEnum", {'MIN': "min", 'MAX': "max", 'ODD': "odd", **_additional})
class Degree:
    def __new__(cls, enum_value):
        instance = DegreeValue(enum_value.value)
        return instance
for name, member in DegreeEnum.__members__.items():
    setattr(Degree, name, Degree(member))

class Filters(FilterASTNode):
    def __init__(self, filters: Union["FilterOps", "And", "Or"]):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filters]
        self.code = filters.code
        self.size = filters.size

class FilterOps(FilterASTNode):
    def __init__(self, operation, *params):
        super().__init__(FilterTypes.FILTER_OPS)
        self.children = list(params)
        self.code = f"{operation}({', '.join([param.code for param in params])})"
        self.size = sum(param.size for param in params)

class And(FilterASTNode):
    def __init__(self, filter1: Filters, filter2: Filters):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filter1, filter2]
        self.code = f"And({filter1.code}, {filter2.code})"
        self.size = 1 + filter1.size + filter2.size

class Or(FilterASTNode):
    def __init__(self, filter1: Filters, filter2: Filters):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filter1, filter2]
        self.code = f"Or({filter1.code}, {filter2.code})"
        self.size = 1 + filter1.size + filter2.size

class FilterByColor(FilterOps):
    def __init__(self, color: Color, exclude: Exclude):
        super().__init__("filter_by_color", color, exclude)
        self.code = f"filter_by_color({color.code}, {exclude.code})"
        self.size = 1 + color.size + exclude.size

class FilterBySize(FilterOps):
    def __init__(self, size: Size, exclude: Exclude):
        super().__init__("filter_by_size", size, exclude)
        self.code = f"filter_by_size({size.code}, {exclude.code})"
        self.size = 1 + size.size + exclude.size

class FilterByDegree(FilterOps):
    def __init__(self, degree: Degree, exclude: Exclude):
        super().__init__("filter_by_degree", degree, exclude)
        self.code = f"filter_by_degree({degree.code}, {exclude.code})"
        self.size = 1 + degree.size + exclude.size

class FilterByNeighborSize(FilterOps):
    def __init__(self, size: Size, exclude: Exclude):
        super().__init__("filter_by_neighbor_size", size, exclude)
        self.code = f"filter_by_neighbor_size({size.code}, {exclude.code})"
        self.size = 1 + size.size + exclude.size

class FilterByNeighborColor(FilterOps):
    def __init__(self, color: Color, exclude: Exclude):
        super().__init__("filter_by_neighbor_color", color, exclude)
        self.code = f"filter_by_neighbor_color({color.code}, {exclude.code})"
        self.size = 1 + color.size + exclude.size

class FilterByNeighborDegree(FilterOps):
    def __init__(self, degree: Degree, exclude: Exclude):
        super().__init__("filter_by_neighbor_degree", degree, exclude)
        self.code = f"filter_by_neighbor_degree({degree.code}, {exclude.code})"
        self.size = 1 + degree.size + exclude.size

import unittest

class TestFilterGrammarRepresentation(unittest.TestCase):

    def test_color_enum(self):
        color_instance = Color.C0
        self.assertEqual(color_instance.nodeType, FilterTypes.COLOR)
        self.assertEqual(color_instance.code, "0")
        self.assertEqual(color_instance.size, 1)
        self.assertEqual(color_instance.children, [])

    def test_size_enum(self):
        size_instance = Size.MIN
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "min")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])
    
    def test_size_enum_dyn(self):
        size_instance = Size.S15
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "15")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])

    def test_degree_enum(self):
        degree_instance = Degree.D1
        self.assertEqual(degree_instance.nodeType, FilterTypes.DEGREE)
        self.assertEqual(degree_instance.code, "1")
        self.assertEqual(degree_instance.size, 1)
        self.assertEqual(degree_instance.children, [])

    def test_exclude_enum(self):
        exclude_instance = Exclude.TRUE
        self.assertEqual(exclude_instance.nodeType, FilterTypes.EXCLUDE)
        self.assertEqual(exclude_instance.code, "True")
        self.assertEqual(exclude_instance.size, 1)
        self.assertEqual(exclude_instance.children, [])

    def test_filter_by_color(self):
        filter_instance = FilterByColor(Color.C1, Exclude.TRUE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "filter_by_color(1, True)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_size(self):
        filter_instance = FilterBySize(Size.MIN, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "filter_by_size(min, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_size_enum(self):
        filter_instance = FilterBySize(Size.S25, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "filter_by_size(25, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_degree(self):
        filter_instance = FilterByDegree(Degree.MAX, Exclude.TRUE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "filter_by_degree(max, True)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_and_operator(self):
        filter1 = FilterByColor(Color.C1, Exclude.TRUE)
        filter2 = FilterByDegree(Degree.MAX, Exclude.TRUE)
        and_instance = And(Filters(filter1), Filters(filter2))
        self.assertEqual(and_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(and_instance.code, "And(filter_by_color(1, True), filter_by_degree(max, True))")
        self.assertEqual(and_instance.size, 7)
        self.assertEqual(len(and_instance.children), 2)

    def test_or_operator(self):
        filter1 = FilterByColor(Color.C1, Exclude.TRUE)
        filter2 = FilterBySize(Size.MIN, Exclude.FALSE)
        or_instance = Or(Filters(filter1), Filters(filter2))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(filter_by_color(1, True), filter_by_size(min, False))")
        self.assertEqual(or_instance.size, 7)
        self.assertEqual(len(or_instance.children), 2)

    def test_filter_by_neighbor_color(self):
        filter_instance = FilterByNeighborColor(Color.C3, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "filter_by_neighbor_color(3, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_neighbor_size(self):
        filter_instance = FilterByNeighborSize(Size.ODD, Exclude.TRUE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "filter_by_neighbor_size(odd, True)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_neighbor_degree(self):
        filter_instance = FilterByNeighborDegree(Degree.MIN, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "filter_by_neighbor_degree(min, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filters_representation(self):
        filter_instance = FilterByColor(Color.C5, Exclude.TRUE)
        filters_instance = Filters(filter_instance)
        self.assertEqual(filters_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(filters_instance.code, "filter_by_color(5, True)")
        self.assertEqual(filters_instance.size, 3)
        self.assertEqual(len(filters_instance.children), 1)

    def test_complex_and_operator(self):
        filter1 = FilterByNeighborSize(Size.ODD, Exclude.TRUE)
        filter2 = FilterByNeighborDegree(Degree.MIN, Exclude.FALSE)
        filter3 = FilterByColor(Color.C1, Exclude.TRUE)
        and_instance = And(Filters(filter1), And(Filters(filter2), Filters(filter3)))
        self.assertEqual(and_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(and_instance.code, "And(filter_by_neighbor_size(odd, True), And(filter_by_neighbor_degree(min, False), filter_by_color(1, True)))")
        self.assertEqual(and_instance.size, 11)
        self.assertEqual(len(and_instance.children), 2)

    def test_complex_or_operator(self):
        filter1 = FilterBySize(Size.MIN, Exclude.FALSE)
        filter2 = FilterByDegree(Degree.MAX, Exclude.TRUE)
        filter3 = FilterByNeighborColor(Color.C7, Exclude.FALSE)
        or_instance = Or(Filters(filter1), Or(Filters(filter2), Filters(filter3)))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(filter_by_size(min, False), Or(filter_by_degree(max, True), filter_by_neighbor_color(7, False)))")
        self.assertEqual(or_instance.size, 11)
        self.assertEqual(len(or_instance.children), 2)

if __name__ == "__main__":
    unittest.main()