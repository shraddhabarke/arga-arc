from enum import Enum
from typing import Union, List, Dict
import typing

class FilterTypes(Enum):
    FILTERS = "Filters"
    FILTER_OPS = "Filter_Ops"
    COLOR = "Color"
    SIZE = "Size"
    EXCLUDE = "Exclude"
    DEGREE = "Degree"

class FilterASTNode:
    def __init__(self, node_type: FilterTypes, task=None):
        self.nodeType: FilterTypes = node_type
        self.code: str = self.__class__.__name__
        self.size: int = 1
        self.children: List[FilterASTNode] = []

class SizeValue:
    def __init__(self, value):
        self.nodeType = FilterTypes.SIZE
        self.code = value
        self.size = 1
        self.children = []

class Size:
    def __new__(cls, enum_value):
        instance = SizeValue(enum_value.value)
        return instance

class DegreeValue:
    def __init__(self, value):
        self.nodeType = FilterTypes.DEGREE
        self.code = value
        self.size = 1
        self.children = []

class Degree:
    def __new__(cls, enum_value):
        instance = DegreeValue(enum_value.value)
        return instance

def setup_size_and_degree_based_on_task(task):
    task_sizes = [w for w in task.object_sizes[task.abstraction]]
    _size_additional = {f'S{item}': str(item) for item in task_sizes}
    SizeEnum = Enum("SizeEnum", {'MIN': "min", 'MAX': "max", 'ODD': "odd", **_size_additional})

    task_degrees = [d for d in task.object_degrees[task.abstraction]]
    _degree_additional = {f'D{item}': str(item) for item in task_degrees}
    DegreeEnum = Enum("DegreeEnum", {'MIN': "min", 'MAX': "max", 'ODD': "odd", **_degree_additional})

    # Set attributes for Size and Degree:
    for name, member in SizeEnum.__members__.items():
        setattr(Size, name, Size(member))
    for name, member in DegreeEnum.__members__.items():
        setattr(Degree, name, Degree(member))

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

class Exclude(FilterASTNode, Enum):
    TRUE = "True"
    FALSE = "False"

    def __init__(self, value):
        super().__init__(FilterTypes.EXCLUDE)
        self.code = value

class Filters(FilterASTNode):
    def __init__(self, filters: Union["FilterOps", "And", "Or"]):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filters]
        self.code = filters.code
        self.size = filters.size

class FilterOps(FilterASTNode):
    def __init__(self):
        super().__init__(FilterTypes.FILTER_OPS)

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
        super().__init__()
        self.nodeType = FilterTypes.FILTER_OPS
        self.code = f"FilterByColor({color.code}, {exclude.code})"
        self.size = 1 + color.size + exclude.size
        self.children = [color, exclude]

    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterBySize(FilterOps):
    def __init__(self, size: Size, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTER_OPS
        self.code = f"FilterBySize({size.code}, {exclude.code})"
        self.size = 1 + size.size + exclude.size
        self.children = [size, exclude]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterByDegree(FilterOps):
    def __init__(self, degree: Degree, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTER_OPS
        self.code = f"FilterByDegree({degree.code}, {exclude.code})"
        self.size = 1 + degree.size + exclude.size
        self.children = [degree, exclude]

class FilterByNeighborSize(FilterOps):
    def __init__(self, size: Size, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTER_OPS
        self.code = f"FilterByNeighborSize({size.code}, {exclude.code})"
        self.size = 1 + size.size + exclude.size
        self.children = [size, exclude]

class FilterByNeighborColor(FilterOps):
    def __init__(self, color: Color, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTER_OPS
        self.code = f"FilterByNeighborColor({color.code}, {exclude.code})"
        self.size = 1 + color.size + exclude.size
        self.children = [color, exclude]

class FilterByNeighborDegree(FilterOps):
    def __init__(self, degree: Degree, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTER_OPS
        self.code = f"FilterByNeighborDegree({degree.code}, {exclude.code})"
        self.size = 1 + degree.size + exclude.size
        self.children = [degree, exclude]
