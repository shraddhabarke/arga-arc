from enum import Enum
from typing import Union, List, Dict
import typing

class FilterTypes(Enum):
    FILTERS = "Filters"
    #FILTER_OPS = "Filter_Ops"
    COLOR = "Color"
    SIZE = "Size"
    EXCLUDE = "Exclude"
    DEGREE = "Degree"

class FilterASTNode:
    def __init__(self, children = None):
        self.nodeType: FilterTypes
        self.code: str = self.__class__.__name__
        self.size: int = 1
        self.children: List[FilterASTNode] = children if children else []
        self.childTypes: List[FilterTypes] = []

class SizeValue:
    def __init__(self, enum_value):
        self.nodeType = FilterTypes.SIZE
        self.code = f"SIZE.{enum_value.name}"
        self.size = 1
        self.children = []

class Size:
    _all_values = set()

    def __new__(cls, enum_value):
        instance = SizeValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    @property
    def arity(cls):
        return 0

    def apply(self, children=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls._all_values)

class DegreeValue:
    def __init__(self, enum_value):
        self.nodeType = FilterTypes.DEGREE
        self.code = f"DEGREE.{enum_value.name}"
        self.size = 1
        self.children = []

class Degree:
    _all_values = set()

    def __new__(cls, enum_value):
        instance = DegreeValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    @property
    def arity(cls):
        return 0

    def apply(self, children=None):
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls._all_values)

def setup_size_and_degree_based_on_task(task):
    task_sizes = [w for w in task.object_sizes[task.abstraction]]
    _size_additional = {f'S{item}': int(item) for item in task_sizes}
    SizeEnum = Enum("SizeEnum", {'MIN': "MIN", 'MAX': "MAX", 'ODD': "ODD", **_size_additional})

    task_degrees = [d for d in task.object_degrees[task.abstraction]]
    _degree_additional = {f'D{item}': int(item) for item in task_degrees}
    DegreeEnum = Enum("DegreeEnum", {'MIN': "min", 'MAX': "max", 'ODD': "odd", **_degree_additional})

    for name, member in SizeEnum.__members__.items():
        setattr(Size, name, Size(member))
    for name, member in DegreeEnum.__members__.items():
        setattr(Degree, name, Degree(member))
    
    Size._enum_members = SizeEnum
    Degree._enum_members = DegreeEnum

class Color(FilterASTNode, Enum):
    C0 = 0
    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4
    C5 = 5
    C6 = 6
    C7 = 7
    C8 = 8
    C9 = 9
    LEAST = "least"
    MOST = "most"

    def __init__(self, value=None):
        super().__init__(FilterTypes.COLOR)
        self.nodeType = FilterTypes.COLOR
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

class Exclude(FilterASTNode, Enum):
    TRUE = True
    FALSE = False

    def __init__(self, value=None):
        super().__init__(FilterTypes.EXCLUDE)
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.nodeType = FilterTypes.EXCLUDE
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

class Filters(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]
    def __init__(self, filters: Union['And', 'Or', 'Filters'] = None):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filters] if filters else []
        self.code = filters.code if filters else ''
        self.size = filters.size if filters else 0
        self.childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class And(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]
    def __init__(self, filter1: Filters, filter2: Filters):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filter1, filter2]
        self.code = f"And({filter1.code}, {filter2.code})"
        self.size = 1 + filter1.size + filter2.size
        self.childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]

    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class Or(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]
    def __init__(self, filter1: Filters, filter2: Filters):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filter1, filter2]
        self.code = f"Or({filter1.code}, {filter2.code})"
        self.size = 1 + filter1.size + filter2.size
        self.childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterByColor(Filters):
    arity = 2
    childTypes = [FilterTypes.COLOR, FilterTypes.EXCLUDE]
    def __init__(self, color: Color, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByColor({color.code}, {exclude.code})"
        self.size = 1 + color.size + exclude.size
        self.children = [color, exclude]
        self.childTypes = [FilterTypes.COLOR, FilterTypes.EXCLUDE]

    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterBySize(Filters):
    arity = 2
    childTypes = [FilterTypes.SIZE, FilterTypes.EXCLUDE]
    def __init__(self, size: Size, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterBySize({size.code}, {exclude.code})"
        self.size = 1 + size.size + exclude.size
        self.children = [size, exclude]
        self.childTypes = [FilterTypes.SIZE, FilterTypes.EXCLUDE]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterByDegree(Filters):
    arity = 2
    childTypes = [FilterTypes.DEGREE, FilterTypes.EXCLUDE]
    def __init__(self, degree: Degree, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByDegree({degree.code}, {exclude.code})"
        self.size = 1 + degree.size + exclude.size
        self.children = [degree, exclude]
        self.childTypes = [FilterTypes.DEGREE, FilterTypes.EXCLUDE]

    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterByNeighborSize(Filters):
    arity = 2
    childTypes = [FilterTypes.SIZE, FilterTypes.EXCLUDE]
    def __init__(self, size: Size, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByNeighborSize({size.code}, {exclude.code})"
        self.size = 1 + size.size + exclude.size
        self.children = [size, exclude]
        self.childTypes = [FilterTypes.SIZE, FilterTypes.EXCLUDE]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterByNeighborColor(Filters):
    arity = 2
    childTypes = [FilterTypes.COLOR, FilterTypes.EXCLUDE]
    def __init__(self, color: Color, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByNeighborColor({color.code}, {exclude.code})"
        self.size = 1 + color.size + exclude.size
        self.children = [color, exclude]
        self.childTypes = [FilterTypes.COLOR, FilterTypes.EXCLUDE]
    
    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])

class FilterByNeighborDegree(Filters):
    arity = 2
    childTypes = [FilterTypes.DEGREE, FilterTypes.EXCLUDE]
    def __init__(self, degree: Degree, exclude: Exclude):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByNeighborDegree({degree.code}, {exclude.code})"
        self.size = 1 + degree.size + exclude.size
        self.children = [degree, exclude]
        self.childTypes = [FilterTypes.DEGREE, FilterTypes.EXCLUDE]

    @classmethod
    def apply(cls, children):
        return cls(children[0], children[1])