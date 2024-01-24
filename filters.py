from enum import Enum
from typing import Union, List, Dict


class FilterTypes(Enum):
    FILTERS = "Filters"
    # FILTER_OPS = "Filter_Ops"
    COLOR = "FColor"
    SIZE = "Size"
    DEGREE = "Degree"
    RELATION = "Relation"


class FilterASTNode:
    def __init__(self, children=None):
        self.nodeType: FilterTypes
        self.code: str = self.__class__.__name__
        self.size: int = 1
        self.children: List[FilterASTNode] = children if children else []
        self.childTypes: List[FilterTypes] = []
        self.values = None


class Relation(FilterASTNode, Enum):
    neighbor = "Neighbor"  # todo: add more relations here

    def __init__(self, value=None):
        super().__init__(FilterTypes.RELATION)
        self.nodeType = FilterTypes.RELATION
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    @classmethod
    @property
    def nodeType(cls):
        return FilterTypes.RELATION

    def execute(self, task, children):
        if self.name == 'neighbor':
            self.values = [{
                node: [
                    neighbor for neighbor in input_graph.graph.neighbors(node)]
                for node in input_graph.graph.nodes()} for input_graph in task.input_abstracted_graphs_original[task.abstraction]]
            self.values = [{(5, 0): [(6, 0)],
                            (5, 1): [(2, 0)],
                            (5, 2): [(8, 0)]},

                           {(5, 0): [(1, 0)],
                            (5, 1): [(7, 0)],
                            (5, 2): [(4, 0)]},

                           {(5, 0): [(1, 0)],
                            (5, 1): [(7, 0)],
                            (5, 2): [(6, 0)]}]
            # todo: testing
        return self

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())


class SizeValue:
    arity = 0

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.SIZE
        self.code = f"SIZE.{enum_value.name}"
        self.size = 1
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls


class DegreeValue:
    arity = 0

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.DEGREE
        self.code = f"DEGREE.{enum_value.name}"
        self.size = 1
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls


class Size(FilterASTNode):
    _all_values = set()
    arity = 0
    nodeType = FilterTypes.SIZE

    def __new__(cls, enum_value):
        instance = SizeValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    def get_all_values(cls):
        return list(cls._enum_members)


class Degree(FilterASTNode):
    _all_values = set()
    arity = 0
    nodeType = FilterTypes.DEGREE

    def __new__(cls, enum_value):
        instance = DegreeValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    def get_all_values(cls):
        return list(cls._enum_members)


def setup_size_and_degree_based_on_task(task):
    task_sizes = [w for w in task.object_sizes[task.abstraction]]
    _size_additional = {f'{item}': int(item) for item in task_sizes}
    SizeEnum = Enum(
        "SizeEnum", {'MIN': "MIN", 'MAX': "MAX", 'ODD': "ODD", **_size_additional})

    task_degrees = [d for d in task.object_degrees[task.abstraction]]
    _degree_additional = {f'{item}': int(item) for item in task_degrees}
    DegreeEnum = Enum(
        "DegreeEnum", {'MIN': "MIN", 'MAX': "MAX", 'ODD': "ODD", **_degree_additional})
    _degrees, _sizes = [], []

    for name, member in SizeEnum.__members__.items():
        setattr(Size, name, SizeValue(member))
        _sizes.append(Size(member))
    for name, member in DegreeEnum.__members__.items():
        setattr(Degree, name, Degree(member))
        _degrees.append(Degree(member))

    Size._enum_members = _sizes
    Degree._enum_members = _degrees


class FColor(FilterASTNode, Enum):
    black = "O"
    blue = "B"
    red = "R"
    green = "G"
    yellow = "Y"
    grey = "X"
    fuchsia = "F"
    orange = "A"
    cyan = "C"
    brown = "W"

    def __init__(self, value=None):
        super().__init__(FilterTypes.COLOR)
        self.nodeType = FilterTypes.COLOR
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.size = 1
        self.children = []
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    @classmethod
    @property
    def nodeType(cls):
        return FilterTypes.COLOR

    def execute(cls, task, children):
        return cls

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


class And(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]
    default_size = 1

    def __init__(self, filter1: Filters, filter2: Filters):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filter1, filter2]
        self.code = f"And({filter1.code}, {filter2.code})"
        self.size = self.default_size + filter1.size + filter2.size
        self.childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]

    @classmethod
    def execute(cls, task, children):
        values1 = children[0].values
        values2 = children[1].values
        intersected_values = [list(set(v1).intersection(set(v2))) if set(
            v1).intersection(set(v2)) else [] for v1, v2 in zip(values1, values2)]
        new_instance = cls(children[0], children[1])
        new_instance.values = intersected_values
        return new_instance


class Or(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]
    default_size = 1

    def __init__(self, filter1: Filters, filter2: Filters):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filter1, filter2]
        self.code = f"Or({filter1.code}, {filter2.code})"
        self.size = self.default_size + filter1.size + filter2.size
        self.childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]

    @classmethod
    def execute(cls, task, children):
        values1 = children[0].values
        values2 = children[1].values
        unioned_values = [list(set(v1).union(set(v2)))
                          for v1, v2 in zip(values1, values2)]
        new_instance = cls(children[0], children[1])
        new_instance.values = unioned_values

        return new_instance


class Not(FilterASTNode):
    arity = 1
    nodeType = FilterTypes.FILTERS
    childTypes = [FilterTypes.FILTERS]
    default_size = 1

    def __init__(self, filter: Filters):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filter]
        self.code = f"Not({filter.code})"
        self.size = self.default_size + filter.size
        self.childTypes = [FilterTypes.FILTERS]

    @classmethod
    def execute(cls, task, children):
        values = children[0].values
        nodes_with_data = []
        # TODO: Optimize
        for input_abstracted_graphs in task.input_abstracted_graphs_original[task.abstraction]:
            local_data = []
            for node, _ in input_abstracted_graphs.graph.nodes(data=True):
                local_data.append(node)
            nodes_with_data.append(local_data)
        result = [[item for item in sublist1 if item not in sublist2]
                  for sublist1, sublist2 in zip(nodes_with_data, values)]
        new_instance = cls(children[0])
        new_instance.values = result
        return new_instance


class FilterByColor(Filters):
    arity = 1
    childTypes = [FilterTypes.COLOR]
    default_size = 1

    def __init__(self, color: FColor):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByColor({color.code})"
        self.size = self.default_size + color.size
        self.children = [color]
        self.childTypes = [FilterTypes.COLOR]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class FilterBySize(Filters):
    arity = 1
    childTypes = [FilterTypes.SIZE]
    default_size = 1

    def __init__(self, size: Size):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterBySize({size.code})"
        self.size = self.default_size + size.size
        self.children = [size]
        self.childTypes = [FilterTypes.SIZE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class FilterByDegree(Filters):
    arity = 1
    childTypes = [FilterTypes.DEGREE]
    default_size = 1

    def __init__(self, degree: Degree):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByDegree({degree.code})"
        self.size = self.default_size + degree.size
        self.children = [degree]
        self.childTypes = [FilterTypes.DEGREE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class FilterByNeighborSize(Filters):
    arity = 1
    childTypes = [FilterTypes.SIZE]
    default_size = 1

    def __init__(self, size: Size):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByNeighborSize({size.code})"
        self.size = self.default_size + size.size
        self.children = [size]
        self.childTypes = [FilterTypes.SIZE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class FilterByNeighborColor(Filters):
    arity = 1
    childTypes = [FilterTypes.COLOR]
    default_size = 1

    def __init__(self, color: FColor):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByNeighborColor({color.code})"
        self.size = self.default_size + color.size
        self.children = [color]
        self.childTypes = [FilterTypes.COLOR]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class FilterByNeighborDegree(Filters):
    arity = 1
    childTypes = [FilterTypes.DEGREE]
    default_size = 1

    def __init__(self, degree: Degree):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"FilterByNeighborDegree({degree.code})"
        self.size = self.default_size + degree.size
        self.children = [degree]
        self.childTypes = [FilterTypes.DEGREE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class FilterByRelation(Filters):
    arity = 2
    childTypes = [FilterTypes.RELATION, FilterTypes.FILTERS]
    default_size = 1

    def __init__(self, relation: Relation, filter: Filters):
        super().__init__()
        self.children = [relation, filter]
        self.code = f"∃y s.t y.({filter.code})"
        self.size = self.default_size + relation.size + filter.size
        self.childTypes = [FilterTypes.RELATION, FilterTypes.FILTERS]
        # todo: need to compute the subsets, all subsets need not have filters

    # check if the filter node satisfies any of the relational nodes
    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0], children[1])
        relation_instance, filter_instance = children
        total_value_dict = [
        {key: [val for val in value if val in filter_set] for key, value in rel_dict.items()}
        for filter_set, rel_dict in zip(filter_instance.values, relation_instance.values)]
        instance.values = [total_value_dict]
        return instance