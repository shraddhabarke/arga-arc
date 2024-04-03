from enum import Enum
from typing import Union, List, Dict
from transform import Dir


class FilterTypes(Enum):
    FILTERS = "Filters"
    COLOR = "FColor"
    SIZE = "Size"
    DEGREE = "Degree"
    SHAPE = "Shape"
    COLUMN = "Column"
    ROW = "Row"
    HEIGHT = "Height"
    WIDTH = "Width"

class FilterASTNode:
    def __init__(self, children=None):
        self.nodeType: FilterTypes
        self.code: str = self.__class__.__name__
        self._size: int = 1
        self.children: List[FilterASTNode] = children if children else []
        self.childTypes: List[FilterTypes] = []
        self.values = None

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value


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


class SizeValue:
    arity = 0
    _sizes = {}

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.SIZE
        self.code = f"SIZE.{enum_value.name}"
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls

    @property
    def size(self):
        return self._sizes.get(self.value, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "Size":
                _, enum_name = parts
                for degree in Degree._all_values:
                    if str(degree.value) == str(enum_name):
                        cls._sizes[degree.value] = size
                        break


class DegreeValue:
    arity = 0
    _sizes = {}

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.DEGREE
        self.code = f"DEGREE.{enum_value.name}"
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls

    @property
    def size(self):
        return self._sizes.get(self.value, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "Degree":
                _, enum_name = parts
                for degree in Degree._all_values:
                    if str(degree.value) == str(enum_name):
                        cls._sizes[degree.value] = size
                        break


class HeightValue:
    arity = 0
    _sizes = {}

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.HEIGHT
        self.code = f"HEIGHT.{enum_value.name}"
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls

    @property
    def size(self):
        return self._sizes.get(self.value, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "Height":
                _, enum_name = parts
                for height in Height._all_values:
                    if str(height.value) == str(enum_name):
                        cls._sizes[height.value] = size
                        break


class WidthValue:
    arity = 0
    _sizes = {}

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.WIDTH
        self.code = f"WIDTH.{enum_value.name}"
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls

    @property
    def size(self):
        return self._sizes.get(self.value, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "Width":
                _, enum_name = parts
                for width in Width._all_values:
                    if str(width.value) == str(enum_name):
                        cls._sizes[width.value] = size
                        break


class Height(FilterASTNode):
    _all_values = set()
    arity = 0
    nodeType = FilterTypes.HEIGHT

    def __new__(cls, enum_value):
        instance = HeightValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    def get_all_values(cls):
        return list(cls._enum_members)


class Width(FilterASTNode):
    _all_values = set()
    arity = 0
    nodeType = FilterTypes.WIDTH

    def __new__(cls, enum_value):
        instance = WidthValue(enum_value)
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


class Column(FilterASTNode):
    _all_values = set()
    arity = 0
    nodeType = FilterTypes.COLUMN

    def __new__(cls, enum_value):
        instance = ColumnValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    def get_all_values(cls):
        return list(cls._enum_members)


class ColumnValue:
    arity = 0
    _sizes = {}

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.COLUMN
        self.code = f"COLUMN.{enum_value.name}"
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls

    @property
    def size(self):
        return self._sizes.get(self.value, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "Column":
                _, enum_name = parts
                for column in Column._all_values:
                    if str(column.value) == str(enum_name):
                        cls._sizes[column.value] = size
                        break


class Row(FilterASTNode):
    _all_values = set()
    arity = 0
    nodeType = FilterTypes.ROW

    def __new__(cls, enum_value):
        instance = RowValue(enum_value)
        cls._all_values.add(instance)
        return instance

    @classmethod
    def get_all_values(cls):
        return list(cls._enum_members)


class RowValue:
    arity = 0
    _sizes = {}

    def __init__(self, enum_value):
        self.value = enum_value.value
        self.nodeType = FilterTypes.ROW
        self.code = f"ROW.{enum_value.name}"
        self.children = []
        self.values = []

    def execute(cls, task, children=None):
        return cls

    @property
    def size(self):
        return self._sizes.get(self.value, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "Row":
                _, enum_name = parts
                for row in Row._all_values:
                    if str(row.value) == str(enum_name):
                        cls._sizes[row.value] = size
                        break


def setup_size_and_degree_based_on_task(task):
    task_sizes = [w for w in task.object_sizes[task.abstraction]]
    _size_additional = {f"{item}": int(item) for item in task_sizes}
    SizeEnum = Enum(
        "SizeEnum", {"MIN": "MIN", "MAX": "MAX",
                     "ODD": "ODD", **_size_additional}
    )

    task_degrees = [d for d in task.object_degrees[task.abstraction]]
    _degree_additional = {f"{item}": int(item) for item in task_degrees}
    DegreeEnum = Enum(
        "DegreeEnum", {"MIN": "MIN", "MAX": "MAX",
                       "ODD": "ODD", **_degree_additional}
    )
    _degrees, _sizes, _heights, _widths, _columns, _rows = [], [], [], [], [], []

    task_heights = [d for d in task.object_heights[task.abstraction]]
    _height_additional = {f"{item}": int(item) for item in task_heights}
    HeightEnum = Enum(
        "HeightEnum", {"MIN": "MIN", "MAX": "MAX",
                       "ODD": "ODD", **_height_additional}
    )

    task_widths = [d for d in task.object_widths[task.abstraction]]
    _width_additional = {f"{item}": int(item) for item in task_widths}
    WidthEnum = Enum(
        "WidthEnum", {"MIN": "MIN", "MAX": "MAX",
                      "ODD": "ODD", **_width_additional}
    )

    task_columns = [d for d in task.columns[task.abstraction]]
    _column_additional = {f"{item}": int(item) for item in task_columns}
    ColumnEnum = Enum(
        "ColumnEnum", {"CENTER": "CENTER", "EVEN": "EVEN", "ODD": "ODD", "EVEN_FROM_RIGHT": "EVEN_FROM_RIGHT",
                       "MOD3": "MOD3", **_column_additional}
    )

    task_rows = [d for d in task.rows[task.abstraction]]
    _row_additional = {f"{item}": int(item) for item in task_rows}
    RowEnum = Enum(
        "RowEnum", {"MIN": "MIN", "MAX": "MAX",
                    "ODD": "ODD", "CENTER": "CENTER", **_row_additional}
    )

    for name, member in SizeEnum.__members__.items():
        setattr(Size, name, SizeValue(member))
        _sizes.append(Size(member))
    for name, member in DegreeEnum.__members__.items():
        setattr(Degree, name, Degree(member))
        _degrees.append(Degree(member))
    for name, member in HeightEnum.__members__.items():
        setattr(Height, name, Height(member))
        _heights.append(Height(member))
    for name, member in WidthEnum.__members__.items():
        setattr(Width, name, Width(member))
        _widths.append(Width(member))
    for name, member in ColumnEnum.__members__.items():
        setattr(Column, name, Column(member))
        _columns.append(Column(member))
    for name, member in RowEnum.__members__.items():
        setattr(Row, name, Row(member))
        _rows.append(Row(member))

    Size._enum_members = _sizes
    Degree._enum_members = _degrees
    Width._enum_members = _widths
    Height._enum_members = _heights
    Column._enum_members = _columns
    Row._enum_members = _rows


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
    most = "most"
    least = "least"

    def __init__(self, value=None):
        super().__init__(FilterTypes.COLOR)
        self.nodeType = FilterTypes.COLOR
        self.code = f"{self.__class__.__name__}.{self.name}"
        # self.size = 1
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

    @property
    def size(self):
        return self._sizes.get(self.name, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "FColor":
                _, enum_name = parts
                for color in cls:
                    if color.value == enum_name:
                        cls._sizes[color.name] = size
                        break


class Shape(FilterASTNode, Enum):
    square = "square"
    enclosed = "enclosed"

    def __init__(self, value=None):
        super().__init__(FilterTypes.SHAPE)
        self.nodeType = FilterTypes.SHAPE
        self.code = f"{self.__class__.__name__}.{self.name}"
        self.children = []
        self.values = []

    @classmethod
    @property
    def arity(cls):
        return 0

    @classmethod
    @property
    def nodeType(cls):
        return FilterTypes.SHAPE

    def execute(cls, task, children):
        return cls

    @classmethod
    def get_all_values(cls):
        return list(cls.__members__.values())

    @property
    def size(self):
        return self._sizes.get(self.name, 1)

    @classmethod
    def set_sizes(cls, new_sizes):
        for key, size in new_sizes.items():
            parts = key.split('.')
            if len(parts) == 2 and parts[0] == "Shape":
                _, enum_name = parts
                for shape in cls:
                    if shape.value == enum_name:
                        cls._sizes[shape.name] = size
                        break


class Filters(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]

    def __init__(self, filters: Union["And", "Or", "Filters"] = None):
        super().__init__(FilterTypes.FILTERS)
        self.children = [filters] if filters else []
        self.code = filters.code if filters else ""
        self.size = filters.size if filters else 0
        self.childTypes = [FilterTypes.FILTERS, FilterTypes.FILTERS]


class IsAnyNeighbor(FilterASTNode):
    size = 1

    def __init__(self, value=None):
        super().__init__(FilterTypes.RELATION)
        self.nodeType = FilterTypes.RELATION
        self.code = f"IsAnyNeighbor"
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

    @classmethod
    def execute(cls, task, children=None):
        if task.abstraction == "na":
            cls.values = []
        else:
            task = task.reset_task()
            cls.values = [
                {node: list(set([
                    neighbor for neighbor in input_graph.graph.nodes() if
                    input_graph.get_relative_pos(node, neighbor) is not None
                    and node != neighbor]
                ))
                    for node in input_graph.graph.nodes()}
                for input_graph in task.input_abstracted_graphs_original[task.abstraction]]

        if all(all(not value for value in node_dict.values()) for node_dict in cls.values):
            cls.values = []
        cls.code = f"IsAnyNeighbor"
        return cls


class Neighbor_Of(Filters):
    arity = 0
    size = 1
    default_size = 1

    def __init__(self):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.size = self.default_size
        self.children = []
        self.values = []
        self.childTypes = []

    @classmethod
    def execute(cls, task, children=None):
        instance = cls()
        if task.abstraction == "na":
            instance.values = []
        else:
            task = task.reset_task()
            instance.values = [
                {node: [neighbor for neighbor in input_graph.graph.neighbors(node)]
                for node in input_graph.graph.nodes()}
                for input_graph in task.input_abstracted_graphs_original[task.abstraction]]

        if all(all(not value for value in node_dict.values()) for node_dict in instance.values):
            instance.values = []
        instance.code = f"Neighbor_Of(Obj) == X"
        return instance


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
        new_instance = cls(children[0], children[1])

        intersected_values = [
            list(set(v1).intersection(set(v2))) if set(
                v1).intersection(set(v2)) else []
            for v1, v2 in zip(values1, values2)
        ]
        res_dict = []
        for i, _ in enumerate(intersected_values):
            filtered_nodes_dict = {node: [] for node in intersected_values[i]}
            res_dict.append(filtered_nodes_dict)
        new_instance.values = res_dict

        if children[0].__class__.__name__ == "Neighbor_Of" and children[1].__class__.__name__ == "Neighbor_Of":
            res_dict = {}  # undefined semantics
            new_instance.values = res_dict
        elif children[0].__class__.__name__ == "Neighbor_Of":
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                intersection_dict = {}
                for key1, values1 in dict1.items():
                    intersection_values = [
                        value for value in values1 if value in dict2.keys()]
                    intersection_dict[key1] = intersection_values
                res_dict.append(intersection_dict)
            new_code = children[1].code.replace("Obj", "X")
            new_instance.code = f"And({children[0].code}, {new_code})"
            new_instance.values = res_dict
        elif children[1].__class__.__name__ == "Neighbor_Of":
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                intersection_dict = {}
                for key2, values2 in dict2.items():
                    intersection_values = [
                        value for value in values2 if value in dict1.keys()]
                    intersection_dict[key2] = intersection_values
                res_dict.append(intersection_dict)
            new_code = children[0].code.replace("Obj", "X")
            new_instance.code = f"And({new_code}, {children[1].code})"
            new_instance.values = res_dict
        elif task.current_spec:
            res_dict = [{key: list(set(dict_a[key]).intersection(set(dict_b[key])))
                        for key in dict_a if key in dict_b}
                        for dict_a, dict_b in zip(values1, values2)]
            res_dict = []
            res_dict_test = []
            for dict1, dict2 in zip(values1, values2):
                # Find common objects between dict1 and dict2
                common_keys = set(dict1.keys()) & set(dict2.keys())
                common_dict = {key: dict1[key] + dict2[key]
                            for key in common_keys} # intersection of objects while preserving the relationships
                res_dict.append(common_dict)
            new_instance.values = res_dict
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
        unioned_values = [
            list(set(v1).union(set(v2))) for v1, v2 in zip(values1, values2)
        ]
        res_dict = []
        for i, _ in enumerate(unioned_values):
            filtered_nodes_dict = {node: [] for node in unioned_values[i]}
            res_dict.append(filtered_nodes_dict)

        new_instance = cls(children[0], children[1])
        new_instance.values = res_dict

        if children[0].__class__.__name__ == "Neighbor_Of" and children[1].__class__.__name__ == "Neighbor_Of":
            res_dict = {}  # undefined semantics
            new_instance.values = res_dict
        elif children[0].__class__.__name__ == "Neighbor_Of":
            res_dict = []
            for dict1, dict2 in zip(values1, values2): # neighbors of object and have other properties
                union_dict = {}
                for key1, values1 in dict1.items():
                    union_values = set(values1)
                    if key1 in dict2:
                        union_values.update(dict2[key1])
                    union_dict[key1] = list(union_values)
                res_dict.append(union_dict)
            new_code = children[1].code.replace("Obj", "X")
            new_instance.code = f"Or({children[0].code}, {new_code})"
            new_instance.values = res_dict
        elif children[1].__class__.__name__ == "Neighbor_Of":
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                union_dict = {}
                for key2, values2 in dict2.items():
                    union_values = set(values2)
                    if key2 in dict1:
                        union_values.update(dict1[key2])
                    union_dict[key2] = list(union_values)
                res_dict.append(union_dict)
            new_code = children[0].code.replace("Obj", "X")
            new_instance.code = f"Or({new_code}, {children[1].code})"
            new_instance.values = res_dict
        elif task.current_spec:
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                # union of keys between dict1 and dict2
                union_keys = set(dict1.keys()) | set(dict2.keys())
                union_dict = {}
                for key in union_keys:
                    values1 = dict1.get(key, [])
                    values2 = dict2.get(key, [])
                    combined_values = list(set(values1) | set(values2)) # always preserve relationships
                    union_dict[key] = combined_values
                res_dict.append(union_dict)

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
        nodes_with_data, values_dict, res_dict = [], [], []
        for input_abstracted_graphs in task.input_abstracted_graphs_original[task.abstraction]:
            local_data = []
            for node, _ in input_abstracted_graphs.graph.nodes(data=True):
                local_data.append(node)
            nodes_with_data.append(local_data)
        
        result = [
            [item for item in sublist1 if item not in sublist2]
            for sublist1, sublist2 in zip(nodes_with_data, values)
        ]
        for i, _ in enumerate(result):
            filtered_nodes_dict = {node: [] for node in result[i]}
            res_dict.append(filtered_nodes_dict)
        new_instance = cls(children[0])
        new_instance.values = res_dict
        if children[0].__class__.__name__ == "Neighbor_Of":
            adjusted_values = []
            for graph_index, node_dict in enumerate(values):
                adjusted_node_dict = {}
                for node, neighbors in node_dict.items():
                    not_neighbors = [neighbor for neighbor in nodes_with_data[graph_index] if neighbor not in neighbors]
                    adjusted_node_dict[node] = not_neighbors # not neighbors, includes self-reference
                adjusted_values.append(adjusted_node_dict)
            new_instance.code =  f"Not(Neighbor_Of(Obj) == X)"
            new_instance.values = adjusted_values
        return new_instance


class Color_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.COLOR]
    default_size = 1

    def __init__(self, color: FColor):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Color_Of(Obj) == {color.code}"
        self.size = self.default_size + color.size
        self.children = [color]
        self.childTypes = [FilterTypes.COLOR]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Size_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.SIZE]
    default_size = 1

    def __init__(self, size: Size):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Size_Of(Obj) == {size.code}"
        self.size = self.default_size + size.size
        self.children = [size]
        self.childTypes = [FilterTypes.SIZE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Height_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.HEIGHT]
    default_size = 1

    def __init__(self, height: Height):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Height_Of(Obj) == {height.code}"
        self.size = self.default_size + height.size
        self.children = [height]
        self.childTypes = [FilterTypes.HEIGHT]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Width_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.WIDTH]
    default_size = 1

    def __init__(self, width: Width):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Width_Of(Obj) == {width.code}"
        self.size = self.default_size + width.size
        self.children = [width]
        self.childTypes = [FilterTypes.WIDTH]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Degree_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.DEGREE]
    default_size = 1

    def __init__(self, degree: Degree):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Degree_Of(Obj) == {degree.code}"
        self.size = self.default_size + degree.size
        self.children = [degree]
        self.childTypes = [FilterTypes.DEGREE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Shape_Of(Filters):
    arity = 1
    default_size = 1
    size = 1
    childTypes = [FilterTypes.SHAPE]

    def __init__(self, shape: Shape):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Shape_Of(Obj) == {shape.code}"
        self.size = self.default_size + shape.size
        self.children = [shape]
        self.childTypes = [FilterTypes.SHAPE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Row_Of(Filters):
    arity = 1
    default_size = 1
    nodeType = FilterTypes.FILTERS
    size = 1
    childTypes = [FilterTypes.ROW]

    def __init__(self, row: Row):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Row_Of({row.code})"
        self.size = self.default_size + row.size
        self.children = [row]
        self.childTypes = [FilterTypes.ROW]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Column_Of(Filters):
    arity = 1
    default_size = 1
    nodeType = FilterTypes.FILTERS
    size = 1
    childTypes = [FilterTypes.COLUMN]

    def __init__(self, col: Column):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Column_Of({col.code})"
        self.size = self.default_size + col.size
        self.children = [col]
        self.childTypes = [FilterTypes.COLUMN]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Neighbor_Size_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.SIZE]
    default_size = 1

    def __init__(self, size: Size):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Neighbor_Size_Of(Obj) == {size.code}"
        self.size = self.default_size + size.size
        self.children = [size]
        self.childTypes = [FilterTypes.SIZE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Neighbor_Color_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.COLOR]
    default_size = 1

    def __init__(self, color: FColor):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Neighbor_Color_Of(Obj) == {color.code}"
        self.size = self.default_size + color.size
        self.children = [color]
        self.childTypes = [FilterTypes.COLOR]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance


class Neighbor_Degree_Of(Filters):
    arity = 1
    childTypes = [FilterTypes.DEGREE]
    default_size = 1

    def __init__(self, degree: Degree):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.code = f"Neighbor_Degree_Of(Obj) == {degree.code}"
        self.size = self.default_size + degree.size
        self.children = [degree]
        self.childTypes = [FilterTypes.DEGREE]

    @classmethod
    def execute(cls, task, children):
        instance = cls(children[0])
        values = task.filter_values(instance)
        instance.values = values
        return instance