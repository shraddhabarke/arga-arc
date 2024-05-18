from enum import Enum
from typing import Union, List, Dict
from transform import Dir
from pprint import pprint


class FilterTypes(Enum):
    FILTERS = "Filters"
    OBJECT = "Object"
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
            parts = key.split(".")
            if len(parts) == 2 and parts[0] == "Size":
                _, enum_name = parts
                for degree in Size._all_values:
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
            parts = key.split(".")
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
            parts = key.split(".")
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
            parts = key.split(".")
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
            parts = key.split(".")
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
            parts = key.split(".")
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
        "SizeEnum",
        {
            "MIN": "MIN",
            "MAX": "MAX",
            "ODD": "ODD",
            "EVEN": "EVEN",
            **_size_additional,
            "SizeOf": "SizeOf",
        },
    )

    task_degrees = [d for d in task.object_degrees[task.abstraction]]
    _degree_additional = {f"{item}": int(item) for item in task_degrees}
    DegreeEnum = Enum(
        "DegreeEnum",
        {
            "MIN": "MIN",
            "MAX": "MAX",
            "ODD": "ODD",
            **_degree_additional,
            "DegreeOf": "DegreeOf",
        },
    )
    _degrees, _sizes, _heights, _widths, _columns, _rows = [], [], [], [], [], []

    task_heights = [d for d in task.object_heights[task.abstraction]]
    _height_additional = {f"{item}": int(item) for item in task_heights}
    HeightEnum = Enum(
        "HeightEnum",
        {
            "MIN": "MIN",
            "MAX": "MAX",
            "ODD": "ODD",
            **_height_additional,
            "HeightOf": "HeightOf",
        },
    )

    task_widths = [d for d in task.object_widths[task.abstraction]]
    _width_additional = {f"{item}": int(item) for item in task_widths}
    WidthEnum = Enum(
        "WidthEnum",
        {
            "MIN": "MIN",
            "MAX": "MAX",
            "ODD": "ODD",
            **_width_additional,
            "WidthOf": "WidthOf",
        },
    )

    task_columns = [d for d in task.columns[task.abstraction]]
    _column_additional = {f"{item}": int(item) for item in task_columns}
    ColumnEnum = Enum(
        "ColumnEnum",
        {
            "CENTER": "CENTER",
            "EVEN": "EVEN",
            "ODD": "ODD",
            "EVEN_FROM_RIGHT": "EVEN_FROM_RIGHT",
            "MOD3": "MOD3",
            **_column_additional,
            "ColumnOf": "ColumnOf",
        },
    )

    task_rows = [d for d in task.rows[task.abstraction]]
    _row_additional = {f"{item}": int(item) for item in task_rows}
    RowEnum = Enum(
        "RowEnum",
        {
            "MIN": "MIN",
            "MAX": "MAX",
            "ODD": "ODD",
            "CENTER": "CENTER",
            **_row_additional,
            "RowOf": "RowOf",
        },
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
    colorof = "ColorOf"

    def __init__(self, value=None):
        super().__init__(FilterTypes.COLOR)
        self.nodeType = FilterTypes.COLOR
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
            parts = key.split(".")
            if len(parts) == 2 and parts[0] == "FColor":
                _, enum_name = parts
                for color in cls:
                    if color.value == enum_name:
                        cls._sizes[color.name] = size
                        break


class Object(FilterASTNode, Enum):
    this = "this"
    var = "var"

    def __init__(self, value=None):
        super().__init__(FilterTypes.OBJECT)
        self.nodeType = FilterTypes.OBJECT
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
        return FilterTypes.OBJECT

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
            parts = key.split(".")
            if len(parts) == 2 and parts[0] == "Object":
                _, enum_name = parts
                for obj in cls:
                    if obj.value == enum_name:
                        cls._sizes[obj.name] = size
                        break


class Shape(FilterASTNode, Enum):
    square = "square"
    enclosed = "enclosed"
    shapeof = "ShapeOf"

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
            parts = key.split(".")
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


class Direct_Neighbor_Of(Filters):
    arity = 0
    default_size = 1
    size = default_size + 2
    def __init__(self):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.size = self.default_size + 2
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
                {
                    node: [neighbor for neighbor in input_graph.graph.neighbors(node)]
                    for node in input_graph.graph.nodes()
                }
                for input_graph in task.input_abstracted_graphs_original[
                    task.abstraction
                ]
            ]
        if all(
            all(not value for value in node_dict.values())
            for node_dict in instance.values
        ):
            instance.values = []
        instance.code = f"Direct_Neighbor_Of(Obj) == X"
        return instance


class Neighbor_Of(Filters):
    arity = 0
    default_size = 1
    size = default_size + 2

    def __init__(self):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.size = self.default_size + 2
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
                {
                    node: list(
                        set(
                            [
                                neighbor
                                for neighbor in input_graph.graph.nodes()
                                if input_graph.get_relative_pos(node, neighbor)
                                is not None
                                and node != neighbor
                            ]
                        )
                    )
                    for node in input_graph.graph.nodes()
                }
                for input_graph in task.input_abstracted_graphs_original[
                    task.abstraction
                ]
            ]

        if all(
            all(not value for value in node_dict.values())
            for node_dict in instance.values
        ):
            instance.values = []
        instance.code = f"Neighbor_Of(Obj) == X"
        return instance


class And(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [[FilterTypes.FILTERS, FilterTypes.FILTERS]]
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
            list(set(v1).intersection(set(v2))) if set(v1).intersection(set(v2)) else []
            for v1, v2 in zip(values1, values2)
        ]
        res_dict = []
        for i, _ in enumerate(intersected_values):
            filtered_nodes_dict = {node: [] for node in intersected_values[i]}
            res_dict.append(filtered_nodes_dict)
        new_instance.values = res_dict

        if (
            "Neighbor_Of" in children[0].code
            and "Neighbor_Of" in children[1].code
            or "Neighbor_Of" in children[0].code
            and "Direct_Neighbor_Of" in children[1].code
            or "Direct_Neighbor_Of" in children[0].code
            and "Neighbor_Of" in children[1].code
            or "Direct_Neighbor_Of" in children[0].code
            and "Direct_Neighbor_Of" in children[1].code
            or all(not d for d in children[0].values)
            or all(not d for d in children[1].values)
        ):
            res_dict = {}  # undefined semantics
            new_instance.values = res_dict

        elif any(
            keyword in child.code
            for child in children[:2]
            for keyword in ["Object.this", "Equals"]
        ):
            # not all([set(dict1.keys()) == set(dict2.keys()) for dict1, dict2 in zip(values1, values2)]):
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                # Find common objects between dict1 and dict2
                common_keys = set(dict1.keys()) & set(dict2.keys())
                common_dict = {
                    key: dict1[key] + dict2[key] for key in common_keys
                }  # intersection of objects while preserving the relationships
                res_dict.append(common_dict)
            new_instance.values = res_dict

        elif (
            ("Object.var" in children[0].code and "Object.var" in children[1].code)
            or children[1].__class__.__name__ == "Neighbor_Of"
            or children[1].__class__.__name__ == "Direct_Neighbor_Of"
            or children[0].__class__.__name__ == "Neighbor_Of"
            or children[0].__class__.__name__ == "Direct_Neighbor_Of"
        ):
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                common_keys = set(dict1.keys()) & set(dict2.keys())
                common_dict = {}
                for key in common_keys:
                    if list(
                        set(dict1[key]) & set(dict2[key])
                    ):  # intersection of relationships
                        common_dict[key] = list(set(dict1[key]) & set(dict2[key]))
                res_dict.append(common_dict)
            new_instance.values = res_dict
        return new_instance


class Or(FilterASTNode):
    arity = 2
    nodeType = FilterTypes.FILTERS
    childTypes = [[FilterTypes.FILTERS, FilterTypes.FILTERS]]
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

        if (
            "Neighbor_Of" in children[0].code
            and "Neighbor_Of" in children[1].code
            or "Neighbor_Of" in children[0].code
            and "Direct_Neighbor_Of" in children[1].code
            or "Direct_Neighbor_Of" in children[0].code
            and "Neighbor_Of" in children[1].code
            or "Direct_Neighbor_Of" in children[0].code
            and "Direct_Neighbor_Of" in children[1].code
            or all(not d for d in children[0].values)
            or all(not d for d in children[1].values)
        ):
            res_dict = {}  # undefined semantics
            new_instance.values = res_dict
        elif any(
            keyword in child.code
            for child in children[:2]
            for keyword in ["Object.this", "Equals"]
        ):
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                # union of keys between dict1 and dict2
                union_keys = set(dict1.keys()) | set(dict2.keys())
                union_dict = {}
                for key in union_keys:  # union of keys
                    values1 = dict1.get(key, [])
                    values2 = dict2.get(key, [])
                    combined_values = list(
                        set(values1) | set(values2)
                    )  # always preserve relationships
                    union_dict[key] = combined_values
                res_dict.append(union_dict)
        elif (
            ("Object.var" in children[0].code and "Object.var" in children[1].code)
            or children[1].__class__.__name__ == "Neighbor_Of"
            or children[1].__class__.__name__ == "Direct_Neighbor_Of"
            or children[0].__class__.__name__ == "Neighbor_Of"
            or children[0].__class__.__name__ == "Direct_Neighbor_Of"
        ):
            res_dict = []
            for dict1, dict2 in zip(values1, values2):
                common_keys, common_dict = set(dict1.keys()) & set(dict2.keys()), {}
                for key in common_keys:
                    if list(
                        set(dict1[key]) | set(dict2[key])
                    ):  # union of relationships
                        common_dict[key] = list(set(dict1[key]) | set(dict2[key]))
                res_dict.append(common_dict)
            new_instance.values = res_dict
        new_instance.values = res_dict
        return new_instance


class Not(FilterASTNode):
    arity = 1
    nodeType = FilterTypes.FILTERS
    childTypes = [[FilterTypes.FILTERS]]
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
        new_instance = cls(children[0])
        nodes_with_data = []
        for input_abstracted_graphs in task.input_abstracted_graphs_original[
            task.abstraction
        ]:
            local_data = []
            for node, _ in input_abstracted_graphs.graph.nodes(data=True):
                local_data.append(node)
            nodes_with_data.append(local_data)
        result = [
            [item for item in sublist1 if item not in sublist2]
            for sublist1, sublist2 in zip(nodes_with_data, values)
        ]

        if (
            "Neighbor_Of" in children[0].code
            or "Direct_Neighbor_Of" in children[0].code
            or all(not d for d in children[0].values)
        ):
            res_dict = {}  # undefined semantics
            new_instance.values = res_dict
        elif children[0].__class__.__name__ == "Neighbor_Of":
            adjusted_values = []
            for graph_index, node_dict in enumerate(values):
                adjusted_node_dict = {}
                for node, neighbors in node_dict.items():
                    not_neighbors = [
                        neighbor
                        for neighbor in nodes_with_data[graph_index]
                        if neighbor not in neighbors
                    ]
                    # not neighbors, includes self-reference
                    adjusted_node_dict[node] = not_neighbors
                adjusted_values.append(adjusted_node_dict)
            new_instance.code = f"Not(Neighbor_Of(Obj) == X)"
            new_instance.values = adjusted_values
        elif any(keyword in children[0].code for keyword in ["Object.this", "Equals"]):
            res_dict = []
            for i, _ in enumerate(result):
                filtered_nodes_dict = {node: [] for node in result[i]}
                res_dict.append(filtered_nodes_dict)
            new_instance = cls(children[0])
            new_instance.values = res_dict
        elif "var" in children[0].code:
            adjusted_values = []
            for graph_index, node_dict in enumerate(values):
                adjusted_node_dict = {}
                for node, neighbors in node_dict.items():
                    not_neighbors = [
                        neighbor
                        for neighbor in nodes_with_data[graph_index]
                        if neighbor not in neighbors
                    ]
                    # not neighbors, includes self-reference
                    adjusted_node_dict[node] = not_neighbors
                adjusted_values.append(adjusted_node_dict)
            new_instance.values = adjusted_values
        return new_instance


class Color_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.COLOR, FilterTypes.COLOR, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, color1: FColor, color2: FColor, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.childTypes = [FilterTypes.COLOR, FilterTypes.COLOR, FilterTypes.OBJECT]
        if color2 == FColor.colorof:
            self.code = f"Color_Of({obj.code}) == {color1.code}"
            self.children = [color1]
            self.size = self.default_size + color1.size + color2.size + 1
        elif color1 == FColor.colorof:
            self.code = f"Color_Of({obj.code}) == {color2.code}"
            self.children = [color2]
            self.size = self.default_size + color1.size + color2.size + 1
        else:
            self.code = f"Equals({color2.code}, {color1.code})"
            self.children = [color1, color2]
            self.size = self.default_size + color1.size + color2.size

    @classmethod
    def execute(cls, task, children):
        if children[0] == FColor.colorof and children[1] == FColor.colorof:
            cls.code = f"(Color_Of({children[2].code}) == Color_Of({children[2].code}))"
            cls.values = []
            return cls
        instance = cls(*children)
        if (not children[0] == FColor.colorof and not children[1] == FColor.colorof) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance


class Size_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.SIZE, FilterTypes.SIZE, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, size1: Size, size2: Size, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.childTypes = [FilterTypes.SIZE, FilterTypes.SIZE, FilterTypes.OBJECT]
        self.size = self.default_size + size1.size + size2.size + 1
        if size2.code == "SIZE.SizeOf":
            self.code = f"Size_Of({obj.code}) == {size1.code}"
            self.children = [size1]
        elif size1.code == "SIZE.SizeOf":
            self.code = f"Size_Of({obj.code}) == {size2.code}"
            self.children = [size2]
        else:
            self.code = f"Equals({size2.code}, {size1.code})"
            self.children = [size1, size2]

    @classmethod
    def execute(cls, task, children):
        if children[0].code == "SIZE.SizeOf" and children[1].code == "SIZE.SizeOf":
            cls.code = f"(Size_Of({children[2].code}) == Size_Of({children[2].code}))"
            cls.values = []
            return cls
        instance = cls(*children)
        if (
            not children[0].code == "SIZE.SizeOf"
            and not children[1].code == "SIZE.SizeOf"
        ) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance


class Height_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.HEIGHT, FilterTypes.HEIGHT, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, height1: Height, height2: Height, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.size = self.default_size + height1.size + height2.size + 1
        self.childTypes = [FilterTypes.HEIGHT, FilterTypes.HEIGHT, FilterTypes.OBJECT]
        if height2.code == "HEIGHT.HeightOf":
            self.code = f"Height_Of({obj.code}) == {height1.code}"
            self.children = [height1]
        elif height1.code == "HEIGHT.HeightOf":
            self.code = f"Height_Of({obj.code}) == {height2.code}"
            self.children = [height2]
        else:
            self.code = f"Equals({height2.code}, {height1.code})"
            self.children = [height1, height2]

    @classmethod
    def execute(cls, task, children):
        if (
            children[0].code == "HEIGHT.HeightOf"
            and children[1].code == "HEIGHT.HeightOf"
        ):
            cls.code = (
                f"(Height_Of({children[2].code}) == Height_Of({children[2].code}))"
            )
            cls.values = []
            return cls
        instance = cls(*children)
        if (
            not children[0].code == "HEIGHT.HeightOf"
            and not children[1].code == "HEIGHT.HeightOf"
        ) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance


class Width_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.WIDTH, FilterTypes.WIDTH, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, width1: Width, width2: Width, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.childTypes = [FilterTypes.WIDTH, FilterTypes.WIDTH, FilterTypes.OBJECT]
        self.size = self.default_size + width1.size + width2.size + 1
        if width2.code == "WIDTH.WidthOf":
            self.code = f"Width_Of({obj.code}) == {width1.code}"
            self.children = [width1]
        elif width1.code == "WIDTH.WidthOf":
            self.code = f"Width_Of({obj.code}) == {width2.code}"
            self.children = [width2]
        else:
            self.code = f"Equals({width2.code}, {width1.code})"
            self.children = [width1, width2]

    @classmethod
    def execute(cls, task, children):
        if children[0].code == "WIDTH.WidthOf" and children[1].code == "WIDTH.WidthOf":
            cls.code = f"(Width_Of({children[2].code}) == Width_Of({children[2].code}))"
            cls.values = []
            return cls
        instance = cls(*children)
        if (
            not children[0].code == "WIDTH.WidthOf"
            and not children[1].code == "WIDTH.WidthOf"
        ) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance


class Degree_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.DEGREE, FilterTypes.DEGREE, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, degree1: Degree, degree2: Degree, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.childTypes = [FilterTypes.DEGREE, FilterTypes.DEGREE, FilterTypes.OBJECT]
        self.size = self.default_size + degree1.size + degree2.size + 1
        if degree2.code == "Degree.DegreeOf":
            self.code = f"Degree_Of({obj.code}) == {degree1.code}"
            self.children = [degree1]
        elif degree1.code == "Degree.DegreeOf":
            self.code = f"Degree_Of({obj.code}) == {degree2.code}"
            self.children = [degree2]
        else:
            self.code = f"Equals({degree2.code}, {degree1.code})"
            self.children = [degree1, degree2]

    @classmethod
    def execute(cls, task, children):
        if (
            children[0].code == "DEGREE.DegreeOf"
            and children[1].code == "DEGREE.DegreeOf"
        ):
            cls.code = (
                f"(Degree_Of({children[2].code}) == Degree_Of({children[2].code}))"
            )
            cls.values = []
            return cls
        instance = cls(*children)
        if (
            not children[0].code == "DEGREE.DegreeOf"
            and not children[1].code == "DEGREE.DegreeOf"
        ) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance


class Shape_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.SHAPE, FilterTypes.SHAPE, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, shape1: Shape, shape2: Shape, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.childTypes = [FilterTypes.SHAPE, FilterTypes.SHAPE, FilterTypes.OBJECT]
        self.size = self.default_size + shape1.size + shape2.size + 1
        if shape2 == Shape.shapeof:
            self.code = f"Shape_Of({obj.code}) == {shape1.code}"
            self.children = [shape1]
        elif shape1 == Shape.shapeof:
            self.code = f"Shape_Of({obj.code}) == {shape2.code}"
            self.children = [shape2]
        else:
            self.code = f"Equals({shape2.code}, {shape1.code})"
            self.children = [shape1, shape2]

    @classmethod
    def execute(cls, task, children):
        if children[0] == Shape.shapeof and children[1] == Shape.shapeof:
            cls.code = f"(Shape_Of({children[2].code}) == Shape_Of({children[2].code}))"
            cls.values = []
            return cls
        instance = cls(*children)
        if (
            not children[0] == Shape.shapeof and not children[1] == Shape.shapeof
        ) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance


class Row_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.ROW, FilterTypes.ROW, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, row1: Row, row2: Row, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.childTypes = [FilterTypes.ROW, FilterTypes.ROW, FilterTypes.OBJECT]
        self.size = self.default_size + row1.size + row2.size + 1
        if row2.code == "ROW.RowOf":
            self.code = f"Row_Of({obj.code}) == {row1.code}"
            self.children = [row1]
        elif row1.code == "ROW.RowOf":
            self.code = f"Row_Of({obj.code}) == {row2.code}"
            self.children = [row2]
        else:
            self.code = f"Equals({row2.code}, {row1.code})"
            self.children = [row1, row2]

    @classmethod
    def execute(cls, task, children):
        if children[0].code == "ROW.RowOf" and children[1].code == "ROW.RowOf":
            cls.code = f"(Row_Of({children[2].code}) == Row_Of({children[2].code}))"
            cls.values = []
            return cls
        instance = cls(*children)
        if (
            not children[0].code == "ROW.RowOf" and not children[1].code == "ROW.RowOf"
        ) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance


class Column_Equals(Filters):
    arity = 2
    childTypes = [[FilterTypes.COLUMN, FilterTypes.COLUMN, FilterTypes.OBJECT]]
    default_size = 1

    def __init__(self, col1: Row, col2: Row, obj: Object):
        super().__init__()
        self.nodeType = FilterTypes.FILTERS
        self.childTypes = [FilterTypes.COLUMN, FilterTypes.COLUMN, FilterTypes.OBJECT]
        self.size = self.default_size + col1.size + col2.size + 1
        if col2.code == "COLUMN.ColumnOf":
            self.code = f"Column_Of({obj.code}) == {col1.code}"
            self.children = [col1]
        elif col1.code == "COLUMN.ColumnOf":
            self.code = f"Column_Of({obj.code}) == {col2.code}"
            self.children = [col2]
        else:
            self.code = f"Equals({col2.code}, {col1.code})"
            self.children = [col1, col2]

    @classmethod
    def execute(cls, task, children):
        if (
            children[0].code == "COLUMN.ColumnOf"
            and children[1].code == "COLUMN.ColumnOf"
        ):
            cls.code = (
                f"(Column_Of({children[2].code}) == Column_Of({children[2].code}))"
            )
            cls.values = []
            return cls
        instance = cls(*children)
        if (
            not children[0].code == "COLUMN.ColumnOf"
            and not children[1].code == "COLUMN.ColumnOf"
        ) or children[2] == Object.this:
            values = task.filter_values(instance)
        elif children[2] == Object.var:
            values = task.var_filter_values(instance)
        instance.values = values
        return instance
