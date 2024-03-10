from enum import Enum
from typing import Union, List, Dict
import typing
from filters import *
from transform import *
import json


class OEValuesManager:
    def is_representative(self, program: Union[FilterASTNode, TransformASTNode]) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class ValuesManager(OEValuesManager):
    def __init__(self):
        self.class_values = set()

    def is_representative(self, values) -> bool:
        def process_element(element):
            key, data = element
            return f"{key}-{';'.join([f'{k}:{v}' for k, v in sorted(data.items())])}"

        def process_inner_list(inner_list):
            return ' | '.join(process_element(e) for e in inner_list)
        if values is None:
            results == ""
        else:
            results = ' || '.join(process_inner_list(inner.graph.nodes(data=True))
                            for inner in values)  # Processing each inner list
        if results == "":
            return False
        if results in self.class_values:
            return False

        self.class_values.add(results)
        return True

    def is_frepresentative(self, program: Union[FilterASTNode, TransformASTNode]) -> bool:
        def serialize_structure(structure):
            if isinstance(structure, dict):
                return {serialize_tuple(key): [serialize_tuple(value) for value in values] for key, values in structure.items()}
            elif isinstance(structure, list):
                return [serialize_structure(elem) for elem in structure]
            else:
                return structure

        def serialize_tuple(tup):
            return "-".join(map(str, tup))
        results_str = str([serialize_structure(ds) for ds in program.values])
        if results_str in self.class_values:
            return False
        else:
            self.class_values.add(results_str)
            return True

    def clear(self) -> None:
        self.class_values.clear()
