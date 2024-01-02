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
            if isinstance(element, dict):
                return '-'.join([f"({k[0]}, {k[1]})-{v}" for k, v in sorted(element.items())])
            elif isinstance(element, tuple):
                return ' | '.join(process_element(d) for d in element)
        results = ' | '.join(process_element(e) for e in values)
        if results == "":
            return False
        if results in self.class_values:
            return False

        self.class_values.add(results)
        return True

    def is_frepresentative(self, program: Union[FilterASTNode, TransformASTNode]) -> bool:
        results = tuple(tuple(inner_list) for inner_list in program.values)
        if results in self.class_values:
            return False
        self.class_values.add(results)
        return True

    def clear(self) -> None:
        self.class_values.clear()
