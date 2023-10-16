from enum import Enum
from typing import Union, List, Dict
import typing
from filters import *
from transform import *

class OEValuesManager:
    def is_representative(self, program: Union[FilterASTNode, TransformASTNode]) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

class ValuesManager(OEValuesManager):
    def __init__(self):
        self.class_values = set()

    def is_representative(self, program: Union[FilterASTNode, TransformASTNode]) -> bool:
        results = tuple(tuple(inner) for inner in program.values)
        if all(len(inner) == 0 for inner in results):
            return False
        if results in self.class_values:
            return False
            
        self.class_values.add(results)
        return True  # returns True if the results are added to class_values, the new filter is the representative of it's class

    def clear(self) -> None:
        self.class_values.clear()