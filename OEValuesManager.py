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

    def _serialize(self, values):
        return ';'.join(','.join(str(item) for item in sublist) for sublist in values)

    def is_representative(self, program: Union[FilterASTNode, TransformASTNode]) -> bool:
        #results = tuple(tuple(inner_list) for inner_list in program.values)
        #results = json.dumps(program.values)
        results = '|'.join(['-'.join(map(str, inner_list)) for inner_list in program.values])
        print("class values:", self.class_values)
        print("Program:", program.code)
        print("Values:", program.values)
        print("Results:", results)

        if results == "":
            return False
        if results in self.class_values:
            return False

        self.class_values.add(results)
        return True
    
    def is_frepresentative(self, program: Union[FilterASTNode, TransformASTNode]) -> bool:
        results = tuple(tuple(inner_list) for inner_list in program.values)

        if all(len(inner) == 0 for inner in results):
            return False
        if results in self.class_values:
            return False
        self.class_values.add(results)
        return True

    def clear(self) -> None:
        self.class_values.clear()