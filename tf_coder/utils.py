import typing as t
from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np

# need this to properly evaluate code
import math


class OutputJSON(t.TypedDict):
    task_id: str
    completions: t.List[str]
    coverage_percentage: float
    description: str
    tf_operators: t.Dict[str, int]
    total_covered: int
    total_in_target: int


class ExamplesJSON(t.TypedDict):
    inputs: str
    outputs: str


class TaskJSON(t.TypedDict):
    constants: str
    description: str
    name: str
    source: str
    target_program: str
    examples: ExamplesJSON


ExamplePrimitiveValue = t.Union[np.ndarray, tf.SparseTensor]
ExampleValue = t.Union[ExamplePrimitiveValue, t.Dict[str, ExamplePrimitiveValue]]


@dataclass
class Example:
    inputs: t.List[ExampleValue]
    output: ExampleValue
    input_names: t.Optional[t.List[str]] = field(default=None)

    @property
    def formals(self) -> str:
        if self.input_names:
            return ", ".join(self.input_names)
        else:
            return ", ".join([f"in{i+1}" for i in range(len(self.inputs))])

    @property
    def max_input_rank(self) -> int:
        shapes = []
        for i in self.__input_primitives:
            shape = self.__get_shape(i)
            if shape:
                shapes.append(shape)
        if len(shapes) == 0:
            return 0
        return max([len(i) for i in shapes])

    @property
    def dimension_lengths(self) -> t.Set[int]:
        shapes = []
        for i in self.__input_primitives:
            shape = self.__get_shape(i)
            if shape:
                shapes.append(shape)
        flat_shapes = [dim for shape in shapes for dim in shape]
        return set(flat_shapes)

    @property
    def output_shape(self) -> t.Optional[t.Tuple[int, ...]]:
        return self.__get_shape(self.output)

    @classmethod
    def from_json(cls, examples: ExamplesJSON):
        input_names = None
        try:
            evaluated_inputs = eval(examples["inputs"])
            if type(evaluated_inputs) == list:
                inputs = [cls.__input_from_json(i) for i in evaluated_inputs]
            elif type(evaluated_inputs) == tuple:
                inputs = [cls.__input_from_json(i) for i in evaluated_inputs]
            elif type(evaluated_inputs) == dict:
                inputs = [cls.__input_from_json(v) for v in evaluated_inputs.values()]
                input_names = list(evaluated_inputs.keys())
            else:
                inputs = [evaluated_inputs]
        except Exception as e:
            print(f"Error evaluating inputs: {e}")
            print(f"Inputs: {examples['inputs']}")
            raise e

        try:
            evaluated_outputs = eval(examples["outputs"])
            if type(evaluated_outputs) == list:
                outputs = np.array(evaluated_outputs)
            # this if overrides the next one, since SparseTensor is a Tensor
            elif isinstance(evaluated_outputs, tf.SparseTensor):
                outputs = evaluated_outputs
            elif isinstance(evaluated_outputs, tf.Tensor):
                outputs = evaluated_outputs.numpy()
            else:
                outputs = evaluated_outputs
        except Exception as e:
            print(f"Error evaluating outputs: {e}")
            print(f"Outputs: {examples['outputs']}")
            raise e

        return cls(inputs, outputs, input_names=input_names)

    @classmethod
    def __input_from_json(cls, input: t.Any) -> ExamplePrimitiveValue:
        if isinstance(input, tf.SparseTensor):
            return input
        elif isinstance(input, tf.Tensor):
            return input.numpy()
        elif type(input) == list:
            return np.array(input)
        else:
            return input

    def toJSON(self):
        return {
            "inputs": [i.tolist() for i in self.inputs],
            "output": self.output.tolist(),
        }

    def __get_shape(self, value: t.Any):
        try:
            return value.shape
        except:
            return None

    @property
    def __input_primitives(self) -> t.List[ExamplePrimitiveValue]:
        ans: t.List[ExamplePrimitiveValue] = []
        for i in self.inputs:
            if isinstance(i, dict):
                for k, v in i.items():
                    ans.append(v)
            else:
                ans.append(i)
