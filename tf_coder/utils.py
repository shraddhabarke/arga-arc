import os
import sys
from pathlib import Path

CURRENT_DIRECTORY = Path(os.getcwd())
ROOT_DIRECTORY = (CURRENT_DIRECTORY / "..").absolute().resolve()

print(f"Current directory: {CURRENT_DIRECTORY}")
print(f"Root directory: {ROOT_DIRECTORY}")

sys.path.append(str(ROOT_DIRECTORY))

import typing as t
import json
from pprint import pprint
from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np
import math
from config import CONFIG
from openai import OpenAI
import re
from collections import Counter
import random
import ast
from ast import Attribute, Name
import traceback
from tqdm import tqdm
import builtins

# need this to properly evaluate code, even though it isn't used explicitly
import math

DATASET_FILE = CONFIG.ROOT_DIR / "tf_coder/tfcoder_dataset.json"
# these were originally randomly sampled, but we're keeping them hardcoded for consistency between runs
IN_CONTEXT_TASK_NAMES = ["google_02", "google_10", "stackoverflow_07"]

OUTPUT_FILE = CONFIG.ROOT_DIR / "tf_coder/tfcoder_output.gpt-3.5-turbo.json"
NUM_COMPLETIONS = 20
MODEL: "ModelName" = "gpt-3.5-turbo"


def main():
    original_print = builtins.print
    builtins.print = tqdm_print

    if OUTPUT_FILE.exists():
        dataset_json: t.List[TaskJSONWithOutput] = json.loads(OUTPUT_FILE.read_text())
        tasks = [Task.from_json_with_output(task_json) for task_json in dataset_json]
    else:
        dataset_json: t.List[TaskJSON] = json.loads(DATASET_FILE.read_text())
        tasks = [Task.from_json(task_json) for task_json in dataset_json]

    in_context_tasks = [task for task in tasks if task.name in IN_CONTEXT_TASK_NAMES]
    in_context_section = "\n\n".join(
        [task.make_in_context_example() for task in in_context_tasks]
    )

    tasks = [task for task in tasks if task.name not in IN_CONTEXT_TASK_NAMES]

    def write():
        write_output(tasks, OUTPUT_FILE)

    print(f"Tasks: {len(tasks)}")
    print(f"generating completions. model: {MODEL}, num_completions: {NUM_COMPLETIONS}")
    for task in tqdm(tasks):
        task.get_completions(
            in_context_section, model=MODEL, n_completions=NUM_COMPLETIONS
        )
        write()

    print(f"computing asts")
    for task in tqdm(tasks):
        task.compute_asts()

    print(f"computing operator coverage")
    for task in tqdm(tasks):
        task.compute_operator_coverage()
        write()

    print(f"computing constants")
    for task in tqdm(tasks):
        task.compute_constants()
        write()


def tqdm_print(*args, **kwargs):
    # if no arguments are passed, write the empty string
    if not args:
        args = [""]
    tqdm.write(*args, **kwargs)


def write_output(tasks: t.List["Task"], output_file: Path):
    output_json = [task.to_json() for task in tasks]
    output_file.write_text(json.dumps(output_json, indent=4))


ModelName = t.Literal[
    "gpt-4",
    "gpt-3.5-turbo",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
]
OPENAI_MODEL_NAMES = ["gpt-4", "gpt-3.5-turbo"]
TOGETHER_MODEL_NAMES = [
    "deepseek-ai/deepseek-coder-33b-instruct",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
]
MODEL_NAMES = OPENAI_MODEL_NAMES + TOGETHER_MODEL_NAMES


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


class TaskJSONWithOutput(TaskJSON):
    completions: t.List[str]
    tf_operators: t.Dict[str, int]
    coverage_percentage: float
    total_covered: int
    total_in_target: int
    parsed_constants: t.List[t.List[int]]
    all_constants: t.List[int]
    constant_counts: t.List["ConstantCounts"]
    aggregate_constant_count: "ConstantCounts"


ExamplePrimitiveValue = t.Union[np.ndarray, tf.SparseTensor]
ExampleValue = t.Union[ExamplePrimitiveValue, t.Dict[str, ExamplePrimitiveValue]]


@dataclass
class Example:
    inputs: t.List[ExampleValue]
    output: ExampleValue
    input_names: t.Optional[t.List[str]] = field(default=None)
    json: t.Optional[ExamplesJSON] = field(default=None)

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

        return cls(inputs, outputs, input_names=input_names, json=examples)

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
        return (
            {
                "inputs": [i.tolist() for i in self.inputs],
                "output": self.output.tolist(),
            }
            if self.json is None
            else self.json
        )

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
        return ans


@dataclass
class Task:
    name: str
    description: str
    target_program: str
    source: str
    constants: str
    examples: Example

    completions: t.List[str] = field(default_factory=list)
    tf_operators: t.Dict[str, int] = field(default_factory=dict)
    coverage_percentage: float = 0.0
    total_covered: int = 0
    total_in_target: int = 0
    parsed_constants: t.List[t.List[int]] = field(default_factory=list)
    all_constants: t.List[int] = field(default_factory=list)
    constant_counts: t.List["ConstantCounts"] = field(default_factory=list)
    aggregate_constant_count: "ConstantCounts" = field(default_factory=dict)

    __asts: t.List[ast.Module] = field(default_factory=list)

    @classmethod
    def from_json(cls, task_json: TaskJSON):
        return cls(
            task_json["name"],
            task_json["description"],
            task_json["target_program"],
            task_json["source"],
            task_json["constants"],
            Example.from_json(task_json["examples"]),
        )

    @classmethod
    def from_json_with_output(cls, task_json: TaskJSONWithOutput):
        ans = cls.from_json(task_json)
        if "completions" in task_json:
            ans.completions = task_json["completions"]
        if "tf_operators" in task_json:
            ans.tf_operators = task_json["tf_operators"]
        if "coverage_percentage" in task_json:
            ans.coverage_percentage = task_json["coverage_percentage"]
        if "total_covered" in task_json:
            ans.total_covered = task_json["total_covered"]
        if "total_in_target" in task_json:
            ans.total_in_target = task_json["total_in_target"]
        if "parsed_constants" in task_json:
            ans.parsed_constants = task_json["parsed_constants"]
        if "all_constants" in task_json:
            ans.all_constants = task_json["all_constants"]
        if "constant_counts" in task_json:
            ans.constant_counts = task_json["constant_counts"]
        if "aggregate_constant_count" in task_json:
            ans.aggregate_constant_count = task_json["aggregate_constant_count"]
        return ans

    def to_json(self) -> TaskJSONWithOutput:
        return {
            "name": self.name,
            "description": self.description,
            "target_program": self.target_program,
            "source": self.source,
            "constants": self.constants,
            "examples": self.examples.toJSON(),
            "completions": self.completions,
            "tf_operators": self.tf_operators,
            "coverage_percentage": self.coverage_percentage,
            "total_covered": self.total_covered,
            "total_in_target": self.total_in_target,
            "parsed_constants": self.parsed_constants,
            "all_constants": self.all_constants,
            "constant_counts": self.constant_counts,
            "aggregate_constant_count": self.aggregate_constant_count,
        }

    def make_user_message(
        self, in_context_examples: str, shuffle_operators=True, order_by_weight=False
    ):
        include_sparse = any(
            isinstance(i, tf.SparseTensor) for i in self.examples.inputs
        ) or isinstance(self.examples.output, tf.SparseTensor)
        return f"""{make_operators_section(shuffle_operators, include_sparse, order_by_weight)}

{in_context_examples}

{self.make_in_context_example(include_program=False)}"""

    def make_in_context_example(self, include_program=True) -> str:
        inputs_str = ""
        for inp in self.examples.inputs:
            inputs_str += f"{inp}\n"
        outputs_str = f"{self.examples.output}\n"

        ans = f"""[TASK DESCRIPTION]
{self.description}

[INPUTS]
{inputs_str}

[OUTPUTS]
{outputs_str}
"""
        ans += f"""[PROGRAM]
def transform({self.examples.formals}):
    """
        if include_program:
            ans += f"return {self.target_program}\n"

        return ans

    def get_completions(
        self, in_context_section: str, model: ModelName = "gpt-4", n_completions=10
    ):
        if len(self.completions) > 0:
            print(f"Task {self.name} already has completions")
            return self.completions

        completions = prompt(
            SYSTEM_PROMPT,
            self.make_user_message(in_context_section),
            n_completions=n_completions,
            model=model,
        )

        self.completions = completions
        return completions

    def compute_operator_coverage(self):
        target_operators = extract_tf_operators(self.target_program)
        completion_operators = []
        for completion in self.completions:
            completion_operators.extend(extract_tf_operators(completion))

        coverage_dict = calculate_tf_operator_coverage_and_count(
            target_operators, completion_operators
        )
        self.tf_operators = coverage_dict["tf_operators"]
        self.coverage_percentage = coverage_dict["coverage_percentage"]
        self.total_covered = coverage_dict["total_covered"]
        self.total_in_target = coverage_dict["total_in_target"]
        return coverage_dict

    def compute_asts(self):
        for completion in self.completions:
            try:
                self.__asts.append(ast.parse(completion))
                continue
            except SyntaxError:
                pass

            normalized = normalize_completion(completion)
            try:
                self.__asts.append(ast.parse(normalized))
            except SyntaxError:
                print(f"Error parsing completion")
                print(f"Original: {completion}")
                print(f"Normalized: {normalized}")
                print(f"Error:")
                traceback.print_exc()
                print()
                self.__asts.append(None)

    def compute_constants(self):
        self.parsed_constants = [
            get_constants(ast_node) if ast_node else [] for ast_node in self.__asts
        ]
        self.all_constants = list(
            set(
                [
                    constant
                    for constants in self.parsed_constants
                    for constant in constants
                ]
            )
        )
        self.constant_counts = [
            count_constants(ast_node, self.examples) if ast_node else None
            for ast_node in self.__asts
        ]
        self.aggregate_constant_count = {
            "common": 0,
            "axis": 0,
            "shape": 0,
            "provided": 0,
            "tf_int32": 0,
            "tf_float32": 0,
            "tf_int64": 0,
            "tf_bool": 0,
            "input_var": 0,
            "shape_tuple": 0,
        }
        for count in self.constant_counts:
            if count:
                self.aggregate_constant_count = add_constant_counts(
                    self.aggregate_constant_count, count
                )


# region PROMPT

SYSTEM_PROMPT = """You are a coding assistant. Be precise and terse.
You will be provided a list of tensorflow operators, a task description, and some input/output examples.
Your task is to generate the body of a python function that will transform the input to the output.
Only use the operators provided in the list.
"""

TFOPERATORS_STR = (
    "tf.abs(x)\ntf.add(x, y)\ntf.add_n(inputs)\ntf.argmax(input, axis)\ntf.argmin(input, axis)\n"
    + "tf.argsort(values, axis, stable=True)\ntf.argsort(values, axis, direction='DESCENDING', stable=True)\ntf.boolean_mask(tensor, mask)\ntf.broadcast_to(input, shape)\n"
    + "tf.cast(x, dtype)\ntf.clip_by_value(t, clip_value_min, clip_value_max)\ntf.concat(values, axis)\ntf.constant(value)\ntf.constant(value, dtype)\ntf.divide(x, y)\n"
    + "tf.equal(x, y)\ntf.exp(x)\ntf.expand_dims(input, axis)\ntf.eye(num_rows)\ntf.eye(num_rows, num_columns)\ntf.eye(num_rows, dtype)\ntf.fill(dims, value)"
    + "tf.gather(params, indices)\ntf.gather(params, indices, axis, batch_dims)\ntf.gather_nd(params, indices)\ntf.gather_nd(params, indices, batch_dims)\ntf.greater(x, y)\n"
    + "tf.greater_equal(x, y)\ntf.math.bincount(arr)\ntf.math.ceil(x)\ntf.math.count_nonzero(input)\ntf.math.count_nonzero(input, axis)\ntf.math.cumsum(x, axis)\n"
    + "tf.math.cumsum(x, axis, exclusive=True)\ntf.math.divide_no_nan(x, y)\ntf.math.floor(x)\ntf.math.log(x)\ntf.math.logical_and(x, y)\ntf.math.logical_not(x)"
    + "tf.math.logical_or(x, y)\ntf.math.logical_xor(x, y)\ntf.math.negative(x)\ntf.math.reciprocal(x)\ntf.math.reciprocal_no_nan(x)\ntf.math.segment_max(data, segment_ids)\n"
    + "tf.math.segment_mean(data, segment_ids)\ntf.math.segment_min(data, segment_ids)\ntf.math.segment_prod(data, segment_ids)\ntf.math.segment_sum(data, segment_ids)\n"
    + "tf.math.squared_difference(x, y)\ntf.math.top_k(input, k)\ntf.math.unsorted_segment_max(data, segment_ids, num_segments)\ntf.math.unsorted_segment_mean(data, segment_ids, num_segments)\n"
    + "tf.math.unsorted_segment_min(data, segment_ids, num_segments)\ntf.math.unsorted_segment_prod(data, segment_ids, num_segments)\ntf.math.unsorted_segment_sum(data, segment_ids, num_segments)\n"
    + "tf.matmul(a, b)\ntf.maximum(x, y)\ntf.minimum(x, y)\ntf.multiply(x, y)\ntf.not_equal(x, y)\ntf.one_hot(indices, depth)\ntf.ones(shape)\ntf.ones_like(input)\n"
    + "tf.pad(tensor, paddings, mode='CONSTANT')\ntf.pad(tensor, paddings, mode='CONSTANT', constant_values)\ntf.pad(tensor, paddings, mode='REFLECT')\n"
    + "tf.pad(tensor, paddings, mode='SYMMETRIC')\ntf.range(start)\ntf.range(start, limit, delta)\ntf.reduce_any(input_tensor, axis)\ntf.reduce_all(input_tensor, axis)\n"
    + "tf.reduce_max(input_tensor)\ntf.reduce_max(input_tensor, axis)\ntf.reduce_mean(input_tensor)\n"
    + "tf.reduce_mean(input_tensor, axis)\ntf.reduce_min(input_tensor)\ntf.reduce_min(input_tensor, axis)\n"
    + "tf.reduce_prod(input_tensor, axis)\ntf.reduce_sum(input_tensor)\ntf.reduce_sum(input_tensor, axis)\n"
    + "tf.repeat(input, repeats)\ntf.repeat(input, repeats, axis)\ntf.reshape(tensor, shape)\n"
    + "tf.reverse(tensor, axis)\ntf.roll(input, shift, axis)\ntf.round(x)\ntf.scatter_nd(indices, updates, shape)\n"
    + "tf.searchsorted(sorted_sequence, values, side='left')\ntf.searchsorted(sorted_sequence, values, side='right')\n"
    + "tf.sequence_mask(lengths)\ntf.sequence_mask(lengths, maxlen)\ntf.shape(input)\ntf.sign(x)\n"
    + "tf.sort(values, axis)\ntf.sort(values, axis, direction='DESCENDING')\ntf.sqrt(x)\n"
    + "tf.square(x)\ntf.squeeze(input)\ntf.squeeze(input, axis)\ntf.stack(values, axis)\ntf.subtract(x, y)\n"
    + "tf.tensor_scatter_nd_update(tensor, indices, updates)\ntf.tensordot(a, b, axes)\ntf.tile(input, multiples)\n"
    + "tf.transpose(a)\ntf.transpose(a, perm)\ntf.unique_with_counts(x)\ntf.unstack(value, axis)\n"
    + "tf.where(condition)\ntf.where(condition, x, y)\ntf.zeros(shape)\ntf.zeros_like(input)"
)
SPARSETF_OPERATORS_STR = (
    "tf.SparseTensor(indices, values, dense_shape)\ntf.sparse.add(a, b)\n"
    + "tf.sparse.concat(axis, sp_inputs)\ntf.sparse.expand_dims(sp_input, axis)\ntf.sparse.from_dense(tensor)\ntf.sparse.maximum(sp_a, sp_b)\n"
    + "tf.sparse.minimum(sp_a, sp_b)\ntf.sparse.reduce_max(sp_input, axis, output_is_sparse)\ntf.sparse.reduce_sum(sp_input, axis, output_is_sparse)\n"
    + "tf.sparse.reset_shape(sp_input)\ntf.sparse.reshape(sp_input, shape)\ntf.sparse.retain(sp_input, to_retain)\ntf.sparse.slice(sp_input, start, size)\n"
    + "tf.sparse.split(sp_input, num_split, axis)\ntf.sparse.to_dense(sp_input)\ntf.sparse.to_dense(sp_input, default_value)\n"
    + "tf.sparse.to_indicator(sp_input, vocab_size)\ntf.sparse.transpose(sp_input)\ntf.sparse.transpose(sp_input, perm)"
)

TFOPERATORS = [op.strip() for op in TFOPERATORS_STR.split("\n") if op.strip() != ""]

SPARSETF_OPERATORS = [
    op.strip() for op in SPARSETF_OPERATORS_STR.split("\n") if op.strip() != ""
]

print(f"TF Operators: {len(TFOPERATORS)}, {TFOPERATORS[:5]}")
print(f"SparseTF Operators: {len(SPARSETF_OPERATORS)}, {SPARSETF_OPERATORS[:5]}")

TFOPERATORS_WEIGHT_ORDER = [
    "tf.cast(x, dtype)",
    "tf.expand_dims(input, axis)",
    "tf.constant(value)",
    "tf.squeeze(input, axis)",
    "tf.constant(value, dtype)",
    "tf.equal(x, y)",
    "tf.gather(params, indices)",
    "tf.greater(x, y)",
    "tf.matmul(a, b)",
    "tf.maximum(x, y)",
    "tf.multiply(x, y)",
    "tf.reduce_max(input_tensor)",
    "tf.reduce_max(input_tensor, axis)",
    "tf.reduce_sum(input_tensor)",
    "tf.reduce_sum(input_tensor, axis)",
    "tf.tensordot(a, b, axes)",
    "tf.transpose(a)",
    "tf.where(condition)",
    "tf.where(condition, x, y)",
    "tf.add(x, y)",
    "tf.boolean_mask(tensor, mask)",
    "tf.divide(x, y)",
    "tf.gather_nd(params, indices)",
    "tf.one_hot(indices, depth)",
    "tf.range(start)",
    "tf.reshape(tensor, shape)",
    "tf.square(x)",
    "tf.subtract(x, y)",
    "tf.tile(input, multiples)",
    "tf.argmax(input, axis)",
    "tf.greater_equal(x, y)",
    "tf.minimum(x, y)",
    "tf.sequence_mask(lengths)",
    "tf.zeros_like(input)",
    "tf.concat(values, axis)",
    "tf.gather_nd(params, indices, batch_dims)",
    "tf.ones_like(input)",
    "tf.shape(input)",
    "tf.stack(values, axis)",
    "tf.squeeze(input)",
    "tf.abs(x)",
    "tf.argsort(values, axis, stable=True)",
    "tf.eye(num_rows)",
    "tf.fill(dims, value)",
    "tf.gather(params, indices, axis, batch_dims)",
    "tf.math.bincount(arr)",
    "tf.math.segment_max(data, segment_ids)",
    "tf.math.segment_sum(data, segment_ids)",
    "tf.math.unsorted_segment_max(data, segment_ids, num_segments)",
    "tf.math.unsorted_segment_sum(data, segment_ids, num_segments)",
    "tf.pad(tensor, paddings, mode='CONSTANT')",
    "tf.reduce_any(input_tensor, axis)",
    "tf.reduce_mean(input_tensor)",
    "tf.reduce_mean(input_tensor, axis)",
    "tf.reduce_min(input_tensor)",
    "tf.reduce_min(input_tensor, axis)",
    "tf.unstack(value, axis)",
    "tf.zeros(shape)",
    "tf.add_n(inputs)",
    "tf.broadcast_to(input, shape)",
    "tf.clip_by_value(t, clip_value_min, clip_value_max)",
    "tf.math.ceil(x)",
    "tf.math.cumsum(x, axis)",
    "tf.math.floor(x)",
    "tf.math.logical_and(x, y)",
    "tf.math.logical_or(x, y)",
    "tf.not_equal(x, y)",
    "tf.ones(shape)",
    "tf.reduce_all(input_tensor, axis)",
    "tf.sequence_mask(lengths, maxlen)",
    "tf.tensor_scatter_nd_update(tensor, indices, updates)",
    "tf.transpose(a, perm)",
    "tf.argmin(input, axis)",
    "tf.argsort(values, axis, direction='DESCENDING', stable=True)",
    "tf.eye(num_rows, dtype)",
    "tf.math.cumsum(x, axis, exclusive=True)",
    "tf.math.logical_not(x)",
    "tf.math.negative(x)",
    "tf.math.segment_min(data, segment_ids)",
    "tf.math.top_k(input, k)",
    "tf.math.unsorted_segment_min(data, segment_ids, num_segments)",
    "tf.reverse(tensor, axis)",
    "tf.roll(input, shift, axis)",
    "tf.sign(x)",
    "tf.unique_with_counts(x)",
    "tf.exp(x)",
    "tf.math.divide_no_nan(x, y)",
    "tf.math.log(x)",
    "tf.math.reciprocal(x)",
    "tf.math.squared_difference(x, y)",
    "tf.pad(tensor, paddings, mode='CONSTANT', constant_values)",
    "tf.reduce_prod(input_tensor, axis)",
    "tf.repeat(input, repeats, axis)",
    "tf.round(x)",
    "tf.scatter_nd(indices, updates, shape)",
    "tf.sort(values, axis)",
    "tf.math.count_nonzero(input)",
    "tf.math.count_nonzero(input, axis)",
    "tf.math.segment_mean(data, segment_ids)",
    "tf.math.unsorted_segment_mean(data, segment_ids, num_segments)",
    "tf.range(start, limit, delta)",
    "tf.repeat(input, repeats)",
    "tf.searchsorted(sorted_sequence, values, side='left')",
    "tf.searchsorted(sorted_sequence, values, side='right')",
    "tf.sqrt(x)",
    "tf.eye(num_rows, num_columns)",
    "tf.math.logical_xor(x, y)",
    "tf.math.reciprocal_no_nan(x)",
    "tf.math.segment_prod(data, segment_ids)",
    "tf.math.unsorted_segment_prod(data, segment_ids, num_segments)",
    "tf.pad(tensor, paddings, mode='REFLECT')",
    "tf.pad(tensor, paddings, mode='SYMMETRIC')",
    "tf.sort(values, axis, direction='DESCENDING')",
]

SPARSETF_OPERATORS_WEIGHT_ORDER = [
    "tf.SparseTensor(indices, values, dense_shape)",
    "tf.sparse.from_dense(tensor)",
    "tf.sparse.to_dense(sp_input)",
    "tf.sparse.expand_dims(sp_input, axis)",
    "tf.sparse.reduce_max(sp_input, axis, output_is_sparse)",
    "tf.sparse.reduce_sum(sp_input, axis, output_is_sparse)",
    "tf.sparse.add(a, b)",
    "tf.sparse.maximum(sp_a, sp_b)",
    "tf.sparse.slice(sp_input, start, size)",
    "tf.sparse.split(sp_input, num_split, axis)",
    "tf.sparse.retain(sp_input, to_retain)",
    "tf.sparse.to_dense(sp_input, default_value)",
    "tf.sparse.transpose(sp_input)",
    "tf.sparse.concat(axis, sp_inputs)",
    "tf.sparse.minimum(sp_a, sp_b)",
    "tf.sparse.reset_shape(sp_input)",
    "tf.sparse.reshape(sp_input, shape)",
    "tf.sparse.to_indicator(sp_input, vocab_size)",
    "tf.sparse.transpose(sp_input, perm)",
]


def make_operators_section(
    shuffle=True, include_sparse=True, order_by_weight=False
) -> str:
    operators = TFOPERATORS_WEIGHT_ORDER if order_by_weight else TFOPERATORS
    sparse_operators = (
        SPARSETF_OPERATORS_WEIGHT_ORDER if order_by_weight else SPARSETF_OPERATORS
    )
    shuffled_tf = operators.copy()
    shuffled_sparse = sparse_operators.copy()
    if shuffle:
        random.shuffle(shuffled_tf)
        random.shuffle(shuffled_sparse)

    tf_str = "\n".join(shuffled_tf)
    sparse_str = "\n".join(shuffled_sparse)

    ans = f"[TENSORFLOW OPERATORS]\n{tf_str}\n"
    if include_sparse:
        ans += f"\n[SPARSE TENSORFLOW OPERATORS]\n{sparse_str}"
    return ans


OPENAI = OpenAI(
    organization=CONFIG.OPENAI_ORGANIZATION, api_key=CONFIG.OPENAI_SECRET_KEY
)

TOGETHER = OpenAI(api_key=CONFIG.TOGETHER_SECRET_KEY, base_url=CONFIG.TOGETHER_BASE_URL)


def prompt(
    system_message: str,
    user_message: str,
    n_completions: int = 10,
    model: ModelName = "gpt-4",
) -> t.List[str]:
    if model in OPENAI_MODEL_NAMES:
        client = OPENAI
    elif model in TOGETHER_MODEL_NAMES:
        client = TOGETHER
    else:
        raise ValueError(f"Invalid model: {model}")

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        n=n_completions,
        temperature=1.0,
        model=model,
    )

    if model in TOGETHER_MODEL_NAMES:
        # sleep for 1s to cover trial rate limit
        time.sleep(1)

    return [choice.message.content for choice in response.choices]


# endregion PROMPT

# region PARSING


def extract_tf_operators(code_snippet):
    pattern = r"tf\.[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"
    return set(re.findall(pattern, code_snippet))


class OperatorCoverageDict(t.TypedDict):
    tf_operators: t.Dict[str, int]
    coverage_percentage: float
    total_in_target: int
    total_covered: int


def calculate_tf_operator_coverage_and_count(
    target_operators, completion_operators
) -> OperatorCoverageDict:
    """Extend to include all completion operators and mark those used in the target program."""
    completion_operators_count = Counter(completion_operators)
    tf_operators_dict = {
        op: completion_operators_count[op] for op in completion_operators_count
    }

    # Calculate coverage based on target program operators found in completions
    covered_operators = set(target_operators).intersection(completion_operators)
    coverage_percentage = (
        len(covered_operators) / len(target_operators) * 100 if target_operators else 0
    )

    return {
        "tf_operators": tf_operators_dict,
        "coverage_percentage": coverage_percentage,
        "total_in_target": len(target_operators),
        "total_covered": len(covered_operators),
    }


CODE_BLOCK_REGEX = r".*```(?:[\w\s]+)?\n(.*?)\n```.*"
CODE_LINE_REGEX = r".*`(.*)`.*"
DEF_LINE_REGEX = r"\s*def\s+\w+\(.*\):"
RETURN_REGEX = r"([\s]+)return"
ASSIGN_REGEX = r"([\s]+)([a-zA-Z_][a-zA-Z0-9_\s,]*)([\s]+)="
IMPORT_REGEX = r"([\s]+)import"
CATCHALL_REGEX = r"^([\s]+)(.*)"


def normalize_completion(completion: str) -> str:
    match = re.match(CODE_BLOCK_REGEX, completion, re.DOTALL)
    if match:
        completion = match.group(1)
    else:
        match = re.match(CODE_LINE_REGEX, completion)
        if match:
            completion = match.group(1)

    try:
        ast.parse(completion)
        return completion
    except SyntaxError:
        pass

    lines = completion.split("\n")
    if re.match(DEF_LINE_REGEX, lines[0]):
        return completion

    normalized_lines = []
    for line in lines:
        normalized_line = re.sub(RETURN_REGEX, "return", line)
        normalized_line = re.sub(ASSIGN_REGEX, r"\2 =", normalized_line)
        normalized_line = re.sub(IMPORT_REGEX, "import", normalized_line)
        normalized_line = re.sub(CATCHALL_REGEX, r"\2", normalized_line)
        normalized_lines.append(normalized_line)
    normalized_completion = "\n".join(normalized_lines)
    return normalized_completion


class ConstantVisitor(ast.NodeVisitor):
    def __init__(self):
        self.constants = []

    def visit_Constant(self, node: ast.Constant):
        if type(node.value) == int:
            self.constants.append(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            self.constants.append(-node.operand.value)
        else:
            self.generic_visit(node)


def get_constants(ast_node: ast.Module) -> t.List[str]:
    visitor = ConstantVisitor()
    visitor.visit(ast_node)
    return visitor.constants


class ConstantCounts(t.TypedDict):
    common: int
    axis: int
    shape: int
    provided: int
    tf_int32: int
    tf_float32: int
    tf_int64: int
    tf_bool: int
    input_var: int
    shape_tuple: int


def add_constant_counts(
    counts1: ConstantCounts, counts2: ConstantCounts
) -> ConstantCounts:
    return {
        key: counts1.get(key, 0) + counts2.get(key, 0)
        for key in set(counts1.keys()) | set(counts2.keys())
    }


def is_common(value: t.Any) -> bool:
    return (type(value) == int or type(value) == bool) and value in [
        0,
        1,
        -1,
        True,
        False,
    ]


def is_axis(value: t.Any, max_input_rank) -> bool:
    return type(value) == int and value in range(2, max_input_rank + 1)


def is_shape(value: t.Any, dimension_lengths) -> bool:
    return type(value) == int and value in dimension_lengths


class CountConstantVisitor(ast.NodeVisitor):
    counts: ConstantCounts
    max_input_rank: int
    dimension_lengths: t.Set[int]
    output_shape: t.Optional[t.Tuple[int, ...]]

    should_print: bool

    def __init__(
        self,
        max_input_rank: int,
        dimension_lengths: t.Set[int],
        output_shape: t.Optional[t.Tuple[int, ...]] = None,
        should_print=False,
    ):
        self.counts = {
            "common": 0,
            "axis": 0,
            "shape": 0,
            "provided": 0,
            "tf_int32": 0,
            "tf_float32": 0,
            "tf_int64": 0,
            "tf_bool": 0,
            "input_var": 0,
            "shape_tuple": 0,
        }
        self.max_input_rank = max_input_rank
        self.dimension_lengths = dimension_lengths
        self.output_shape = output_shape
        self.should_print = should_print

    def __print(self, *args):
        if self.should_print:
            print(*args)

    def visit_Constant(self, node: ast.Constant):
        if is_common(node.value):
            self.counts["common"] += 1
            self.__print("common", node.value)
        elif is_axis(node.value, self.max_input_rank):
            self.counts["axis"] += 1
            self.__print("axis", node.value)
        elif is_shape(node.value, self.dimension_lengths):
            self.counts["shape"] += 1
            self.__print("shape", node.value)
        elif type(node.value) == int:
            self.counts["provided"] += 1
            self.__print("provided", node.value)
        else:
            self.__print("unknown constant", node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            value = -node.operand.value
            if is_common(value):
                self.counts["common"] += 1
                self.__print("common", value)
            elif type(value) == int:
                self.counts["provided"] += 1
                self.__print("provided", value)
            else:
                self.__print("unknown constant", -node.operand.value)
        else:
            self.generic_visit(node)

    def visit_Name(self, node: Name):
        if (
            node.id.startswith("in")
            or node.id == "tensor"
            or node.id == "indices"
            or node.id == "updates"
        ):
            self.counts["input_var"] += 1
            self.__print("input_var", node.id)
        else:
            self.__print("unknown name", node.id)

    def visit_Attribute(self, node: Attribute):
        code = ast.unparse(node)
        if code == "tf.int32":
            self.counts["tf_int32"] += 1
            self.__print("tf.int32")
        elif code == "tf.float32":
            self.counts["tf_float32"] += 1
            self.__print("tf.float32")
        elif code == "tf.int64":
            self.counts["tf_int64"] += 1
            self.__print("tf.int64")
        elif code == "tf.bool":
            self.counts["tf_bool"] += 1
            self.__print("tf.bool")
        else:
            self.generic_visit(node.value)

    def visit_Tuple(self, node: ast.Tuple):
        try:
            value = ast.literal_eval(node)
        except:
            value = None
        if (
            value is not None
            and len(value) == len(self.output_shape)
            and all([v == o for v, o in zip(value, self.output_shape)])
        ):
            self.counts["shape_tuple"] += 1
            self.__print("shape_tuple", value)
        else:
            for elt in node.elts:
                self.generic_visit(elt)


def count_constants(
    ast_node: ast.Module, example: Example, should_print=False
) -> ConstantCounts:
    visitor = CountConstantVisitor(
        example.max_input_rank,
        example.dimension_lengths,
        example.output_shape,
        should_print,
    )
    visitor.visit(ast_node)
    return visitor.counts


# endregion PARSING

if __name__ == "__main__":
    main()
