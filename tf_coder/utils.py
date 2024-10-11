import os

# disable tensorflow debug logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

tf.get_logger().disabled = True
tf.autograph.set_verbosity(0)

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
import click
from datetime import datetime
import logging
import time
import tokenize
import io

# need this to properly evaluate code, even though it isn't used explicitly
import math

LOGGER = logging.getLogger(__name__)

DATASET_FILE = CONFIG.ROOT_DIR / "tf_coder/tfcoder_dataset.json"
# these were originally randomly sampled, but we're keeping them hardcoded for consistency between runs
IN_CONTEXT_TASK_NAMES = ["google_02", "google_10", "stackoverflow_07"]


ModelName = t.Literal[
    "gpt-4",
    "gpt-4o-2024-05-13",
    "gpt-3.5-turbo",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
]
OPENAI_MODEL_NAMES = ["gpt-4", "gpt-4o-2024-05-13", "gpt-3.5-turbo"]
TOGETHER_MODEL_NAMES = [
    "deepseek-ai/deepseek-coder-33b-instruct",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
]
MODEL_NAMES = OPENAI_MODEL_NAMES + TOGETHER_MODEL_NAMES

# region CLI


def compute_output_file(model: ModelName, n: t.Optional[int] = None):
    return (
        CONFIG.ROOT_DIR
        / "tf_coder"
        / f"tfcoder_output.{model.replace('/', '__')}{'.' if n is None else '.' + str(n) + '.'}json"
    )


def setup_logging(name: str):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename=CONFIG.ROOT_DIR
        / "tf_coder"
        / f"utils_{name.replace('/', '__')}_{now}.log",
        filemode="w",
    )


def main():
    original_print = builtins.print
    builtins.print = tqdm_print

    cli()


@click.group()
def cli():
    pass


@cli.command(help="resample completions for all benchmarks")
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODEL_NAMES),
    required=True,
    help="The model to use for sampling completions",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    multiple=True,
    help="The number of completions to sample for each problem",
)
def resample(model: ModelName, num_samples: t.List[int]):
    output_file = compute_output_file(model)
    if output_file.exists():
        dataset_json: t.List[TaskJSONWithOutput] = json.loads(output_file.read_text())
        tasks = [Task.from_json_with_output(task_json) for task_json in dataset_json]
    else:
        print(f"Output file {output_file} does not exist")
        return

    for n in num_samples:
        print(f"Resampling for {n} completions")
        resampled = []
        for task in tasks:
            if task.name in IN_CONTEXT_TASK_NAMES:
                continue

            if len(task.completions) < n:
                print(f"Task {task.name} has less than {n} usable completions")
                continue

            idxs = random.sample(range(len(task.completions)), n)
            resampled_task = Task(
                task.name,
                task.description,
                task.target_program,
                task.source,
                task.constants,
                task.examples,
                completions=[task.completions[i] for i in idxs],
                parsed_constants=[task.parsed_constants[i] for i in idxs],
                constant_counts=[task.constant_counts[i] for i in idxs],
            )
            resampled_task.compute_asts()
            resampled_task.compute_operator_coverage()
            resampled_task.compute_constants()
            resampled.append(resampled_task)
        resampled_output_file = compute_output_file(model, n)
        write_output(resampled, resampled_output_file)


@cli.command(help="merge files with completions")
@click.option(
    "-f",
    "--file",
    multiple=True,
    type=click.Path(exists=True),
    required=True,
    help="The files to merge",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="The output file",
)
def merge(file, output):
    merged: t.List[TaskJSONWithOutput] = []
    for f in file:
        with open(f, "r") as file:
            print(f"merging file {f}")
            tasks_json: t.List[TaskJSONWithOutput] = json.load(file)
            if len(merged) == 0:
                for task_json in tasks_json:
                    merged_task = {
                        **task_json,
                        "completions": [],
                        "parsed_constants": [],
                        "all_constants": [],
                        "constant_counts": [],
                        "aggregate_constant_count": {},
                    }
                    merged.append(merged_task)

            for i, task_json in enumerate(tasks_json):
                merged[i]["completions"].extend(task_json["completions"])
                merged[i]["parsed_constants"].extend(task_json["parsed_constants"])
                merged[i]["all_constants"].extend(task_json["all_constants"])
                merged[i]["constant_counts"].extend(task_json["constant_counts"])
                merged[i]["aggregate_constant_count"] = add_constant_counts(
                    merged[i]["aggregate_constant_count"],
                    task_json["aggregate_constant_count"],
                )

    with open(output, "w") as file:
        json.dump(merged, file, indent=4)


@cli.command(help="parse and recompute operators and constants")
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODEL_NAMES),
    required=True,
    help="The model to use for sampling completions",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    required=False,
    help="The number of completions to sample for each problem. used to identify the output file",
)
def parse(model, num_samples):
    setup_logging(f"{model}_parse")
    output_file = compute_output_file(model, num_samples)
    if output_file.exists():
        dataset_json: t.List[TaskJSONWithOutput] = json.loads(output_file.read_text())
        tasks = [Task.from_json_with_output(task_json) for task_json in dataset_json]
    else:
        print(f"Output file {output_file} does not exist")
        return

    tasks = [task for task in tasks if task.name not in IN_CONTEXT_TASK_NAMES]

    def write():
        write_output(tasks, output_file)

    print(f"computing asts")
    for task in tqdm(tasks):
        task.compute_asts()
        write()

    print(f"computing operator coverage")
    for task in tqdm(tasks):
        task.compute_operator_coverage()
        write()

    print(f"computing constants")
    for task in tqdm(tasks):
        task.compute_constants()
        write()


@cli.command(help="print statistics about completions")
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODEL_NAMES),
    required=True,
    help="The model to use for sampling completions",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    required=False,
    help="The number of completions to sample for each problem. used to identify the output file",
)
@click.option(
    "-lt",
    "--log-task-stats",
    is_flag=True,
    default=False,
    help="Whether to log stats for each task",
)
@click.option(
    "-lr",
    "--log-runtime-errors",
    is_flag=True,
    default=False,
    help="Whether to log runtime errors",
)
def stats(model, num_samples, log_task_stats, log_runtime_errors):
    output_file = compute_output_file(model, num_samples)
    if output_file.exists():
        dataset_json: t.List[TaskJSONWithOutput] = json.loads(output_file.read_text())
        tasks = [Task.from_json_with_output(task_json) for task_json in dataset_json]
    else:
        print(f"Output file {output_file} does not exist")
        return

    num_completions = sum([len(task.completions) for task in tasks])
    average_num_completion = num_completions / len(tasks)

    aggregate_correctness_stats: CorrectnessStats = {
        "total": 0,
        "num_correct": 0,
        "num_correct_tasks": 0,
        "num_incorrect": 0,
        "num_runtime_error": 0,
        "num_syntax_error": 0,
    }
    for task in tasks:
        correctness_stats = task.compute_correctness_stats(
            log_runtime_errors=log_runtime_errors
        )
        if log_task_stats:
            print(f"# {task.name}")
            print(f"total: {correctness_stats['total']}")
            print(f"num correct: {correctness_stats['num_correct']}")
            print(f"num incorrect samples: {correctness_stats['num_incorrect']}")
            print(f"num runtime error: {correctness_stats['num_runtime_error']}")
            print(f"num syntax error: {correctness_stats['num_syntax_error']}")
            print()
            print()

        aggregate_correctness_stats["total"] += correctness_stats["total"]
        aggregate_correctness_stats["num_correct"] += correctness_stats["num_correct"]
        aggregate_correctness_stats["num_correct_tasks"] += correctness_stats[
            "num_correct_tasks"
        ]
        aggregate_correctness_stats["num_incorrect"] += correctness_stats[
            "num_incorrect"
        ]
        aggregate_correctness_stats["num_runtime_error"] += correctness_stats[
            "num_runtime_error"
        ]
        aggregate_correctness_stats["num_syntax_error"] += correctness_stats[
            "num_syntax_error"
        ]

    print(f"Total tasks: {len(tasks)}")
    print(f"Total completions: {num_completions}")
    print(f"Average completions per task: {average_num_completion}")

    num_tasks_with_operator_coverage = sum(
        [0 if len(task.tf_operators.keys()) == 0 else 1 for task in tasks]
    )

    print(f"Total tasks with operator coverage: {num_tasks_with_operator_coverage}")

    num_constants = sum(
        [
            len(
                [
                    constants
                    for constants in task.constant_counts
                    if constants is not None
                ]
            )
            for task in tasks
        ]
    )
    average_num_constants = num_constants / len(tasks)

    print(f"Total num completions with constants: {num_constants}")
    print(f"Average num completions with constants per task: {average_num_constants}")

    print()
    print()

    print(f"# aggregate evaluation stats")
    print(f"total: {aggregate_correctness_stats['total']}")
    print(f"num correct samples: {aggregate_correctness_stats['num_correct']}")
    print(f"num correct tasks: {aggregate_correctness_stats['num_correct_tasks']}")
    print(f"num incorrect: {aggregate_correctness_stats['num_incorrect']}")
    print(f"num runtime error: {aggregate_correctness_stats['num_runtime_error']}")
    print(f"num syntax error: {aggregate_correctness_stats['num_syntax_error']}")
    print()
    print()


@cli.command(help="sample completions")
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODEL_NAMES),
    required=True,
    help="The model to use for sampling completions",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    default=100,
    help="The number of completions to sample for each problem",
)
def sample(model, num_samples):
    setup_logging(model)
    output_file = compute_output_file(model)

    if output_file.exists():
        dataset_json: t.List[TaskJSONWithOutput] = json.loads(output_file.read_text())
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
        write_output(tasks, output_file)

    print(f"Tasks: {len(tasks)}")
    print(f"generating completions. model: {model}, num_completions: {num_samples}")
    for task in tqdm(tasks):
        try:
            task.get_completions(
                in_context_section, model=model, n_completions=num_samples
            )
            write()
        except Exception as e:
            print(f"Error sampling completions for task {task.name}")
            print(f"Error: {e}")
            traceback.print_exc()
            continue

    # print(f"computing asts")
    # for task in tqdm(tasks):
    #     task.compute_asts()
    #     write()

    # print(f"computing operator coverage")
    # for task in tqdm(tasks):
    #     task.compute_operator_coverage()
    #     write()

    # print(f"computing constants")
    # for task in tqdm(tasks):
    #     task.compute_constants()
    #     write()


def tqdm_print(*args, **kwargs):
    # if no arguments are passed, write the empty string
    if not args:
        args = [""]
    tqdm.write(*args, **kwargs)
    LOGGER.info(" ".join([str(arg) for arg in args]))


def write_output(tasks: t.List["Task"], output_file: Path):
    output_json = [task.to_json() for task in tasks]
    output_file.write_text(json.dumps(output_json, indent=4))


# endregion CLI
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
    time_millis: float
    usage: t.Any


class TaskJSONWithOutput(TaskJSON):
    completions: t.List[str]
    normalized_completions: t.List[str]
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
    time_millis: float = 0.0
    usage: t.Any = None
    normalized_completions: t.List[str] = field(default_factory=list)
    tf_operators: t.Dict[str, int] = field(default_factory=dict)
    lex_tf_operators: t.Dict[str, int] = field(default_factory=dict)
    coverage_percentage: float = 0.0
    total_covered: int = 0
    total_in_target: int = 0
    parsed_constants: t.List[t.List[int]] = field(default_factory=list)
    all_constants: t.List[int] = field(default_factory=list)
    constant_counts: t.List["ConstantCounts"] = field(default_factory=list)
    aggregate_constant_count: "ConstantCounts" = field(default_factory=dict)

    __asts: t.List[ast.Module] = field(default_factory=list)
    __tokens: t.List[t.List[tokenize.TokenInfo]] = field(default_factory=list)

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
        if "normalized_completions" in task_json:
            ans.normalized_completions = task_json["normalized_completions"]
        if "tf_operators" in task_json:
            ans.tf_operators = task_json["tf_operators"]
        if "lex_tf_operators" in task_json:
            ans.lex_tf_operators = task_json["lex_tf_operators"]
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
        if "time_millis" in task_json:
            ans.time_millis = task_json["time_millis"]
        if "usage" in task_json:
            ans.usage = task_json["usage"]
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
            "normalized_completions": self.normalized_completions,
            "tf_operators": self.tf_operators,
            "lex_tf_operators": self.lex_tf_operators,
            "coverage_percentage": self.coverage_percentage,
            "total_covered": self.total_covered,
            "total_in_target": self.total_in_target,
            "parsed_constants": self.parsed_constants,
            "all_constants": self.all_constants,
            "constant_counts": self.constant_counts,
            "aggregate_constant_count": self.aggregate_constant_count,
            "time_millis": self.time_millis,
            "usage": self.usage,
        }

    @property
    def function_definition(self) -> str:
        return f"def transform({self.examples.formals}):"

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
{self.function_definition}
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

        completions, time_millis, usage = prompt(
            SYSTEM_PROMPT,
            self.make_user_message(in_context_section),
            n_completions=n_completions,
            model=model,
        )

        self.completions = completions
        self.time_millis = time_millis
        self.usage = usage
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

        for i, completion in enumerate(self.completions):
            if self.__asts[i] is not None:
                continue

            tokens = self.__tokens[i]
            if tokens is None:
                continue

            completion_operators = count_operators_lexed(tokens)

            for operator, count in completion_operators.items():
                if operator in self.lex_tf_operators:
                    self.lex_tf_operators[operator] += count
                else:
                    self.lex_tf_operators[operator] = count

        return coverage_dict

    def compute_asts(self):
        num_errored = 0
        self.normalized_completions = []
        for completion in self.completions:
            extracted = extract_code(completion)
            normalized = None

            if extracted is None:
                num_errored += 1
                print(f"# Error extracting completion")
                print(f"## completion")
                print('"""' + completion + '"""')
                print()
                print()
                self.__asts.append(None)
                self.__tokens.append(None)
                self.normalized_completions.append(None)
                continue

            normalized = normalize_code(extracted, self.function_definition)
            try:
                self.__asts.append(ast.parse(normalized))
                self.__tokens.append(lex(normalized))
                self.normalized_completions.append(normalized)
            except Exception:
                num_errored += 1
                print(f"# Error parsing completion")
                print(f"## completion)")
                print(completion)
                print()
                print("## Normalized")
                print(normalized)
                print()
                print(f"## Error:")
                traceback.print_exc()
                print()
                print()
                self.__asts.append(None)
                self.__tokens.append(lex(normalized))
                self.normalized_completions.append(None)
        print(f"Errored: {num_errored}")

    def compute_constants(self):
        self.parsed_constants = [
            (
                get_constants(ast_node)
                if ast_node
                else get_constants_lexed(tokens) if tokens is not None else []
            )
            for ast_node, tokens in zip(self.__asts, self.__tokens)
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
            (
                count_constants(ast_node, self.examples)
                if ast_node
                else count_constants_lexed(tokens, self.examples) if tokens else None
            )
            for ast_node, tokens in zip(self.__asts, self.__tokens)
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

    def compute_correctness_stats(self, log_runtime_errors=False) -> "CorrectnessStats":
        ans = {
            "total": 0,
            "num_correct_tasks": 0,
            "num_correct": 0,
            "num_incorrect": 0,
            "num_runtime_error": 0,
            "num_syntax_error": 0,
        }

        for normalized_completion in self.normalized_completions:
            ans["total"] += 1

            if normalized_completion is None:
                ans["num_syntax_error"] += 1
                continue

            # these programs cause a segfault in TF, and I don't wanna come up with a graceful way to handle them!
            if normalized_completion.strip() in [
                """def transform(in1):
    argmax = tf.cast(tf.argmax(in1, axis=1), tf.int32)
    shape = tf.shape(in1, out_type=tf.int32)
    indices = tf.meshgrid(tf.range(shape[0]), argmax, indexing='ij')
    updates = tf.ones_like(argmax)
    return tf.tensor_scatter_nd_update(tf.zeros_like(in1, dtype=tf.int32), tf.stack(indices, axis=-1), updates)""",
                """def transform(in1):
    max_value = tf.reduce_max(in1)
    indices = tf.stack([tf.cast(in1, tf.int32), tf.zeros(tf.shape(in1), tf.int32)], -1)
    on_values = tf.ones(tf.shape(in1)[:-1], tf.int32)
    return tf.scatter_nd(indices, on_values, [max_value+1, 1])[:, 0]""",
                """in1 = tf.constant(in1)
out1 = tf.add(in1[0::2], in1[1::2])

out1
inburgh_airport = tf.zeros((32,))
freswick_airport = tf.zeros((32,))
inns_bridge = tf.zeros((32,))
craiglockhart = tf.zeros((32,))

def filter(a):
    avg = tf.reduce_mean(a)
    stddev = tf.math.reduce_std(a)
    return a[(a > (avg - 2 * stddev)) & (a < (avg + 2 * stddev))]

for i in range(1000):
    u = tf.random.normal((32,), mean=0, stddev=1)
    v = tf.random.normal((32,), mean=1, stddev=1)
    w = tf.random.normal((32,), mean=2, stddev=1)
    array = filter(u) + filter(v) + filter(w)
    array = array / tf.norm(array)
    array = tf.math""",
                """def transform(in1, in2):
    return tf.sparse.reorder(tf.sparse.from_dense(tf.scatter_nd(tf.expand_dims(in2, 1), in1, [3, 5])))""",
                """def transform(in1):
    return tf.scatter_nd(in1[:, None], tf.ones(tf.shape(in1)[0], tf.int32), [tf.shape(in1)[0], 9])""",
                """def transform(in1):
    return tf.scatter_nd(in1[:, None], tf.ones(tf.shape(in1)[0], tf.int32), (tf.shape(in1)[0], 9))""",
            ]:
                ans["num_runtime_error"] += 1
                continue

            try:
                # print("### total so far: " + str(ans["total"]))
                if log_runtime_errors:
                    print("### running completion")
                    print('"""' + normalized_completion.strip() + '"""')
                try:
                    blockPrint()
                    exec(normalized_completion)
                except:
                    # runtime errors in the completion are OK as long as the transform function is defined and runs without errors.
                    pass
                finally:
                    enablePrint()
                transform_fn = locals()["transform"]
                output = transform_fn(*self.examples.inputs)
                if matches_expected_value(output, self.examples.output):
                    ans["num_correct"] += 1
                    ans["num_correct_tasks"] = 1
                else:
                    ans["num_incorrect"] += 1
            except Exception as e:
                if log_runtime_errors:
                    print("# completion for " + self.name)
                    print(normalized_completion)
                    print()
                    print("## exception")
                    print(str(e))
                    print()
                    print()
                ans["num_runtime_error"] += 1

        return ans


# region EXECUTING CODE


def matches_expected_value(
    actual: t.Union[np.ndarray, tf.SparseTensor],
    expected: t.Union[np.ndarray, tf.SparseTensor],
) -> bool:
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return np.array_equal(actual, expected)
    elif isinstance(actual, tf.SparseTensor) and isinstance(expected, tf.SparseTensor):
        return tf.sparse.equal(actual, expected)
    else:
        return False


class CorrectnessStats(t.TypedDict):
    total: int
    num_correct: int
    num_correct_tasks: int
    num_incorrect: int
    num_runtime_error: int
    num_syntax_error: int


def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# endregion EXECUTING CODE

# region PROMPT

SYSTEM_PROMPT = """You are a coding assistant. Be precise and terse.
You will be provided a list of tensorflow operators, a task description, and some input/output examples.
Your task is to generate the body of a python function that will transform the input to the output.
Only use the operators provided in the list.
Your answer should be as short as possible while still being correct.
Make sure to only generate python code.
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
) -> t.Tuple[t.List[str], float, t.Any]:
    if model in OPENAI_MODEL_NAMES:
        client = OPENAI
    elif model in TOGETHER_MODEL_NAMES:
        client = TOGETHER
    else:
        raise ValueError(f"Invalid model: {model}")

    start = datetime.now()
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        n=n_completions,
        temperature=1.0,
        model=model,
        max_tokens=300,
        stop=["[TASK DESCRIPTION]", "[SYSTEM]", "[USER]"],
    )
    end = datetime.now()
    diff = end - start
    time_millis = diff.total_seconds() * 1000
    
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    if model in TOGETHER_MODEL_NAMES:
        # sleep for 1s to cover trial rate limit
        time.sleep(1)

    return [choice.message.content for choice in response.choices], time_millis, usage


# endregion PROMPT

# region LEXING


def lex(code: str) -> t.List[tokenize.TokenInfo]:
    ans = []
    try:
        for token in tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline):
            ans.append(token)
    except:
        pass
    return ans


ALL_OPERATORS = TFOPERATORS + SPARSETF_OPERATORS


def count_operators_lexed(tokens: t.List[tokenize.TokenInfo]) -> t.Dict[str, int]:
    seen_tf = False
    seen_sparse = False

    ans = {}

    for token in tokens:
        if token.string == ".":
            continue

        if token.string == "tf":
            seen_tf = True
            continue

        if seen_tf:
            if seen_sparse:
                if any(
                    f"tf.sparse.{token.string}(" in operator
                    for operator in ALL_OPERATORS
                ):
                    ans[f"tf.sparse.{token.string}"] = (
                        ans.get(f"tf.sparse.{token.string}", 0) + 1
                    )
                seen_sparse = False
                seen_tf = False
            elif token.string == "sparse":
                seen_sparse = True
            else:
                if any(f"tf.{token.string}(" in operator for operator in ALL_OPERATORS):
                    ans[f"tf.{token.string}"] = ans.get(f"tf.{token.string}", 0) + 1
                seen_tf = False

    return ans


def get_constants_lexed(tokens: t.List[tokenize.TokenInfo]) -> t.List[int]:
    ans = []
    seen_minus = False
    for token in tokens:
        if token.type == tokenize.OP:
            if token.string == "-":
                seen_minus = True
                continue
        elif token.type == tokenize.NUMBER:
            try:
                value = -int(token.string) if seen_minus else int(token.string)
            except:
                continue
            ans.append(value)

        seen_minus = False
    return list(set(ans))


def count_constants_lexed(
    tokens: t.List[tokenize.TokenInfo], example: Example
) -> "ConstantCounts":
    counts = {
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
    # example.max_input_rank,
    #     example.dimension_lengths,
    #     example.output_shape,
    seen_minus = False
    seen_tf = False
    for token in tokens:
        if token.type == tokenize.OP:
            if token.string == "-":
                seen_minus = True
                continue
        elif token.type == tokenize.NUMBER:
            try:
                value = -int(token.string) if seen_minus else int(token.string)
            except:
                continue
            if is_common(value):
                counts["common"] += 1
            elif is_axis(value, example.max_input_rank):
                counts["axis"] += 1
            elif is_shape(value, example.dimension_lengths):
                counts["shape"] += 1
            else:
                counts["provided"] += 1
        elif token.type == tokenize.NAME:
            if token.string == "tf":
                seen_tf = True
                continue
            elif seen_tf:
                if token.string == "int32":
                    counts["tf_int32"] += 1
                elif token.string == "float32":
                    counts["tf_float32"] += 1
                elif token.string == "int64":
                    counts["tf_int64"] += 1
                elif token.string == "bool":
                    counts["tf_bool"] += 1
        elif token.type == tokenize.DOT:
            continue

        # TODO: shape tuple (not sure if it's worth implementing at the lexer level)
        # if we see an open paren, take all the tokens until the close paren, attempt to parse a shape tuple out of them. if that works, count as shape. else, treat as tokens
        seen_minus = False
        seen_tf = False

    return counts


# endregion LEXING

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


def extract_code(completion: str) -> t.Optional[str]:
    try:
        ast.parse(completion)
        return completion
    except:
        pass

    plain_code_result = extract_plain_code(completion)
    code_block_result = extract_code_block(completion)

    if plain_code_result is None:
        return code_block_result

    if code_block_result is None:
        return plain_code_result

    if code_block_result in plain_code_result:
        return code_block_result

    return plain_code_result


def normalize_code(code: str, def_line: str) -> str:
    if "def" in code:
        return code

    try:
        _ast = ast.parse(code)
        # if ast is an expression
        if isinstance(_ast.body[0], ast.Expr):
            code = f"return {code.strip()}"
    except:
        pass

    indent_level = detect_indent_level(code)
    if indent_level == 0:
        indent_level = 4
    lowest_indent_level, highest_indent_level = indent_level_range(code)

    lines = code.split("\n")
    indented_lines = [
        (" " * indent_level) + line[lowest_indent_level:] for line in lines
    ]
    return def_line + "\n" + "\n".join(indented_lines)


HEADER_REGEX = r"^\w*\[[\w\s]+\]\w*$"


def remove_empty_or_header_leading_lines(code: str) -> str:
    lines = code.split("\n")
    ans = []
    in_code = False
    # remove any lines with just whitespace from the beginning and end
    for line in lines:
        is_empty_or_header = line.strip() == "" or (
            re.match(HEADER_REGEX, line) is not None
        )
        if is_empty_or_header and not in_code:
            continue
        if not is_empty_or_header:
            in_code = True
        ans.append(line)

    return "\n".join(ans)


DEF_LINE_REGEX = r"\s*def\s+\w+\(.*\):"
DEF_TRANSFORM_REGEX = r"^\s*def\s+transform\(.*\):"
RETURN_REGEX = r"([\s]*)return"


def extract_plain_code(completion: str) -> t.Optional[str]:
    lines = remove_empty_or_header_leading_lines(completion).split("\n")

    match = re.search(DEF_TRANSFORM_REGEX, completion, re.MULTILINE)
    if match:
        # lines = lines after def transform
        def_idx = next(
            (i for i, line in enumerate(lines) if re.match(DEF_TRANSFORM_REGEX, line)),
            None,
        )
        return_ids = next(
            (i for i, line in enumerate(lines) if re.search(RETURN_REGEX, line)), None
        )
        if def_idx is not None:
            lines = lines[def_idx:]

    code_lines = []
    seen_multiline_return = False
    for line in lines:
        code_lines.append(line)
        if re.search(RETURN_REGEX, line):
            # unmatched open-parens indicate a multi-line return
            if line.count("(") > line.count(")"):
                seen_multiline_return = True
            else:
                return "\n".join(code_lines)
        if seen_multiline_return and line.strip() == "":
            return "\n".join(code_lines)

    if seen_multiline_return:
        return "\n".join(code_lines)
    else:
        return None


CODE_BLOCK_REGEX = r"```((?:(?!\n).)*\n)?((?:(?!```).)*)```"


def extract_code_block(completion: str) -> t.Optional[str]:
    match = re.search(CODE_BLOCK_REGEX, completion, re.DOTALL)
    if match:
        group_number = len(match.groups())
        return remove_empty_or_header_leading_lines(match.group(group_number)).rstrip()
    else:
        return None


def detect_indent_level(code):
    lines = code.split("\n")
    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]

    if not indent_levels:
        return 0

    gcd_indent_level = indent_levels[0]
    for indent_level in indent_levels[1:]:
        gcd_indent_level = math.gcd(gcd_indent_level, indent_level)

    return gcd_indent_level


def indent_level_range(code):
    lines = code.split("\n")
    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]

    if not indent_levels:
        return 0, 0

    return min(indent_levels), max(indent_levels)


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
            and self.output_shape is not None
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
