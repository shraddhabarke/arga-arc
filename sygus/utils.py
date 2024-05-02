import os
import sys
from pathlib import Path

CURRENT_DIRECTORY = Path(os.getcwd())
ROOT_DIRECTORY = (CURRENT_DIRECTORY / "..").absolute().resolve()

print(f"Current directory: {CURRENT_DIRECTORY}")
print(f"Root directory: {ROOT_DIRECTORY}")

sys.path.append(str(ROOT_DIRECTORY))

import typing as t
from pprint import pprint
from dataclasses import dataclass
import sexpdata as sexp
from sexpdata import Symbol
from openai import OpenAI
from config import CONFIG
from datetime import datetime
import json
import random
import math
import re
import time
import traceback

OPENAI = OpenAI(
    organization=CONFIG.OPENAI_ORGANIZATION, api_key=CONFIG.OPENAI_SECRET_KEY
)
client = OpenAI(
    organization=CONFIG.OPENAI_ORGANIZATION, api_key=CONFIG.OPENAI_SECRET_KEY
)

TOGETHER = OpenAI(api_key=CONFIG.TOGETHER_SECRET_KEY, base_url=CONFIG.TOGETHER_BASE_URL)

ExampleTuple = t.Tuple[t.List[str], str]


@dataclass
class SygusProblem:
    synth_fun: str
    signature: t.Tuple[str, t.List[t.Any], t.Any]
    examples: t.List[t.Tuple[t.List[str], str]]
    natural_language_spec: str

    @classmethod
    def from_sexps(cls, sexps: t.List[t.Any], comments: t.List[str]) -> "SygusProblem":
        constraints = get_constraints(sexps)
        examples = [constraint_to_io(constraint) for constraint in constraints]
        return cls.from_sexps_with_examples(sexps, comments, examples)

    @classmethod
    def from_sexps_with_examples(
        cls,
        sexps: t.List[t.Any],
        comments: t.List[str],
        examples: t.List[t.Tuple[str, str]],
    ) -> "SygusProblem":
        synth_fun = get_synth_fun(sexps)
        assert synth_fun is not None
        synth_fun_str = sexp.dumps(synth_fun)

        signature = get_signature(synth_fun)
        num_args = len(signature[1])

        return cls(
            synth_fun=synth_fun_str,
            signature=signature,
            examples=examples,
            natural_language_spec="\n".join(comments),
        )

    @property
    def num_args(self) -> int:
        return len(self.signature[1])

    @property
    def function_definition_prefix(self) -> str:
        return f"(define-fun {self.signature[0]} ({' '.join([f'{arg[0]} {arg[1]}' for arg in self.signature[1]])}) {self.signature[2]}"

    def completion_to_function_definition(self, completion: str) -> t.Optional[str]:
        try:
            completion = cleanup_completion(completion)
            return add_sygus_prefix(completion, self.function_definition_prefix)
        except Exception as e:
            print(f"Error processing completion: {e}")
            print(traceback.format_exc())
            return None

    @property
    def user_message(self) -> str:
        EXAMPLES = ""
        for args, output in self.examples:
            EXAMPLES += f"{' , '.join(args)} -> {output}\n"
        return f"""[GRAMMAR]
{self.synth_fun}

[NATURAL LANGUAGE SPECIFICATION]
{self.natural_language_spec}

[EXAMPLES]
{EXAMPLES}

[SOLUTION]
{self.function_definition_prefix}"""


SYSTEM_PROMPT = """You are a coding assistant. Be precise and terse.
You will be given a SyGuS grammar, a natural language specification, and a set of input-output examples.
Your task is to complete the provided function definition with an implementation that is correct according to the grammar, specification, and examples.
Make sure that your answer is a valid s-expression."""

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

CODE_BLOCK_REGEX = r".*```(?:\w+)?\n(.*?)\n```.*"
CODE_LINE_REGEX = r".*`(.*)`.*"


def cleanup_completion(completion: str) -> str:
    def remove_leading_close_paren(completion: str) -> str:
        return (
            completion.strip()[1:]
            if completion.strip().startswith(")")
            else completion.strip()
        )

    match = re.match(CODE_BLOCK_REGEX, completion, re.DOTALL)
    if match:
        return remove_leading_close_paren(match.group(1))

    match = re.match(CODE_LINE_REGEX, completion)
    if match:
        return remove_leading_close_paren(match.group(1))

    return completion


def sample_gpt_solutions(
    problem: SygusProblem, n: int = 10, model: ModelName = "gpt-4"
) -> t.Tuple[t.List[str], int]:
    if model in OPENAI_MODEL_NAMES:
        client = OPENAI
    elif model in TOGETHER_MODEL_NAMES:
        client = TOGETHER
    else:
        raise ValueError(f"Invalid model: {model}")

    start_time = datetime.now()
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.user_message},
        ],
        n=n,
        temperature=0.5,
        model=model,
    )
    end_time = datetime.now()
    time_diff_ms = (end_time - start_time).microseconds / 1000

    if model in TOGETHER_MODEL_NAMES:
        # sleep for 1s to cover trial rate limit
        time.sleep(1)

    return [choice.message.content for choice in response.choices], time_diff_ms


class CompletionJSON(t.TypedDict):
    completions: t.List[str]
    solutions: t.List[str]
    time_ms: float


class SygusBenchmark:
    problems: t.Dict
    comments: t.Dict
    sygus: t.Dict[str, SygusProblem]
    output: t.Dict[str, CompletionJSON]

    def __init__(
        self,
        directory: Path,
        examples: t.Optional[t.Dict[str, t.List[ExampleTuple]]] = None,
    ):
        self.problems = {}
        self.comments = {}
        self.sygus = {}
        self.output = {}

        print(f"loading benchmarks from {directory}")
        for file in directory.iterdir():
            print(file.name)
            with open(file, "r") as f:
                contents = file.read_text()
                self.problems[file.name] = sexp.loads(contents)
                self.comments[file.name] = [
                    line for line in contents.split("\n") if is_comment(line)
                ]

        for filename, sexps in self.problems.items():
            if examples is not None:
                problem_examples = examples[filename]
                self.sygus[filename] = SygusProblem.from_sexps_with_examples(
                    sexps, self.comments[filename], problem_examples
                )
            else:
                self.sygus[filename] = SygusProblem.from_sexps(
                    sexps, self.comments[filename]
                )

    @classmethod
    def read_from_file(
        cls,
        filename: Path,
        directory: Path,
        examples: t.Optional[t.Dict[str, t.List[ExampleTuple]]] = None,
    ):
        with open(filename, "r") as f:
            output = json.load(f)
            benchmark = cls(directory, examples)
            benchmark.output = output
            return benchmark

    def reset_output(self):
        self.output = {}

    def sample_solutions(
        self,
        model: ModelName = "gpt-4",
        n: int = 10,
        filename_of_interest: t.Optional[str] = None,
    ) -> t.Dict:
        for filename, problem in self.sygus.items():
            if filename in self.output:
                continue
            if filename_of_interest is not None and filename != filename_of_interest:
                continue
            try:
                print(f"Sampling completions for {filename}")
                completions, time_diff_ms = sample_gpt_solutions(
                    problem, n=n, model=model
                )
                self.output[filename] = {
                    "completions": completions,
                    "time_diff_ms": time_diff_ms,
                }
            except Exception as e:
                print(f"Error generating completions for {filename}: {e}")
                print(traceback.format_exc())
                continue
        return self.update_solutions()

    def update_solutions(self):
        for filename, problem in self.sygus.items():
            if filename not in self.output:
                continue
            completions = self.output[filename]["completions"]
            try:
                solutions = [
                    problem.completion_to_function_definition(c) for c in completions
                ]
                pprint(solutions)
                self.output[filename]["solutions"] = solutions
            except Exception as e:
                print(f"Error parsing solution for {filename}: {e}")
                print(traceback.format_exc())
                continue
        return self.output

    def log_outputs(self):
        for filename, output in self.output.items():
            print(f"Output for {filename}:")
            pprint(output["solutions"])

    def clear_outputs(self, filenames: t.List[str]):
        for filename in filenames:
            if filename in self.output:
                del self.output[filename]

    def write(self, output_file: Path):
        with open(output_file, "w") as f:
            json.dump(self.output, f, indent=2)


# region SEXP PARSING


SExpItem = t.Union[sexp.Symbol, str, int]
SExp = t.Union[SExpItem, t.List["SExp"]]


def is_function_definition(sexp: t.Any) -> bool:
    return isinstance(sexp[0], Symbol) and sexp[0].value() == "define-fun"


def get_function_definitions(sexps: t.List[t.Any]) -> t.List[t.Any]:
    return [sexp for sexp in sexps if is_function_definition(sexp)]


def is_synth_fun(sexp) -> bool:
    ans = isinstance(sexp[0], Symbol) and sexp[0].value() == "synth-fun"
    return ans


def get_synth_fun(sexps: t.List[t.Any]) -> t.Optional[t.Any]:
    for sexp in sexps:
        if is_synth_fun(sexp):
            return sexp
    return None


def get_signature(synth_fun: t.Any) -> t.Tuple[str, t.List[t.Any], t.Any]:
    def get_type(type_sexp: t.Any) -> t.Any:
        if isinstance(type_sexp, list):
            return f"({type_sexp[0]} {type_sexp[1]})"
        else:
            return type_sexp.value()

    name = synth_fun[1].value()
    args = [(arg[0].value(), get_type(arg[1])) for arg in synth_fun[2]]
    ret_type = get_type(synth_fun[3])
    return (name, args, ret_type)


def is_constraint(sexp) -> bool:
    return isinstance(sexp[0], Symbol) and sexp[0].value() == "constraint"


def get_constraints(sexps: t.List[t.Any]) -> t.List[t.Any]:
    return [sexp for sexp in sexps if is_constraint(sexp)]


def is_define_fun(sexp) -> bool:
    return isinstance(sexp[0], Symbol) and sexp[0].value() == "define-fun"


def get_define_fun_pieces(sexp: SExp):
    if not is_define_fun(sexp):
        raise ValueError(f"Expected a define-fun sexp, got {sexp}")

    name = sexp[1]
    args = sexp[2]
    ret_type = sexp[3]
    body = sexp[4]
    return name, args, ret_type, body


def is_comment(line: str) -> bool:
    return line.strip().startswith(";")


def constraint_to_io(sexp) -> t.Tuple[t.List[str], str]:
    assert is_constraint(sexp)
    eq_exp = sexp[1]
    f_exp = eq_exp[1]
    output_exp = eq_exp[2]
    f_args_exp = f_exp[1:]
    # assert len(f_args_exp) == num_args
    return ([str(arg) for arg in f_args_exp], output_exp)


def add_sygus_prefix(completion: str, prefix: str) -> str:
    parsed_completion = parse_and_repair(completion)
    if parsed_completion is not None:
        parsed_completion = parsed_completion[0]
        if is_define_fun(parsed_completion):
            _, _, _, body = get_define_fun_pieces(parsed_completion)
            parsed_completion = body
        completion = sexp.dumps(parsed_completion)

    ans = f"{prefix}\n{completion}"

    parsed_completion = parse_and_repair(ans)
    if parsed_completion is None:
        return None

    parsed_completion = parsed_completion[0]
    if not is_define_fun(parsed_completion):
        return None

    return sexp.dumps(parsed_completion)


def add_closing_bracket(completion: str) -> str:
    return completion + ")"


def remove_closing_bracket(completion: str) -> str:
    return completion[:-1]


def parse_and_repair(completion: str) -> t.Optional[SExp]:
    try:
        parsed: SExp = sexp.loads(completion)
        return parsed
    except Exception as e:
        if "Not enough closing brackets." in str(e):
            return parse_and_repair(add_closing_bracket(completion))
        if "Too many closing brackets." in str(e):
            return parse_and_repair(remove_closing_bracket(completion))
        else:
            print(f"Error parsing completion: {e}")
            print(traceback.format_exc())
            return None


# endregion SEXP PARSING
