# region PREAMBLE
import os
import sys
from pathlib import Path

import octoai.client

CURRENT_DIRECTORY = Path(os.getcwd())
ROOT_DIRECTORY = (CURRENT_DIRECTORY / "..").absolute().resolve()

print(f"Current directory: {CURRENT_DIRECTORY}")
print(f"Root directory: {ROOT_DIRECTORY}")

sys.path.append(str(ROOT_DIRECTORY))
# endregion PREAMBLE
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
import octoai
import builtins
from tqdm import tqdm
import logging
import click

LOGGER = logging.getLogger(__name__)

ModelName = t.Literal[
    "gpt-4",
    "gpt-3.5-turbo",
    "Phind/Phind-CodeLlama-34B-v2",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
]
OPENAI_MODEL_NAMES = ["gpt-4", "gpt-3.5-turbo"]
TOGETHER_MODEL_NAMES = [
    "deepseek-ai/deepseek-coder-33b-instruct",
    "Phind/Phind-CodeLlama-34B-v2",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
]
OCTO_MODEL_NAMES = ["codellama-34b-instruct"]
MODEL_NAMES = OPENAI_MODEL_NAMES + TOGETHER_MODEL_NAMES + OCTO_MODEL_NAMES

OPENAI = OpenAI(
    organization=CONFIG.OPENAI_ORGANIZATION, api_key=CONFIG.OPENAI_SECRET_KEY
)
client = OpenAI(
    organization=CONFIG.OPENAI_ORGANIZATION, api_key=CONFIG.OPENAI_SECRET_KEY
)

TOGETHER = OpenAI(api_key=CONFIG.TOGETHER_SECRET_KEY, base_url=CONFIG.TOGETHER_BASE_URL)

OCTO = octoai.client.OctoAI(api_key=CONFIG.OCTO_SECRET_KEY)

ExampleTuple = t.Tuple[t.List[str], str]

BenchmarkName = t.Literal["larger-string", "string"]  # , "circuit", "hackers-delight"]
BENCHMARK_NAMES = ["larger-string", "string"]  # , "circuit", "hackers-delight"]

BENCHMARKS_DIRECTORY = CONFIG.ROOT_DIR / "sygus/Probe/src/test/benchmarks"
BENCHMARK_DIRECTORIES: dict[BenchmarkName, Path] = {
    "larger-string": BENCHMARKS_DIRECTORY / "larger-grammar",
    "string": BENCHMARKS_DIRECTORY / "string",
    # "circuit": BENCHMARKS_DIRECTORY / "circuit/test",
    # "hackers-delight": BENCHMARKS_DIRECTORY / "hackers-delight",
}
EXAMPLE_FILES: dict[BenchmarkName, t.Optional[Path]] = {
    "larger-string": None,
    "string": None,
    # "circuit": CONFIG.ROOT_DIR / "sygus/io-results-circuit.json",
    # "hackers-delight": CONFIG.ROOT_DIR / "sygus/io-results-bitvec.json",
}

# region CLI COMMANDS


@click.group()
def cli():
    pass


@cli.command(help="sample completions for all benchmarks")
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
def sample(model: ModelName, num_samples: int):
    setup_logging(model)
    for benchmark in reversed(BENCHMARK_NAMES):
        sample_benchmark(benchmark, model, num_samples)


@cli.command(help="merge 2 sets of samples")
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
    merged: t.Dict[str, CompletionJSON] = {}
    for f in file:
        with open(f, "r") as file:
            data: t.Dict[str, CompletionJSON] = json.load(file)
            for key, completions in data.items():
                if key not in merged:
                    merged[key] = {
                        "completions": [],
                        "solutions": [],
                        "constants": [],
                        "all_constants": [],
                        "time_diff_ms": 0,
                    }

                merged[key]["completions"].extend(completions["completions"])
                if "solutions" in completions:
                    merged[key]["solutions"].extend(completions["solutions"])
                if "constants" in completions:
                    merged[key]["constants"].extend(completions["constants"])
                if "all_constants" in completions:
                    merged[key]["all_constants"] = list(
                        set(merged[key]["all_constants"] + completions["all_constants"])
                    )
                merged[key]["time_diff_ms"] += completions["time_diff_ms"]

    with open(output, "w") as file:
        json.dump(merged, file, indent=2)


@cli.command(help="print statistics for all benchmarks")
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODEL_NAMES),
    required=True,
    help="The model to use for sampling completions",
)
def stats(model):
    for benchmark in BENCHMARK_NAMES:
        output_file = compute_output_file(benchmark, model)
        benchmark: SygusBenchmark = SygusBenchmark.read_from_file(
            benchmark, output_file, BENCHMARK_DIRECTORIES[benchmark]
        )
        benchmark.print_statistics()


@cli.command(help="fixup completions for all benchmarks")
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODEL_NAMES),
    required=True,
    help="The model to use for sampling completions",
)
def fixup(model):
    setup_logging(model)
    for benchmark in BENCHMARK_NAMES:
        fixup_benchmark(benchmark, model)


@cli.command(help="parse constants for all benchmarks")
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODEL_NAMES),
    required=True,
    help="The model to use for sampling completions",
)
def constants(model):
    setup_logging(model)
    for benchmark in BENCHMARK_NAMES:
        parse_constants_for(benchmark, model)


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
    for benchmark in BENCHMARK_NAMES:
        for n in num_samples:
            resample_benchmark(benchmark, model, n)


def main():
    original_print = builtins.print
    builtins.print = tqdm_print

    cli()


def compute_output_file(
    benchmark: BenchmarkName, model: ModelName, n: t.Optional[int] = None
):
    return (
        CONFIG.ROOT_DIR
        / "sygus"
        / f"{benchmark}-completions.{model.replace('/', '__')}{'.' if n is None else '.' + str(n) + '.'}json"
    )


def resample_benchmark(benchmark: BenchmarkName, model: ModelName, n: int):
    output_file = compute_output_file(benchmark, model)
    example_file = EXAMPLE_FILES[benchmark]
    if example_file is not None:
        examples = get_examples(example_file)
    else:
        examples = None

    if output_file.exists():
        print(f"Reading from {output_file}")
        benchmark_obj: SygusBenchmark = SygusBenchmark.read_from_file(
            benchmark,
            output_file,
            BENCHMARK_DIRECTORIES[benchmark],
            examples,
        )
    else:
        print(f"No completions found for {benchmark}")
        return

    print(f"Resampling {n} completions for {benchmark} using {model}")

    resampled = SygusBenchmark(
        benchmark,
        BENCHMARK_DIRECTORIES[benchmark],
        examples,
    )
    for filename, output in benchmark_obj.output.items():
        non_null_solution_idxs = [
            i for i, s in enumerate(output["solutions"]) if s is not None
        ]
        non_null_solutions = [output["solutions"][i] for i in non_null_solution_idxs]
        non_null_completions = [
            output["completions"][i] for i in non_null_solution_idxs
        ]
        non_null_constants = [output["constants"][i] for i in non_null_solution_idxs]

        if len(non_null_solutions) < n:
            print(
                f"for {filename}, Expected at least {n} solutions to resample, got {len(non_null_solutions)}"
            )
            resampled.output[filename] = None
            continue

        idxs = random.sample(range(len(non_null_solutions)), n)
        resampled_output: CompletionJSON = {
            "completions": [non_null_completions[i] for i in idxs],
            "solutions": [non_null_solutions[i] for i in idxs],
            "constants": [non_null_constants[i] for i in idxs],
            "all_constants": list(set(sum([non_null_constants[i] for i in idxs], []))),
            "time_diff_ms": output["time_diff_ms"],
        }

        resampled.output[filename] = resampled_output

    resampled_output_file = compute_output_file(benchmark, model, n)
    resampled.write(resampled_output_file)


def sample_benchmark(benchmark: BenchmarkName, model: ModelName, n: int):
    print(f"Sampling {n} completions for {benchmark} using {model}")
    example_file = EXAMPLE_FILES[benchmark]
    output_file = compute_output_file(benchmark, model)

    if example_file is not None:
        examples = get_examples(example_file)
    else:
        examples = None

    if output_file.exists():
        print(f"Reading from {output_file}")
        benchmark: SygusBenchmark = SygusBenchmark.read_from_file(
            benchmark, output_file, BENCHMARK_DIRECTORIES[benchmark], examples=examples
        )
    else:
        benchmark: SygusBenchmark = SygusBenchmark(
            benchmark, BENCHMARK_DIRECTORIES[benchmark], examples=examples
        )

    print(f"Sampling {n} completions for {benchmark.name} using {model}")
    benchmark.sample_solutions(model=model, n=n, output_file=output_file)


def fixup_benchmark(benchmark: BenchmarkName, model: ModelName):
    output_file = compute_output_file(benchmark, model)

    if output_file.exists():
        print(f"Reading from {output_file}")
        benchmark: SygusBenchmark = SygusBenchmark.read_from_file(
            benchmark, output_file, BENCHMARK_DIRECTORIES[benchmark]
        )
    else:
        print(f"No completions found for {benchmark}")
        return

    print(f"Fixing completions for {benchmark.name} using {model}")
    benchmark.fixup_solutions()
    benchmark.write(output_file)


def parse_constants_for(benchmark: BenchmarkName, model: ModelName):
    output_file = compute_output_file(benchmark, model)

    if output_file.exists():
        print(f"Reading from {output_file}")
        benchmark: SygusBenchmark = SygusBenchmark.read_from_file(
            benchmark, output_file, BENCHMARK_DIRECTORIES[benchmark]
        )
    else:
        print(f"No completions found for {benchmark}")
        return

    print(f"Parsing constants for {benchmark.name} using {model}")
    benchmark.parse_constants()
    benchmark.write(output_file)


def get_examples(example_file: Path):
    examples_json = json.loads(example_file.read_text())
    ans = {}
    for filename, examples in examples_json.items():

        ans[filename] = [(example["inputs"], example["output"]) for example in examples]
        ans[filename] = random.sample(ans[filename], min(10, len(ans[filename])))

    return ans


def setup_logging(name: str):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=CONFIG.ROOT_DIR
        / "sygus"
        / f"utils_{name.replace('/', '__')}_{now}.log",
        filemode="w",
    )


def tqdm_print(*args, **kwargs):

    # if no arguments are passed, write the empty string
    if not args:
        args = [""]
    tqdm.write(*args, **kwargs)
    LOGGER.info(" ".join(args))


# endregion CLI COMMANDS


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
        examples = (
            random.sample(self.examples, 5) if len(self.examples) > 5 else self.examples
        )
        for args, output in examples:
            EXAMPLES += f"{' , '.join(args)} -> {output}\n"

        ans = f"""[GRAMMAR]
{self.synth_fun}

"""
        if self.natural_language_spec.strip() != "":
            ans += f"""[NATURAL LANGUAGE SPECIFICATION]
{self.natural_language_spec}

"""

        ans += f"""[EXAMPLES]
{EXAMPLES}

[SOLUTION]
{self.function_definition_prefix}"""

        return ans


SYSTEM_PROMPT = """You are a coding assistant. Be precise and terse.
You will be given a SyGuS grammar, a natural language specification, and a set of input-output examples.
Your task is to complete the provided function definition with an implementation that is correct according to the grammar, specification, and examples.
Your answer should be as short as possible while still being correct.
Make sure that your answer is a valid s-expression."""


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
) -> t.Tuple[t.Optional[t.List[str]], int]:
    if model in OPENAI_MODEL_NAMES:
        client = OPENAI
    elif model in TOGETHER_MODEL_NAMES:
        client = TOGETHER
    elif model in OCTO_MODEL_NAMES:
        client = OCTO
    else:
        raise ValueError(f"Invalid model: {model}")

    start_time = datetime.now()
    response = get_chat_completion_retrying(
        client,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.user_message},
        ],
        n=n,
        temperature=0.5,
        model=model,
        max_tokens=200,
    )
    end_time = datetime.now()
    time_diff_ms = (end_time - start_time).microseconds / 1000

    if response is None:
        return None, time_diff_ms

    if model in TOGETHER_MODEL_NAMES:
        # sleep for 1s to cover trial rate limit
        time.sleep(1)

    return [choice.message.content for choice in response.choices], time_diff_ms


MAX_RETRIES = 3


def get_chat_completion_retrying(client, *args, **kwargs):
    for i in range(MAX_RETRIES):
        try:
            print("getting chat completion")
            return client.chat.completions.create(*args, **kwargs)
        except Exception as e:
            print(f"Error getting completion: {e}")
            print(traceback.format_exc())
            continue
    return None


class CompletionJSON(t.TypedDict):
    completions: t.List[str]
    solutions: t.List[str]
    constants: t.List[t.List[t.Union[int, str]]]
    all_constants: t.List[t.Union[int, str]]
    time_ms: float


class SygusBenchmark:
    name: BenchmarkName
    problems: t.Dict
    comments: t.Dict
    sygus: t.Dict[str, SygusProblem]
    output: t.Dict[str, CompletionJSON]

    def __init__(
        self,
        name: BenchmarkName,
        directory: Path,
        examples: t.Optional[t.Dict[str, t.List[ExampleTuple]]] = None,
    ):
        self.name = name
        self.problems = {}
        self.comments = {}
        self.sygus = {}
        self.output = {}

        for file in directory.iterdir():
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
        name: BenchmarkName,
        filename: Path,
        directory: Path,
        examples: t.Optional[t.Dict[str, t.List[ExampleTuple]]] = None,
    ):
        assert filename.exists(), f"{filename} does not exist"
        with open(filename, "r") as f:
            output = json.load(f)
            benchmark = cls(name, directory, examples)
            benchmark.output = output
            return benchmark

    def print_statistics(self):
        num_items = len(self.sygus)

        num_completions = 0
        for output in self.output.values():
            if "completions" in output and output["completions"] is not None:
                num_completions += len(output["completions"])

        num_solutions = 0
        for output in self.output.values():
            if "solutions" in output:
                num_solutions += len([s for s in output["solutions"] if s is not None])

        average_num_completions = num_completions / num_items
        average_num_solutions = num_solutions / num_items

        min_num_completions = min(
            (
                len(output["completions"])
                for output in self.output.values()
                if output["completions"] is not None
            )
        )
        max_num_completions = max(
            (
                len(output["completions"])
                for output in self.output.values()
                if output["completions"] is not None
            )
        )

        print(f"# {self.name}")
        print(f"Total problems: {num_items}")
        print(f"Total completions: {num_completions}")
        print(f"Total parsed solutions: {num_solutions}")
        print(f"Average completions per problem: {average_num_completions}")
        print(f"Minimum completions per problem: {min_num_completions}")
        print(f"Maximum completions per problem: {max_num_completions}")
        print(f"Average solutions per problem: {average_num_solutions}")

        if len([o for o in self.output.values() if "solutions" in o]) == 0:
            return

        minimum_num_solutions = min(
            (
                len([s for s in output["solutions"] if s is not None])
                for output in self.output.values()
                if "solutions" in output
            ),
            default=0,
        )

        median_num_solutions = sorted(
            len([s for s in output["solutions"] if s is not None])
            for output in self.output.values()
            if "solutions" in output
        )[len(self.output) // 2]

        maximum_num_solutions = max(
            (
                len([s for s in output["solutions"] if s is not None])
                for output in self.output.values()
                if "solutions" in output
            ),
            default=0,
        )

        num_solution_deciles = []
        num_solutions_each = [
            len([s for s in output["solutions"] if s is not None])
            for output in self.output.values()
            if "solutions" in output
        ]
        for i in range(10):
            num_solution_deciles.append(
                sorted(num_solutions_each)[len(num_solutions_each) * i // 10]
            )

        for key, output in self.output.items():
            if "solutions" not in output:
                print(f"Problem {key} has no solutions")
                continue
            num_solutions_for_problem = len(
                [s for s in output["solutions"] if s is not None]
            )
            if num_solutions_for_problem < 90:
                print(f"Problem {key} has {num_solutions_for_problem} solutions")

        print(f"Minimum solutions per problem: {minimum_num_solutions}")
        print(f"Maximum solutions per problem: {maximum_num_solutions}")
        print(f"Median solutions per problem: {median_num_solutions}")
        print(f"Deciles of solutions per problem: {num_solution_deciles}")
        print()

    def reset_output(self):
        self.output = {}

    def sample_solutions(
        self,
        model: ModelName = "gpt-4",
        n: int = 10,
        filename_of_interest: t.Optional[str] = None,
        output_file: t.Optional[Path] = None,
    ) -> t.Dict:
        num_items = len(self.sygus)
        for idx, (filename, problem) in tqdm(
            list(enumerate(self.sygus.items())), desc=self.name
        ):
            if (
                filename in self.output
                and "completions" in self.output[filename]
                and self.output[filename]["completions"] is not None
            ):
                print(f"already have output for {filename}")
                continue
            if filename_of_interest is not None and filename != filename_of_interest:
                continue
            try:
                print(f"sampling completions for {filename}")
                completions, time_diff_ms = sample_gpt_solutions(
                    problem, n=n, model=model
                )
                self.output[filename] = {
                    "completions": completions,
                    "time_diff_ms": time_diff_ms,
                }
                if completions is None:
                    print(f"Error generating completions for {filename}")
                    continue

                if output_file is not None:
                    self.write(output_file)
            except Exception as e:
                print(f"Error generating completions for {filename}: {e}")
                print(traceback.format_exc())
                continue

        self.fixup_solutions()
        return self.parse_constants()

    def fixup_solutions(self):
        for filename, problem in tqdm(
            list(self.sygus.items()), desc=f"{self.name}-fixup"
        ):
            if filename not in self.output:
                continue
            completions = self.output[filename]["completions"]
            self.output[filename]["solutions"] = []
            for completion in completions:
                try:
                    parsed = sexp.loads(completion)
                    self.output[filename]["solutions"].append(sexp.dumps(parsed))
                    continue
                except Exception:
                    pass

                extracted = extract_code(completion)
                normalized = None

                if extracted is None:
                    print(f"# Error extracting completion")
                    print(f"## completion")
                    print('"""' + completion + '"""')
                    print()
                    print()
                    self.output[filename]["solutions"].append(None)
                    continue

                try:
                    normalized = normalize_code(
                        extracted, self.sygus[filename].function_definition_prefix
                    )
                    self.output[filename]["solutions"].append(normalized)
                except Exception as e:
                    self.output[filename]["solutions"].append(None)
                    print(f"Error parsing solution for {filename}: {e}")
                    print(traceback.format_exc())
        return self.output

    def parse_constants(self):
        for filename, problem in tqdm(
            list(self.sygus.items()), desc=f"{self.name}-constants"
        ):
            if filename not in self.output:
                continue
            if "solutions" not in self.output[filename]:
                continue
            solutions = self.output[filename]["solutions"]
            try:
                constants = [
                    get_constants(s) if s is not None else None for s in solutions
                ]
                self.output[filename]["constants"] = constants
                self.output[filename]["all_constants"] = list(
                    set(sum((c for c in constants if c is not None), []))
                )
            except Exception as e:
                print(f"Error parsing constants for {filename}: {e}")
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

CODE_BLOCK_REGEX = r"```((?:(?!\n).)*\n)?((?:(?!```).)*)```"


def extract_code_block(completion: str) -> t.Optional[str]:
    match = re.search(CODE_BLOCK_REGEX, completion, re.DOTALL)
    if match:
        group_number = len(match.groups())
        return match.group(group_number).rstrip()
    else:
        return None


def extract_plain_code(completion: str) -> t.Optional[str]:
    return completion.split("\n\n")[0].strip()


def remove_leading_close_paren(completion: str) -> str:
    return (
        completion.strip()[1:]
        if completion.strip().startswith(")")
        else completion.strip()
    )


def extract_code(completion: str) -> t.Optional[str]:
    code_block_result = extract_code_block(completion)

    if code_block_result is not None:
        ans = code_block_result
    else:
        ans = extract_plain_code(completion)

    return remove_leading_close_paren(ans)


def add_definition(code: str, definition: str) -> str:
    if "define-fun" in code:
        return code

    return definition + code


def add_closing_bracket(completion: str) -> str:
    return completion + ")"


def remove_closing_bracket(completion: str) -> str:
    return completion[:-1]


def balance_parens(completion: str) -> t.Optional[str]:
    current_completion = completion.strip()
    for _ in range(10):
        try:
            parsed = sexp.loads(current_completion)
            return sexp.dumps(parsed)
        except Exception as e:
            if "Not enough closing brackets." in str(e):
                current_completion = add_closing_bracket(completion)
            elif "Too many closing brackets." in str(e):
                current_completion = remove_closing_bracket(completion)
            else:
                print(f"Caught unexpected error parsing completion:")
                print(f"{completion}")
                print(traceback.format_exc())
                return None


def normalize_code(code: str, definition: str) -> str:
    return balance_parens(add_definition(code, definition))


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


def get_define_fun_pieces(sexp_v: SExp):
    if not is_define_fun(sexp_v):
        raise ValueError(f"Expected a define-fun sexp, got {sexp_v}")

    name = sexp_v[1]
    args = sexp_v[2]
    ret_type = sexp_v[3]
    body = sexp_v[4]
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
            print(f"Caught unexpected error parsing completion:")
            print(f"{completion}")
            print(traceback.format_exc())
            return None


class recursionlimit:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)


def walk(s: SExp):
    prune = yield s
    if prune is None:
        prune = False
    if prune:
        return

    if isinstance(s, list):
        for i in s:
            yield from walk(i)


def get_constants(solution: str) -> t.List[str]:
    parsed = sexp.loads(solution)
    return list(
        set([node for node in walk(parsed) if type(node) == str or type(node) == int])
    )


# endregion SEXP PARSING

if __name__ == "__main__":
    main()
