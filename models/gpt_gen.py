import os
import sys
import json
import re

import defs

import models.lm
import models.prompt
import dsl.v0_3.parser

def is_parseable(src):
    """
    Returns True if the source code is parseable, False otherwise.
    """
    try:
        grammar_file = "dsl/v0_3/dsl.lark"
        dsl.v0_3.parser.Parser(grammar_file).parse_tree(src)
        return True
    except Exception as e:
        # print(f"Error parsing code: {e}")
        return False

def get_last_code_snippet(response:str) -> str:
    """
    Given a response from the model, we want to extract the last code snippet from it. 
    It could be that the response is bare code, or that it is in Markdown format.
    In the former case, we just return the response, in the latter we extract the last
    code block.
    """
    # If the response is bare code, return it
    if is_parseable(response):
        return response
    # Otherwise, extract the last valid code block
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    code_blocks = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
    for block in code_blocks[::-1]:
        if is_parseable(block):
            return block
    return None

def query_task(task_id, n_responses, output_dir, model):
    """
    Returns a list of n_responses responses to the prompt
    """
    pmpt = models.prompt.build_prompt_v0_3(task_id)
    lm = models.lm.LanguageModel(model=model)
    system_prompt_path = os.path.join(defs.PROJECT_ROOT, "models/templates/system.txt")
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    completions = lm.query(pmpt, n_responses, system_prompt=system_prompt, log=True)
    valid, invalid = [], []
    # split the completions
    for completion in completions:
        code = get_last_code_snippet(completion)
        if code is not None:
            valid.append(code)
        else:
            invalid.append(completion)
            # print(f"Error parsing code: ")
            # print(completion)
    # print the number of correct and incorrect completions
    print(f"Task {task_id}")
    print(f"Valid: {len(valid)}")
    print(f"Invalid: {len(invalid)}")
    # save the completions to files
    timestamp = defs.get_timestamp(micros=False)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{task_id}_valid.txt"), "w") as f:
        f.write("\n\n".join(valid))
    with open(os.path.join(output_dir, f"{task_id}_invalid.txt"), "w") as f:
        f.write("\n\n".join(invalid))
    return completions

def query_task_list(task_ids, n_responses, model):
    # create a directory for the output
    timestamp = defs.get_timestamp(micros=False)
    output_dir = os.path.join(defs.PROJECT_ROOT, "models/logs", f"gens_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # query each task
    for task_id in task_ids:
        query_task(task_id, n_responses, output_dir, model)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: python lm.py <task_list_file> <n_responses>")
        exit(1)
    task_list_file = args[0]
    n_responses = int(args[1])
    with open(task_list_file, "r") as f:
        task_ids = [id for id in f.read().splitlines() if id]

    # model = "gpt-3.5-turbo-0125"
    model = "gpt-4-0125-preview"
    query_task_list(task_ids, n_responses, model)