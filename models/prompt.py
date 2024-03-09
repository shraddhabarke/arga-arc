import os
import sys

import tiktoken

import defs
import data

def read_file(filename):
    with open(filename, "r") as f:
        return f.read()

# def build_prompt(task_id):
#     task = data.get_task(task_id)
#     path_dict = {
#         "preamble": "models/templates/preamble.txt",
#         "dsl": "models/templates/dsl.txt",
#         "nl_examples": "models/templates/nl_examples.txt",
#         "task_preamble": "models/templates/task_preamble.txt",
#         "query": "models/templates/query.txt",
#     }
#     path_dict = {k: os.path.join(defs.PROJECT_ROOT, v) for k, v in path_dict.items()}
#     prompt = []
#     with open(path_dict["preamble"], "r") as f:
#         prompt.append(f.read())
#     with open(path_dict["dsl"], "r") as f:
#         prompt.append(f.read())
#     with open(path_dict["nl_examples"], "r") as f:
#         prompt.append(f.read()) 
#     with open(path_dict["task_preamble"], "r") as f:
#         prompt.append(f.read())
#     prompt.append(data.task_description(task_id, color_map="char"))
#     with open(path_dict["query"], "r") as f:
#         prompt.append(f.read())

#     return "\n\n".join(prompt)

def build_prompt_v0_3(task_id):
    template_dict = {
        "preamble": "models/templates/preamble.txt",
        "dsl_preamble": "models/templates/dsl_preamble.txt",
        "nl_examples": "models/templates/v0_3/nl_examples.txt",
        "task_preamble": "models/templates/task_preamble.txt",
        "query": "models/templates/query.txt",
    }
    prompt = []

    prompt.append(read_file(template_dict["preamble"]))

    prompt.append(read_file(template_dict["dsl_preamble"]))

    dsl_path = "dsl/v0_3/dsl.lark"
    dsl = read_file(dsl_path).strip()
    dsl = "```\n" + dsl + "\n```\n"
    prompt.append(dsl)

    prompt.append(read_file(template_dict["nl_examples"]))

    prompt.append(read_file(template_dict["task_preamble"]))
    
    prompt.append(data.task_description(task_id, print_test=True, color_map="char"))

    prompt.append(read_file(template_dict["query"]))

    return "\n\n".join(prompt)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python prompt.py <task_id>")
        exit(1)
    task_id = args[0]
    # prompt = build_prompt(task_id)
    prompt = build_prompt_v0_3(task_id)
    print(prompt)

    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(prompt)
    print(f"{len(tokens)} tokens")