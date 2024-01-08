import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# sys.path.append("..")
import defs
import data

def build_prompt(task_id):
    task = data.get_task(task_id)
    path_dict = {
        "preamble": "models/templates/preamble.txt",
        "dsl": "models/templates/dsl.txt",
        "nl_examples": "models/templates/nl_examples.txt",
        "task_preamble": "models/templates/task_preamble.txt",
        "query": "models/templates/query.txt",
    }
    path_dict = {k: os.path.join(defs.PROJECT_ROOT, v) for k, v in path_dict.items()}
    prompt = []
    with open(path_dict["preamble"], "r") as f:
        prompt.append(f.read())
    with open(path_dict["dsl"], "r") as f:
        prompt.append(f.read())
    with open(path_dict["nl_examples"], "r") as f:
        prompt.append(f.read()) 
    with open(path_dict["task_preamble"], "r") as f:
        prompt.append(f.read())
    prompt.append(data.task_description(task_id, color_map="char"))
    with open(path_dict["query"], "r") as f:
        prompt.append(f.read())

    return "\n\n".join(prompt)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python prompt.py <task_id>")
        exit(1)
    task_id = args[0]
    prompt = build_prompt(task_id)
    print(prompt)