from openai import OpenAI

client = OpenAI(api_key='')
import os, json, copy
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from collections import Counter
import operator
import numpy as np
import typing, re
from openai import OpenAI

def truncate(completion, include_comments=False):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    if not include_comments:
        terminals = [
            re.compile(r, re.MULTILINE)
            for r in
            [
                '^#',
                "^<",
                re.escape('</code>'),
                "^'''",
                '^"""',
                '\n\n\n',
                '^# Python',
                '^print',
            ]
        ]
    else:
        terminals = [
            re.compile(r, re.MULTILINE)
            for r in
            [
                "^<",
                re.escape('</code>'),
                "^'''",
                '^"""',
                '\n\n\n',
                '^# Python',
                '^print',
            ]
        ]
    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion

def get_code_from_output(output):
    start = output.find("```python")
    print("start:", start)
    if start == -1:
        return ""
    end = output[start+10:].find("```")
    if end == -1:
        end = len(output[start+10:])
    
    result = []
    for l in output[start+10:start+10+end].split("\n"):
        if not l.startswith("#") and not l.startswith("import tensorflow") and not l.startswith("print"): # ignore comments
            print("inside:", l)
            result.append(l)
    return "\n".join(result)

persona = """
You are a coding assistant. Be precise and short. Look at the data in the given data frame and give a Python solution using Tensorflow accordingly. Explain your solution first and explain how this fits the given data. Then present the proposed code at the end in a format like this: 
```python
<python-code>
```
Do only return the Python code at the end!
"""
def get_completion(prompt, model="gpt-4"):
    completion =  client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", 
                   "content": persona}, 
        {
            "role": "user",
            "content": prompt,
        },
    ], 
    temperature=0.5,
    n=10,
    max_tokens=1024,
    stop="```\n")
    return completion

def generate_prompt_for_task(task: dict) -> str:
    """
    Generates a coding task prompt for a given task dictionary loaded from a JSON file.
    """
    task_preamble = ("<TASK>: You are an expert Tensorflow programmer. You are given a task in natural language "
                    "for which you need to generate code using Tensorflow operators.\n"
                    "You are also given input and output examples of the desired behavior. "
                    "Your task is to generate clear and concise Tensorflow code for the examples given below.")
    inputs_str = task["examples"]["inputs"]
    outputs_str = task["examples"]["outputs"]
    try:
        inputs = eval(inputs_str)
        outputs = eval(outputs_str)
    except Exception as e:
        inputs = inputs_str
        outputs = outputs_str

    examples_str = ""
    for inp, out in zip(inputs, outputs):
        examples_str += f"\nInput: {json.dumps(inp)}\nOutput: {json.dumps(out)}\n"

    # Assembling the prompt
    prompt = f"{task_preamble}\n\nTask Description: {task['description']}\nExamples ---> {examples_str}\nPROGRAM:"
    return prompt

def extract_tf_operators(code_snippet):
    pattern = r"tf\.[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"
    return set(re.findall(pattern, code_snippet))

def calculate_tf_operator_coverage_and_count(target_operators, completion_operators):
    """Extend to include all completion operators and mark those used in the target program."""
    completion_operators_count = Counter(completion_operators)
    tf_operators_dict = {op: completion_operators_count[op] for op in completion_operators_count}

    # Calculate coverage based on target program operators found in completions
    covered_operators = set(target_operators).intersection(completion_operators)
    coverage_percentage = len(covered_operators) / len(target_operators) * 100 if target_operators else 0

    return {
        "tf_operators": tf_operators_dict,
        "coverage_percentage": coverage_percentage,
        "total_in_target": len(target_operators),
        "total_covered": len(covered_operators)
    }

def load_and_generate_prompts(file_path: str):
    with open(file_path, 'r') as f:
        tasks = json.load(f)

    # Generate and print a prompt for each task
    task_responses = []
    #start_time = datetime.datetime.now()
    for task in tasks:
        prompt = generate_prompt_for_task(task)
        current_response = get_completion(prompt)
        full_output = [r.message.content for r in current_response.choices]
        completions = [truncate(get_code_from_output(r.message.content)) for r in current_response.choices]

        # Extract TensorFlow operators from completions and target program
        completion_tf_operators = [op for completion in completions for op in extract_tf_operators(completion)]
        target_program = task["target_program"]
        target_tf_operators = extract_tf_operators(target_program)
        # Calculate coverage and count, adjusting for all completion operators
        tf_operator_info = calculate_tf_operator_coverage_and_count(target_tf_operators, completion_tf_operators)

        task_response = {
            "task_id": task.get("name", "unknown"),
            "completions": completions,
            "target-program": task["target_program"],
            "description": task["description"],
            **tf_operator_info  # Includes adjusted tf_operators dictionary
        }
        task_responses.append(task_response)
    
    with open("output_tfcoder.json", "w") as f_out:
        json.dump(task_responses, f_out, indent=4)

load_and_generate_prompts("tfcoder_dataset.json")