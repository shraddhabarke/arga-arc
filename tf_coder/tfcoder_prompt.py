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

tfoperators = "\nTensorFlow functions to use:\n---------------------\ntf.abs(x)\ntf.add(x, y)\ntf.add_n(inputs)\ntf.argmax(input, axis)\ntf.argmin(input, axis)\n"+\
"tf.argsort(values, axis, stable=True)\ntf.argsort(values, axis, direction='DESCENDING', stable=True)\ntf.boolean_mask(tensor, mask)\ntf.broadcast_to(input, shape)\n"+\
"tf.cast(x, dtype)\ntf.clip_by_value(t, clip_value_min, clip_value_max)\ntf.concat(values, axis)\ntf.constant(value)\ntf.constant(value, dtype)\ntf.divide(x, y)"+\
"tf.equal(x, y)\ntf.exp(x)\ntf.expand_dims(input, axis)\ntf.eye(num_rows)\ntf.eye(num_rows, num_columns)\ntf.eye(num_rows, dtype)\ntf.fill(dims, value)"+\
"tf.gather(params, indices)\ntf.gather(params, indices, axis, batch_dims)\ntf.gather_nd(params, indices)\ntf.gather_nd(params, indices, batch_dims)\ntf.greater(x, y)"+\
"tf.greater_equal(x, y)\ntf.math.bincount(arr)\ntf.math.ceil(x)\ntf.math.count_nonzero(input)\ntf.math.count_nonzero(input, axis)\ntf.math.cumsum(x, axis)"+\
"tf.math.cumsum(x, axis, exclusive=True)\ntf.math.divide_no_nan(x, y)\ntf.math.floor(x)\ntf.math.log(x)\ntf.math.logical_and(x, y)\ntf.math.logical_not(x)"+\
"tf.math.logical_or(x, y)\ntf.math.logical_xor(x, y)\ntf.math.negative(x)\ntf.math.reciprocal(x)\ntf.math.reciprocal_no_nan(x)\ntf.math.segment_max(data, segment_ids)"+\
"tf.math.segment_mean(data, segment_ids)\ntf.math.segment_min(data, segment_ids)\ntf.math.segment_prod(data, segment_ids)\ntf.math.segment_sum(data, segment_ids)"+\
"tf.math.squared_difference(x, y)\ntf.math.top_k(input, k)\ntf.math.unsorted_segment_max(data, segment_ids, num_segments)\ntf.math.unsorted_segment_mean(data, segment_ids, num_segments)"+\
"tf.math.unsorted_segment_min(data, segment_ids, num_segments)\ntf.math.unsorted_segment_prod(data, segment_ids, num_segments)\ntf.math.unsorted_segment_sum(data, segment_ids, num_segments)"+\
"tf.matmul(a, b)\ntf.maximum(x, y)\ntf.minimum(x, y)\ntf.multiply(x, y)\ntf.not_equal(x, y)\ntf.one_hot(indices, depth)\ntf.ones(shape)\ntf.ones_like(input)"+\
"tf.pad(tensor, paddings, mode='CONSTANT')\ntf.pad(tensor, paddings, mode='CONSTANT', constant_values)\ntf.pad(tensor, paddings, mode='REFLECT')"+\
"tf.pad(tensor, paddings, mode='SYMMETRIC')\ntf.range(start)\ntf.range(start, limit, delta)\ntf.reduce_any(input_tensor, axis)\ntf.reduce_all(input_tensor, axis)"+\
"tf.reduce_max(input_tensor)\ntf.reduce_max(input_tensor, axis)\ntf.reduce_mean(input_tensor)"+\
"tf.reduce_mean(input_tensor, axis)\ntf.reduce_min(input_tensor)\ntf.reduce_min(input_tensor, axis)"+\
"tf.reduce_prod(input_tensor, axis)\ntf.reduce_sum(input_tensor)\ntf.reduce_sum(input_tensor, axis)"+\
"tf.repeat(input, repeats)\ntf.repeat(input, repeats, axis)\ntf.reshape(tensor, shape)"+\
"tf.reverse(tensor, axis)\ntf.roll(input, shift, axis)\ntf.round(x)\ntf.scatter_nd(indices, updates, shape)"+\
"tf.searchsorted(sorted_sequence, values, side='left')\ntf.searchsorted(sorted_sequence, values, side='right')"+\
"tf.sequence_mask(lengths)\ntf.sequence_mask(lengths, maxlen)\ntf.shape(input)\ntf.sign(x)"+\
"tf.sort(values, axis)\ntf.sort(values, axis, direction='DESCENDING')\ntf.sqrt(x)"+\
"tf.square(x)\ntf.squeeze(input)\ntf.squeeze(input, axis)\ntf.stack(values, axis)\ntf.subtract(x, y)"+\
"tf.tensor_scatter_nd_update(tensor, indices, updates)\ntf.tensordot(a, b, axes)\ntf.tile(input, multiples)"+\
"tf.transpose(a)\ntf.transpose(a, perm)\ntf.unique_with_counts(x)\ntf.unstack(value, axis)"+\
"tf.where(condition)\ntf.where(condition, x, y)\ntf.zeros(shape)\ntf.zeros_like(input)"+\
"\n\nSparseTensor functions:\n-----------------------\ntf.SparseTensor(indices, values, dense_shape)\ntf.sparse.add(a, b)"+\
"tf.sparse.concat(axis, sp_inputs)\ntf.sparse.expand_dims(sp_input, axis)\ntf.sparse.from_dense(tensor)\ntf.sparse.maximum(sp_a, sp_b)"+\
"tf.sparse.minimum(sp_a, sp_b)\ntf.sparse.reduce_max(sp_input, axis, output_is_sparse)\ntf.sparse.reduce_sum(sp_input, axis, output_is_sparse)"+\
"tf.sparse.reset_shape(sp_input)\ntf.sparse.reshape(sp_input, shape)\ntf.sparse.retain(sp_input, to_retain)\ntf.sparse.slice(sp_input, start, size)"+\
"tf.sparse.split(sp_input, num_split, axis)\ntf.sparse.to_dense(sp_input)\ntf.sparse.to_dense(sp_input, default_value)"+\
"tf.sparse.to_indicator(sp_input, vocab_size)\ntf.sparse.transpose(sp_input)\ntf.sparse.transpose(sp_input, perm)"

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
    if start == -1:
        return ""
    end = output[start+10:].find("```")
    if end == -1:
        end = len(output[start+10:])
    
    result = []
    for l in output[start+10:start+10+end].split("\n"):
        if not l.startswith("#") and not l.startswith("import tensorflow") and not l.startswith("print"): # ignore comments
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
    prompt = f"{task_preamble}\n\n{tfoperators}\n\nTask Description: {task['description']}\nExamples ---> {examples_str}\nPROGRAM:"
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
        print("prompt:", prompt)
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