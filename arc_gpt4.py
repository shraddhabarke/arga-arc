import openai
import os, json, copy
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from collections import Counter
import operator
import numpy as np
import ast
openai.api_key = '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'

def differing_rows(matrix1, matrix2):
    differences = []
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same dimensions.")
    for i in range(matrix1.shape[0]):
        if not np.array_equal(matrix1[i], matrix2[i]):
            differences.append((i))
    return differences

def get_completion(prompt, model="gpt-4"):
    messages = [#{"role": "system", "content": preamble}, 
                {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0 # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def numColor(prompt):
    cvals  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "teal", "brown"]
    zipdict = dict(zip(cvals, colors))
    for key, val in zipdict.items():
        prompt = prompt.replace(str(key), val)
    return prompt

def colorNum(prompt):
    cvals  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "teal", "brown"]
    zipdict = dict(zip(colors, cvals))
    for key, val in zipdict.items():
        prompt = prompt.replace(str(key), str(val))
    return prompt

def plot_2d_grid(data):
    cvals  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    fig, axs = plt.subplots(1, 3, figsize=(5, len(data['test']) * 3))
    axs[0].set_title('Test Input')
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].imshow(np.array(data['test'][0]['input']), cmap=cmap, vmin=0, vmax=9)
    axs[1].set_title('Test Output')
    axs[1].set_xticks([]); axs[1].set_yticks([])
    axs[1].imshow(np.array(data['test'][0]['output']), cmap=cmap, vmin=0, vmax=9)
    if data['gpt_output'] is not None:
        axs[2].set_title('GPT Output')
        axs[2].set_xticks([]); axs[2].set_yticks([])
        axs[2].imshow(np.array(data['gpt_output']), cmap=cmap, vmin=0, vmax=9) 
    else:
        axs[2].axis('off')

    fig, axs = plt.subplots(len(data['train']), 2, figsize=(5, len(data['train']) * 3))
    for i, example in enumerate(data['train']):
        axs[i, 0].set_title(f'Training Input {i}')
        axs[i, 0].set_xticks([]); axs[i, 0].set_yticks([])
        axs[i, 0].imshow(np.array(example['input']), cmap=cmap, vmin=0, vmax=9)
        axs[i, 1].set_title(f'Training Output {i}')
        axs[i, 1].set_xticks([]); axs[i, 1].set_yticks([])
        axs[i, 1].imshow(np.array(example['output']), cmap=cmap, vmin=0, vmax=9)
    plt.tight_layout()
    plt.show()

task_preamble = "<TASK>: I want you to act like an expert in pattern recognition and programming. You are given a series of " +\
"input and output 2D arrays, representing a 2D grid with values from 0-9. The values represent different colors, do not perform arithmetic operations " +\
"on them. The grids consists of objects (same or differently colored) which are continuous squares connected horizontally, vertically and/or diagonally. " +\
"Objects in the training input grids are manipulated using a relation to generate the output grids. " +\
"Input/output pairs may not reflect all possibilities, you are to infer the simplest possible relation making use of symmetry and invariance as much as possible."

def load_json_data(folder):
    json_files = [per_json for per_json in os.listdir(folder) if per_json.endswith('.json')]
    data = {}
    for file in json_files:
        with open(os.path.join(folder, file)) as json_file:
            data[file] = json.load(json_file)
    return data

folder = 'dataset'
json_data = load_json_data(folder)

# Formatting Grids: Train Tasks
def generate_train_grid(task_id):
    grids = ""
    json_task = copy.deepcopy(json_data[task_id +'.json'])
    for input_example in json_task['train']:
        grids += "Training Input Grid: " + numColor(str(input_example['input'])) + "\n"
        grids += "Training Output Grid: " + numColor(str(input_example['output'])) + "\n"
    grids += "Test Input Grid: " + numColor(str(json_task['test'][0]['input'])) + "\n"
    grids += "Test Output Grid: to_be_filled"
    outputgrid = numColor(str(json_task['test'][0]['output']))
    return grids, outputgrid

# Formatting Grids: Test Tasks
def generate_test_grid(task_id):
    grids = "<TRAINING GRIDS>:\n"
    json_task = copy.deepcopy(json_data[task_id +'.json'])
    goldsols = []
    for input_example in json_task['train']:
        grids += "Training Input Grid: " + str(input_example['input']) + "\n"
        grids += "Training Output Grid: " + str(input_example['output']) + "\n"
        goldsols.append(input_example['output'])
    #grids += "Test Input Grid: " + str(json_task['test'][0]['input']) + "\n"
    #grids += "Test Output Grid: to_be_filled"
    goldsol = str(json_task['test'][0]['output'])
    return grids, goldsols
# ========================================================================================================================================
# Attempt2: ARGA DSL

arga_prompt = "Each of the input-output relation can be expressed with one or more functions chained together. You are to come up with a program " +\
"composing the functions provided below:\n" +\
"Functions in the format: \n" +\
"{function_name(params)}: {Function description}\n" +\
"- update_color(node, color): update node color to given color.\n" +\
"- add_border(node, border_color): add a border with thickness 1 and border_color around the given node.\n" +\
"- hollow_rectangle(node, fill_color): hollowing the rectangle containing the given node with the given color.\n" +\
"- move_node_max(node, direction: Direction) : move node in a given direction until it hits another node or the edge of the image.\n" +\
"- extend_node(node, direction: Direction, overlap: bool = False): extend node in a given direction, if overlap is true, extend node even if it overlaps with another node. If overlap is false, stop extending before it overlaps with another node.\n\n" +\
"Each transformation is applied only on a subset of objects selected by a filter. The filters can be:\n" +\
"- filter_by_color(node, color: int, exclude: bool = False): returns true if node has given color. If exclude, returns true if node does not have given color.\n" +\
"- filter_by_size(node, size, exclude: bool = False): returns true if node has size equal to given size. If exclude, returns true if node does not have size equal to given size.\n" +\
"- filter_by_degree(node, size, exclude: bool = False): returns true if node has degree equal to given degree. If exclude, returns true if node does not have degree equal to given degree.\n" +\
"- filter_by_neighbor_size(node, size, exclude: bool = False): returns true if node has a neighbor of a given size. If exclude, returns true if node does not have a neighbor of a given size.\n" +\
"- param_bind_neighbor_by_size(node, size, exclude: bool = False): returns the neighbor of node satisfying given size filter.\n"

arga_prompt_abstraction = "Each of the input-output relation can be expressed with one or more functions chained together. You are to come up with a program " +\
"composing the functions provided below:\n" +\
"Functions in the format: \n" +\
"{function_name(params)}: {Function description}\n" +\
"- update_color(node, color): update node color to given color.\n" +\
"- add_border(node, border_color): add a border with thickness 1 and border_color around the given node.\n" +\
"- hollow_rectangle(node, fill_color): hollowing the rectangle containing the given node with the given color.\n" +\
"- move_node_max(node, direction: Direction) : move node in a given direction until it hits another node or the edge of the image.\n" +\
"- extend_node(node, direction: Direction, overlap: bool = False): extend node in a given direction, if overlap is true, extend node even if it overlaps with another node. If overlap is false, stop extending before it overlaps with another node.\n\n" +\
"Each transformation is applied only on a subset of objects selected by a filter. The filters can be:\n" +\
"- filter_by_color(node, color: int, exclude: bool = False): returns true if node has given color. If exclude, returns true if node does not have given color.\n" +\
"- filter_by_size(node, size, exclude: bool = False): returns true if node has size equal to given size. If exclude, returns true if node does not have size equal to given size.\n" +\
"- filter_by_degree(node, size, exclude: bool = False): returns true if node has degree equal to given degree. If exclude, returns true if node does not have degree equal to given degree.\n" +\
"- filter_by_neighbor_size(node, size, exclude: bool = False): returns true if node has a neighbor of a given size. If exclude, returns true if node does not have a neighbor of a given size.\n" +\
"- param_bind_neighbor_by_size(node, size, exclude: bool = False): returns the neighbor of node satisfying given size filter.\n\n"

test_format_prompt = "Your task is to perform the following actions:\n" +\
"1. Generate a program encoded as a sequence or nested sequence of filter-transformation pairs in the following format by filling in the holes 'to_be_filled':\n" +\
"[{'filters': ['to_be_filled'], 'filter_params': [{'to_be_filled', 'exclude': 'to_be_filled'}], 'transformation': ['to_be_filled'], 'transformation_params': [{'to_be_filled'}]}].\n" +\
"2. Mark the components that you are least confident about in the previously generated program.\n\n" +\
"<SOLUTION>:\n"

# Test Prompt
def test_prompt(task_id):
    formatted_grid = generate_test_grid(task_id)[0]
    gold_sol = generate_test_grid(task_id)[1]
    prompt = f"""{task_preamble}\n{arga_prompt}\n{formatted_grid}\n{test_format_prompt2}"""
    return prompt, gold_sol

test = ['4258a5f9', 'bb43febb', 'ae3edfdc', '7f4411dc', '08ed6ac7',
        '00d62c1b', 'ddf7fa4f', 'd43fd935', 'd2abd087', '3906de3d']

for test_id in test:
    print(test_id)
    print(test_prompt(test_id)[0])
    print(test_prompt(test_id)[1])
    #response = get_completion(test_prompt(test_id)[0])
    #print(response)

# "nbvcg - get_non_background_vertical_connected_components_graph" +\
# "nbccg - get_non_black_components_graph" +\ 
# "ccgbr - get_connected_components_graph_background_removed" +\
# "ccg - get_connected_components_graph" +\
# "lrg - get_largest_rectangle_graph" +\