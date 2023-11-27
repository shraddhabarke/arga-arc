import openai
import os, json, copy
import numpy as np
from collections import Counter
import operator
import numpy as np
import ast
import data

#openai.api_key = ''

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

task_preamble = "<TASK>: I want you to act like an expert in pattern recognition and programming.\nYou are given a series of " +\
"input and output 2D arrays, representing a 2D grid with different colors. (black color is represented as O)\n" +\
"The grids consist of objects (same or differently colored) which are continuous squares connected horizontally, vertically and/or diagonally.\n" +\
"Objects in the input grids are manipulated using a relation to generate the output grids.\n"
# ========================================================================================================================================
# Attempt2: Abstraction Prompt

abstraction_prompt = "The aim is to parse the grids into a graph of objects with spatial and other relations.\nGraph abstraction allows us to modify groups of objects at once, instead of individual pixels separately.\n" +\
"The following abstractions are supported, and are presented in the format: \n" +\
"{abstraction}: {Abstraction description}\n" +\
"- nbccg (non_black_components_graph) : group of adjacent pixels of the same color in the original graph, excluding background color.\n" +\
"- nbvcg (non_background_vertical_connected_components_graph) : group of vertically adjacent pixels of the same color in the original graph, excluding background color.\n" +\
"- ccgbr (connected_components_graph_background_removed) : a group of adjacent pixels of the same color in the original graph. remove nodes identified as background. background is defined as a node that includes a corner and has the most common color.\n" +\
"- ccg (connected_components_graph) : a group of adjacent pixels of the same color in the original graph.\n" +\
"- nbhcg (non_background_horizontal_connected_components_graph): a group of horizontally adjacent pixels of the same color in the original graph, excluding background color. \n" +\
"- lrg (largest_rectangle_graph): a group of adjacent pixels of the same color in the original graph that makes up a rectangle, excluding black. rectangles are identified from largest to smallest.\n" +\
"- mcccg (multicolor_connected_components_graph): a group of adjacent pixels of any non-background color in the original graph. \n" +\
"- na: (no_abstraction_graph): the entire graph as one multi-color node. \n"
test_format_prompt = "Your task is to pick the most likely abstraction from the above list that would allow us to find a solution for the provided few-shot examples.\n"

# Test Prompt
def test_prompt(task_id):
    print(task_id)
    grids = data.task_description(task_id, color_map="char")
    prompt = f"""{task_preamble}\n{abstraction_prompt}\n{grids}\n{test_format_prompt}"""
    return prompt#, gold_sol

test = ['08ed6ac7', '1e0a9b12', '25ff71a9', '3906de3d', '4258a5f9',
        '50cb2852', '543a7ed5', '6455b5f5', '67385a82', '694f12f3',
        '6e82a1ae', '7f4411dc', 'a79310a0', 'aedd82e4', 'b1948b0a',
        'b27ca6d3', 'bb43febb', 'c8f0f002', 'd2abd087', 'dc1df850', 'ea32f347']

#for test_id in test[0:]:
test_id = 'ea32f347'
print(test_id)
print(test_prompt(test_id))
    #print(test_prompt(test_id)[1])
    #response = get_completion(test_prompt(test_id)[0])
    #print(response)

# "nbvcg - get_non_background_vertical_connected_components_graph" +\
# "nbccg - get_non_black_components_graph" +\ 
# "ccgbr - get_connected_components_graph_background_removed" +\
# "ccg - get_connected_components_graph" +\
# "lrg - get_largest_rectangle_graph" +\