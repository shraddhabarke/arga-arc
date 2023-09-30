import pandas as pd
from itertools import product
from task import *

class Operator:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

taskNumber = "bb43febb"
task = Task("dataset/" + taskNumber + ".json")
task.abstraction = "nbccg"
task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                               input in task.train_input]
task.get_static_object_attributes(task.abstraction)

filter_operators = [
    Operator("filter_by_color", ["color", "exclude"]), 
    Operator("filter_by_size", ["size", "exclude"]),
    Operator("filter_by_degree", ["degree", "exclude"]),
    Operator("filter_by_neighbor_size", ["size", "exclude"]),
    Operator("filter_by_neighbor_color", ["color", "exclude"])
]

filter_parameter_values = {
    "color": [c for c in range(10)] + ["most", "least"],
    "exclude": [True, False],
    "size": [w for w in task.object_sizes[task.abstraction]] + ["min", "max", "odd"],
    "degree": [d for d in task.object_degrees[task.abstraction]] + ["min", "max", "odd"]
}

transformation_operators = [
    Operator("update_color", ["color"]),
    Operator("move_node", ["direction"]),
    Operator("extend_node", ["direction", "overlap"]),
    Operator("move_node_max", ["direction"]),
    Operator("rotate_node", ["rotation_dir"]),
    Operator("add_border", ["border_color"]),
    Operator("fill_rectangle", ["fill_color", "overlap"]),
    Operator("hollow_rectangle", ["fill_color"]),
    Operator("mirror", ["mirror_axis"]),
    Operator("flip", ["mirror_direction"]),
    Operator("insert", ["relative_pos"])
]

transformation_parameter_values = {
    "color": [c for c in range(10)] + ["most", "least"],
    "direction": ["UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"],
    "overlap": [True, False],
    "rotation_dir": ["CW", "CCW", "CW2"],
    "border_color": [c for c in range(10)],
    "fill_color": [c for c in range(10)],
    "mirror_axis": [(y, x) for y in [None, "y"] for x in [None, "x"] if y or x],
    "mirror_direction": ["VERTICAL", "HORIZONTAL", "DIAGONAL_LEFT", "DIAGONAL_RIGHT"],
    "relative_pos": ["SOURCE", "TARGET", "MIDDLE"]
}

def construct_uniform_pcfg_with_values(operators, parameter_values):
    pcfg = {}
    total_count = len(operators)
    
    for values in parameter_values.values():
        total_count += len(values)
    uniform_prob = 1.0 / total_count

    # Assign the uniform probability to every operator
    for op in operators:
        pcfg[(op.name, None)] = uniform_prob
    for param, values in parameter_values.items():
        for value in values:
            pcfg[(param, value)] = uniform_prob
    return pcfg

def construct_blown_up_pcfg(operators, parameter_values):
    pcfg = {}
    # Create combinations for each operator
    for op in operators:
        if len(op.parameters) == 1: 
            for value in parameter_values[op.parameters[0]]:
                pcfg[(op.name, value)] = None
        else:
            params_combinations = product(*[parameter_values[param] for param in op.parameters])
            for combination in params_combinations:
                pcfg[(op.name, *combination)] = None

    # Calculate the uniform probability for each rule in the blown-up PCFG
    uniform_prob = 1.0 / len(pcfg)
    for key in pcfg.keys():
        pcfg[key] = uniform_prob
    return pcfg

from math import log2, floor, ceil

def create_round_value(num):
    """Round a number to the nearest integer."""
    return ceil(num) if num - floor(num) > 0.5 else floor(num)

def create_prior(pcfg):
    """Convert probabilities in the PCFG to their integer log2 representations."""
    prior = {}
    for key, prob in pcfg.items():
        value = -log2(prob)
        prior[key] = create_round_value(value)
    return prior

t_pcfg = construct_uniform_pcfg_with_values(transformation_operators, transformation_parameter_values)
print("initial:", len(t_pcfg), t_pcfg)
t_blown_up_pcfg = construct_blown_up_pcfg(transformation_operators, transformation_parameter_values)
print("blown-up", len(t_blown_up_pcfg), t_blown_up_pcfg)

f_pcfg = construct_uniform_pcfg_with_values(filter_operators, filter_parameter_values)
print("initial:", len(f_pcfg), f_pcfg)
f_blown_up_pcfg = construct_blown_up_pcfg(filter_operators, filter_parameter_values)
print("blown-up", len(f_blown_up_pcfg), f_blown_up_pcfg)

def visualize_pcfg_as_table(pcfg):
    df = pd.DataFrame(list(pcfg.items()), columns=["Rule", "Probability"])
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

visualize_pcfg_as_table(t_blown_up_pcfg)
visualize_pcfg_as_table(f_blown_up_pcfg)

initial_prior = create_prior(t_pcfg)
blown_up_prior = create_prior(t_blown_up_pcfg)

print("Initial Prior:", len(initial_prior), initial_prior)
print("Blown-up Prior:", len(blown_up_prior), blown_up_prior)