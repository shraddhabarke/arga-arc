import pandas as pd
from itertools import product
from task import *
from collections import defaultdict
import itertools
from math import log2, floor, ceil

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

#visualize_pcfg_as_table(t_blown_up_pcfg)
#visualize_pcfg_as_table(f_blown_up_pcfg)

class PCFGConstructor:
    def __init__(self):
        self.pcfg_filters = {}
        self.pcfg_transformations = {}

def get_nodes_from_params(params_list):
    nodes = {}
    for params in params_list:
        for key, value in params.items():
            if isinstance(value, dict):
                inner_nodes = get_nodes_from_params([value])
                for k, v in inner_nodes.items():
                    nodes[k] = nodes.get(k, 0) + v
            else:
                nodes[(key, value)] = nodes.get((key, value), 0) + 1
    return nodes

def getAllNodefromprograms(program):
    filter_nodes = {}
    transformation_nodes = {}
    # Extract filter nodes
    for filter_type, params in zip(program['filters'], program['filter_params']):
        filter_nodes[(filter_type, None)] = filter_nodes.get((filter_type, None), 0) + 1
        for k, v in get_nodes_from_params([params]).items():
            filter_nodes[k] = filter_nodes.get(k, 0) + v

    # Extract transformation nodes
    for transformation_type, params in zip(program['transformation'], program['transformation_params']):
        transformation_nodes[(transformation_type, None)] = transformation_nodes.get((transformation_type, None), 0) + 1
        for k, v in get_nodes_from_params([params]).items():
            transformation_nodes[k] = transformation_nodes.get(k, 0) + v
    return filter_nodes, transformation_nodes

def getAllNodesFromSeriesOfPrograms(programs):
    all_filter_nodes = {}
    all_transformation_nodes = {}

    for program in programs:
        filter_nodes, transformation_nodes = getAllNodefromprograms(program)
        for k, v in filter_nodes.items():
            all_filter_nodes[k] = all_filter_nodes.get(k, 0) + v
            #print(all_filter_nodes)
        for k, v in transformation_nodes.items():
            all_transformation_nodes[k] = all_transformation_nodes.get(k, 0) + v
            #print(all_transformation_nodes)

    return all_filter_nodes, all_transformation_nodes

program = {
    'filters': ['filter_by_color'],
    'filter_params': [{'color': 5, 'exclude': False}],
    'transformation': ['add_border'],
    'transformation_params': [{'border_color': 1}]
}

programs = [
 {'filters': ['filter_by_degree'], 'filter_params': [{'size': 4, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_neighbor_degree'], 'filter_params': [{'degree': 3, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_size'], 'filter_params': [{'size': 5, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_neighbor_color'], 'filter_params': [{'color': 1, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_degree'], 'filter_params': [{'size': 2, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_neighbor_degree'], 'filter_params': [{'degree': 1, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_size'], 'filter_params': [{'size': 3, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_color'], 'filter_params': [{'color': 2, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]},
 {'filters': ['filter_by_neighbor_color'], 'filter_params': [{'color': 2, 'exclude': True}], 'transformation': ['update_color'], 'transformation_params': [{'color': 0}]}
]

# Context-Dependent PCFG!
def get_context_dependent_nodes_from_params(params, operator):
    nodes = {}
    combo_values = [[value] if not isinstance(value, (list, tuple)) else value for value in params.values()]
    for combo in itertools.product(*combo_values):
        context_key = (operator,) + combo
        nodes[context_key] = nodes.get(context_key, 0) + 1
    return nodes

def getAllContextDependentNodefromprograms(program):
    filter_nodes = {}
    transformation_nodes = {}

    # Extract filter nodes
    for filter_type, params in zip(program['filters'], program['filter_params']):
        nodes = get_context_dependent_nodes_from_params(params, filter_type)
        for k, v in nodes.items():
            filter_nodes[k] = filter_nodes.get(k, 0) + v

    # Extract transformation nodes
    for transformation_type, params in zip(program['transformation'], program['transformation_params']):
        nodes = get_context_dependent_nodes_from_params(params, transformation_type)
        for k, v in nodes.items():
            transformation_nodes[k] = transformation_nodes.get(k, 0) + v

    return filter_nodes, transformation_nodes

def getAllContextDependentNodesFromSeriesOfPrograms(programs):
    all_filter_nodes = {}
    all_transformation_nodes = {}

    for program in programs:
        filter_nodes, transformation_nodes = getAllContextDependentNodefromprograms(program)
        for k, v in filter_nodes.items():
            all_filter_nodes[k] = all_filter_nodes.get(k, 0) + v
        for k, v in transformation_nodes.items():
            all_transformation_nodes[k] = all_transformation_nodes.get(k, 0) + v

    return all_filter_nodes, all_transformation_nodes

def expo(base, exponent):
    """Exponentiation helper function."""
    return base ** exponent

def update_pcfg_with_transforms(initial_pcfg, changed_transforms):
    print("Init:", initial_pcfg)
    print("Change:", changed_transforms)
    # Step 1: Adjust probabilities based on the changed transforms
    total_programs = 10
    for changed_node, count in changed_transforms.items():
        print("c1:", changed_node)
        print("c2:", count)
        fit = count / total_programs
        print("fit:", fit)
        if changed_node in initial_pcfg:
            initial_pcfg[changed_node] = expo(initial_pcfg[changed_node], 1 - fit)

    # Step 2: Normalize the probabilities
    total_probability = sum(initial_pcfg.values())
    for node, prob in initial_pcfg.items():
        initial_pcfg[node] = prob / total_probability

    # Step 3: Convert the updated probabilities into their integer log2 representations
    for node, prob in initial_pcfg.items():
        value = -log2(prob)
        initial_pcfg[node] = create_round_value(value)

    return initial_pcfg

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

def round_value(num):
    """Round a number to the nearest integer."""
    return ceil(num) if num - floor(num) > 0.5 else floor(num)

changed_filters = getAllNodesFromSeriesOfPrograms(programs)[0]
changed_transformations = getAllNodesFromSeriesOfPrograms(programs)[1]

init_transforms = construct_uniform_pcfg_with_values(transformation_operators, transformation_parameter_values)
init_filters = construct_uniform_pcfg_with_values(filter_operators, filter_parameter_values)

# Context-Dependant!
changed_filters1, changed_transformations1 = getAllContextDependentNodesFromSeriesOfPrograms(programs)

init_transforms1 = construct_blown_up_pcfg(transformation_operators, transformation_parameter_values)
init_filters1 = construct_blown_up_pcfg(filter_operators, filter_parameter_values)

# Calling the function
updated_pcfg = update_pcfg_with_transforms(init_transforms, changed_transformations)
print("update_pcfg", updated_pcfg)

updated_pcfg1 = update_pcfg_with_transforms(init_transforms1, changed_transformations1)
print("update_pcfg1", updated_pcfg1)

visualize_pcfg_as_table(init_transforms)
# Random Testing
initial_prior = create_prior(init_transforms)
blown_up_prior = create_prior(t_blown_up_pcfg)

print("Initial Prior:", len(initial_prior), initial_prior)
print("Blown-up Prior:", len(blown_up_prior), blown_up_prior)