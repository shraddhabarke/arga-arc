from task import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

gpt_programs = [[{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['hollow_rectangle'], 'transformation_params': [{'fill_color': 2}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                    'transformation': ['move_node_max'], 'transformation_params': [{'direction': 'RIGHT'}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['extend_node'], 'transformation_params': [{'direction': 'DOWN', 'overlap': True}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['add_border'], 'transformation_params': [{'border_color': 2}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['update_color'], 'transformation_params': [{'color': 2}]}]]


def get_filters():
    ret_apply_filter_calls = []  # final list of filter calls
    for filter_op in ARCGraph.filter_ops:
        # first, we generate all possible values for each parameter
        sig = signature(getattr(ARCGraph, filter_op))
        generated_params = []
        for param in sig.parameters:
            param_name = sig.parameters[param].name
            param_type = sig.parameters[param].annotation
            param_default = sig.parameters[param].default
            if param_name == "self" or param_name == "node":
                continue
            if param_name == "color":
                generated_params.append(
                    [c for c in range(10)] + ["most", "least"])
            elif param_name == "size":
                generated_params.append(
                    [w for w in task.object_sizes[task.abstraction]] + ["min", "max", "odd"])
            elif param_name == "degree":
                generated_params.append(
                    [d for d in task.object_degrees[task.abstraction]] + ["min", "max", "odd"])
            elif param_type == bool:
                generated_params.append([True, False])
            elif issubclass(param_type, Enum):
                generated_params.append([value for value in param_type])

        # then, we combine all generated values to get all possible combinations of parameters
        for item in product(*generated_params):
            # generate dictionary, keys are the parameter names, values are the corresponding values
            param_vals = {}
            # skip "self", "node"
            for i, param in enumerate(list(sig.parameters)[2:]):
                param_vals[sig.parameters[param].name] = item[i]
            candidate_filter = {"filters": [
                filter_op], "filter_params": [param_vals]}
            ret_apply_filter_calls.append(candidate_filter)
    return ret_apply_filter_calls


def enumerator():
    """
    Implements a naive enumerator that tries all candidate filters and transformation parameters for each transform operation in the GPT program.
    """
    flag = False
    enumerated_filters = get_filters()
    for program in gpt_programs:
        transform = program[0]['transformation']
        for filter in enumerated_filters:  # TODO: do lazy evaluation of filters
            sig = signature(getattr(ARCGraph, transform[0]))
            transform_ops = task.parameters_generation(sig)
            for item in product(*transform_ops):
                if flag:
                    print("Solution:", ret_apply_call)
                    return
                param_vals = {}
                for i, param in enumerate(list(sig.parameters)[2:]):
                    param_vals[sig.parameters[param].name] = item[i]
                ret_apply_call = filter.copy()
                ret_apply_call["transformation"] = transform
                ret_apply_call["transformation_params"] = [param_vals]
                actual = task.apply_solution(
                    [ret_apply_call], task.abstraction)[0]
                expected = task.apply_solution(
                    [ret_apply_call], task.abstraction)[1]
                # compare actual and expected
                satisfied = True
                for iter in range(len(actual)):
                    for act, exp in list(zip(actual[iter].graph.nodes(data=True), expected[iter].graph.nodes(data=True))):
                        satisfied = satisfied and set(act[1]['nodes']) == set(
                            exp[1]['nodes']) and act[1]['color'] == exp[1]['color'] and act[1]['size'] == exp[1]['size']
                    print("Satisfied:", satisfied)
                flag = satisfied


if __name__ == "__main__":
    # 3906de3d.json: single rule
    taskNumber = "3906de3d"
    task = Task("dataset/" + taskNumber + ".json")
    task.abstraction = "nbvcg"
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                               input in task.train_input]
    task.get_static_object_attributes(task.abstraction)
    enumerator()
