from task import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# GPT4 output: TODO need to tie it together with the call to the model's API
gpt_programs = [[{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                'transformation': ['move_node_max'], 'transformation_params': [{'direction': 'RIGHT'}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['hollow_rectangle'], 'transformation_params': [{'fill_color': 2}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['extend_node'], 'transformation_params': [{'direction': 'DOWN', 'overlap': True}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['add_border'], 'transformation_params': [{'border_color': 2}]}],
                [{'filters': ['filter_by_color'], 'filter_params': [{'color': 1, 'exclude': False}],
                  'transformation': ['update_color'], 'transformation_params': [{'color': 2}]}]]


def check_fields(inp1, inp2):
    """
    Given two dictionaries with nodes as keys, and colors, sizes as values, check if they are the same
    """
    return inp1['color'] == inp2['color'] and inp1['size'] == inp2['size']


def setCoverage(programDict, totalPixels):
    """
    Implements the greedy set cover algorithm to find the minimum number of rules needed to cover all the pixels,
    adapted from https://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm
    """
    rulesSoFar = []
    chosenNodes = [[] for x in range(len(list(programDict.values())[0]))]
    transformToNodeMapping = {}  # New data structure to store mapping

    while len([item for sublist in chosenNodes for item in sublist]) != totalPixels:
        # TODO: algorithm is not complete since it assumes that all the transformation rules required are in the GPT4 output
        smallestSum = 0
        bestRule = 0
        for rule in programDict.keys():
            currentSum, tempNodes = 0, [[] for x in range(len(list(programDict.values())[0]))]
            if rule not in rulesSoFar:
                for datapoint in range(len(programDict[rule])):
                    for x in range(len(programDict[rule][datapoint])):
                        if programDict[rule][datapoint][x] not in chosenNodes[datapoint]:
                            currentSum += 1
                            tempNodes[datapoint].append(programDict[rule][datapoint][x])

                # greedily increase total number of nodes covered
                if currentSum > smallestSum and currentSum != 0:
                    smallestSum = currentSum
                    bestRule = rule
        if bestRule != 0:
            rulesSoFar.append(bestRule)
            for datapoint in range(len(programDict[bestRule])):
                for x in range(len(programDict[bestRule][datapoint])):
                    if programDict[bestRule][datapoint][x] not in chosenNodes[datapoint]:
                        chosenNodes[datapoint].append(
                            programDict[bestRule][datapoint][x])
        else:
            raise ValueError("Best rule should not be zero!")
    print("Transformation Rules Found:", rulesSoFar)
    return chosenNodes


def computeCorrectPixels(outDict):
    """
    Computes the mapping from program transformation to the number of pixels in the training input it covers correctly.
    Returns a dictionary of the form {(transformation, transformation_params): [correctPixels]} where correctPixels is a list of pixels per training example.
    """
    programDict = {}
    all_programs = task.get_candidate_transformations(
        [program[0]['transformation'] for program in gpt_programs])
    for program in all_programs:
        correctPixels = []
        transformed_graphs = task.apply_solution_nofilter(program['transformation'],
                                                          program['transformation_params'][0], task.abstraction)
        assert len(outDict) == len(
            transformed_graphs), "The length of output and transformed graphs is not the same!"
        # By considering abstracted nodes, we do not assign more weight to abstracted nodes that correspond to more actual nodes in the set coverage algorithm.
        for iter in range(len(outDict)):
            pixels = []
            for trans in transformed_graphs[iter].graph.nodes(data=True):
                if tuple(set(trans[1]['nodes'])) in outDict[iter].keys() and check_fields(trans[1], outDict[iter][tuple(set(trans[1]['nodes']))]):
                    pixels.append(trans[0])
                else:
                    continue
            correctPixels.append(pixels)

        programDict[(program['transformation'][0],
                     program['transformation_params'][0].values())] = correctPixels
    return programDict


def generate_filters(rows_covered):
    training_in = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                   input in task.train_input]
    # start enumerating filters:
    satisfy = False
    for filter_op in ARCGraph.filter_ops:
        if satisfy:
            break
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
            # check if the filter is valid for all training examples
            satisfy = True
            for iter in range(len(training_in)):
                satisfy = satisfy and all([training_in[iter].apply_filters(node, candidate_filter["filters"],
                                                                 candidate_filter["filter_params"])
                                 for node in rows_covered[iter]])
            if satisfy:
                print("Filter Found:", candidate_filter)
                break
            # TODO: generate more specific filters for multiple rules
            # TODO: add support for a combination of filters

def build_node_representation(node_data):
    return tuple(set(node_data['nodes'])), {
        'color': node_data['color'],
        'size': node_data['size']
    }

def main():

    trainIn = [getattr(input, Image.abstraction_ops[task.abstraction])() for input in task.train_input]
    trainOut = [getattr(output, Image.abstraction_ops[task.abstraction])() for output in task.train_output]

    assert len(trainIn) == len(trainOut), "The number of input and output training examples should be the same!"

    inputNodeList = [{build_node_representation(node_data)[0]: build_node_representation(node_data)[1]
                 for _, node_data in train.graph.nodes(data=True)} for train in trainIn]

    totalPixels = 0
    outputNodeList = []
    for idx, train in enumerate(trainOut):
        out = {}
        for _, node_data in train.graph.nodes(data=True):
            node_rep, node_attrs = build_node_representation(node_data)

            if inputNodeList[idx].get(node_rep) != node_attrs:
                totalPixels += 1
                out[node_rep] = node_attrs

        outputNodeList.append(out)

    programDict = computeCorrectPixels(outputNodeList)
    # TODO: prune transformation params based on filters or vice versa!
    chosenNodes = setCoverage(programDict, totalPixels)
    print("Chosen Nodes:", chosenNodes)
    generate_filters(chosenNodes)


if __name__ == "__main__":
    # 3906de3d.json: single rule
    # 08ed6ac7.json: multiple non-interacting rules
    taskNumber = "08ed6ac7"
    task = Task("dataset/" + taskNumber + ".json")
    task.abstraction = "nbccg"
    main()
