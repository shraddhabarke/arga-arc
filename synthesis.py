from task import *
import matplotlib.pyplot as plt
from collections import Counter
from inspect import signature
from transform import *
from filters import *
from typing import *
from OEValuesManager import *
from VocabMaker import *
from filter_synthesis import FSizeEnumerator
import concurrent.futures
from transform_synthesis import TSizeEnumerator
import itertools

def synthesize_filter(taskNumber: str, subset=None, timeout: int=0):
    taskNumber = "08ed6ac7"
    task = Task("dataset/" + taskNumber + ".json")
    task.abstraction = "nbccg"
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                            input in task.train_input]
    task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                            output in task.train_output]
    task.get_static_inserted_objects()
    task.get_static_object_attributes(task.abstraction)
    setup_size_and_degree_based_on_task(task)
    vocabMakers = [FColor, Size, Degree, FilterByColor, FilterBySize, FilterByDegree,
                   FilterByNeighborColor, FilterByNeighborSize,
                   FilterByNeighborDegree, Or, And]
    vocab = VocabFactory.create(vocabMakers)
    enumerator = FSizeEnumerator(task, vocab, ValuesManager())
    i = 0
    # check correctness
    while enumerator.hasNext():
        program = enumerator.next()
        i += 1
        actual_result = program.values
        print(f"Program: {program.code}: {actual_result, program.size}")
        results = program.values
        if results != []:
            results = results[1]
        if set(map(tuple, results)) == set(map(tuple, subset)):
            print("Solution!", program.code, program.size)
            return program
    # TODO: Step2: change the transform enumerators to take the filter as input such that it can do more fine-grained OE :)
    # TODO: Step3: once we have the subsets and filters, start parallel transform enumerators then break as soon as one of them finds a program
    # TODO: OE for Transforms sequence

task = Task("dataset/" + "08ed6ac7" + ".json")
task.abstraction = "nbccg"
task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                        input in task.train_input]
task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                        output in task.train_output]

def get_cartesian_product_of_subsets(input_abstracted_graphs):
    # Generate subsets for each input point
    all_subsets_per_input = [
        graph.get_all_subsets() for graph in input_abstracted_graphs
    ]

    # Compute Cartesian product of these subsets
    cartesian_product = itertools.product(*all_subsets_per_input)
    return list(cartesian_product)

task.get_static_inserted_objects()
task.get_static_object_attributes(task.abstraction)
setup_size_and_degree_based_on_task(task)
cartesian_product_of_subsets = get_cartesian_product_of_subsets(task.input_abstracted_graphs_original[task.abstraction])
subsets = task.input_abstracted_graphs_original[task.abstraction][1].get_all_subsets()
print("Subsets:", subsets)

results_dict = {}

for subset_tuple in cartesian_product_of_subsets:
    # Process each dictionary in the tuple to extract keys
    combined_keys = []
    for subset_dict in subset_tuple:
        combined_keys.extend(subset_dict.keys())

    filter_result = synthesize_filter(taskNumber="08ed6ac7", subset=combined_keys)
    print("Solution!:", filter_result.code)
    if filter_result is None:
        raise ValueError("No solution found for subset:", combined_keys)
    results_dict[tuple(combined_keys)] = filter_result

for subset in subsets:
    subset = list(subset.keys())
    filter_result = synthesize_filter(taskNumber="08ed6ac7", subset=subset)
    if filter_result is None:
        raise ValueError("No solution found for subset:", subset)
    results_dict[tuple(subset)] = filter_result
    print("Subset:", subset, filter_result.code)

def find_transformations(filter_dict, taskName, vocab, oeManager, contexts):
    filter_transform_pairs = {}
    # The set to track all points covered by the transforms
    all_covered_points = set()

    for subset, filter_node in filter_dict.items():
        transform_enumerator = TSizeEnumerator(taskName, vocab, filter_node, oeManager, contexts)

        while transform_enumerator.hasNext():
            transform = transform_enumerator.next()
            # Assuming you have some method to check if this transform is valid for this filter. Update accordingly.
            if True: # correctly transformed
                filter_transform_pairs[filter_node] = transform
                # Add the covered points by this transform to our set
                all_covered_points.update(subset)
                break

        if all_covered_points == set([item for sublist in filter_dict.keys() for item in sublist]):
            break
    return filter_transform_pairs

print("Results:", results_dict)