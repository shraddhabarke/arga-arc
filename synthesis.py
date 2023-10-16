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
    vocabMakers = [Degree, Size, FColor, Exclude, FilterByColor, FilterByDegree, FilterByNeighborColor, FilterBySize, FilterByNeighborSize, FilterByNeighborDegree,
                   Or, And]
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
            results = results[0]
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
task.get_static_inserted_objects()
task.get_static_object_attributes(task.abstraction)
setup_size_and_degree_based_on_task(task)
subsets = task.input_abstracted_graphs_original[task.abstraction][0].get_all_subsets()
results_dict = {}

for subset in subsets:
    subset = list(subset.keys())
    filter_result = synthesize_filter(taskNumber="08ed6ac7", subset=subset)
    if filter_result is None:
        raise ValueError("No solution found for subset:", subset)
    results_dict[tuple(subset)] = filter_result

print(results_dict)