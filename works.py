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
from transform_synthesis import TSizeEnumerator

expected_nodes = []
def setUp(taskNumber, abstraction):
    task = Task("ARC/data/training/" + taskNumber + ".json")
    task.abstraction = abstraction
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                            input in task.train_input]
    task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                            output in task.train_output]
    task.get_static_inserted_objects()
    task.get_static_object_attributes(task.abstraction)
    setup_size_and_degree_based_on_task(task)
    for input_abstracted_graphs in task.input_abstracted_graphs_original[task.abstraction]:
        local_data = []
        for node, data in input_abstracted_graphs.graph.nodes(data=True):
            local_data.extend(data['nodes'])
        expected_nodes.append(local_data)
    return task

def sort_leaf_makers(leaf_makers, probabilities):
    def get_probability(leaf_maker):
        if isinstance(leaf_maker, Color):
            return probabilities['COLOR'].get(leaf_maker.value, 0)
        elif isinstance(leaf_maker, Dir):
            return probabilities['DIRECTION'].get(leaf_maker.value, 0)
        elif isinstance(leaf_maker, Overlap):
            return probabilities['OVERLAP'].get(leaf_maker.value, 0)
        elif isinstance(leaf_maker, Rotation_Angle):
            return probabilities['ROTATION_ANGLE'].get(leaf_maker.value, 0)
        elif isinstance(leaf_maker, Symmetry_Axis):
            return probabilities['SYMMETRY_AXIS'].get(leaf_maker.value, 0)
        else:
            return 0
    return sorted(leaf_makers, key=get_probability, reverse=True)

def sort_vocab_makers(vocab_makers, probabilities):
    def get_transform_probability(vocab_maker):
        transform_mapping = {
            UpdateColor: ('UpdateColor', 'COLOR'),
            MoveNode: ('MoveNode', 'DIRECTION'),
            ExtendNode: ('ExtendNode', 'DIRECTION', 'OVERLAP'),
            MoveNodeMax: ('MoveNodeMax', 'DIRECTION'),
            RotateNode: ('RotateNode', 'ROT_ANGLE'),
            AddBorder: ('AddBorder', 'COLOR'),
            FillRectangle: ('FillRectangle', 'COLOR', 'OVERLAP'),
            HollowRectangle: ('HollowRectangle', 'COLOR'),
            Flip: ('Flip', 'SYMMETRY_AXIS')
        }
        transform_key = transform_mapping.get(vocab_maker, ())
        return probabilities['Transform'].get(transform_key, 0)
    return sorted(vocab_makers, key=get_transform_probability, reverse=True)

from pcfg_compute import laplace_smoothing
transform_probabilities = laplace_smoothing()

tleaf_makers = [Color.black, Color.blue, Color.red, Color.green, Color.yellow, Color.grey, Color.fuchsia, Color.orange, Color.cyan, Color.brown,
            NoOp(), Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT, Dir.DOWN_LEFT, Dir.DOWN_RIGHT, Dir.UP_LEFT,
            Dir.UP_RIGHT, Overlap.TRUE, Overlap.FALSE, Rotation_Angle.CW, Rotation_Angle.CCW, Rotation_Angle.CW2, Symmetry_Axis.VERTICAL,
            Symmetry_Axis.HORIZONTAL, Symmetry_Axis.DIAGONAL_LEFT, Symmetry_Axis.DIAGONAL_RIGHT]
t_vocabMakers = [UpdateColor, MoveNode, ExtendNode, MoveNodeMax, RotateNode, AddBorder, FillRectangle, HollowRectangle, Flip]
fleaf_makers = [FColor.black, FColor.blue, FColor.red, FColor.green, FColor.yellow, FColor.grey, FColor.fuchsia, FColor.orange, FColor.cyan, FColor.brown]
f_vocabMakers = [Size, Degree, FilterByColor, FilterBySize, FilterByDegree, FilterByNeighborColor, FilterByNeighborSize, FilterByNeighborDegree, Or, And, Not]

transform_vocab = VocabFactory(tleaf_makers, t_vocabMakers)
pcfg = False # toggle this

if pcfg:
    sorted_leaf_makers = sort_leaf_makers(tleaf_makers, transform_probabilities)
    sorted_vocab_makers = sort_vocab_makers(t_vocabMakers, transform_probabilities)
    transform_vocab = VocabFactory(sorted_leaf_makers, sorted_vocab_makers)
    print("PCFG Sorted:", sorted_leaf_makers)

def filter_matches(matches, expected):
    filtered_matches = []
    for match_list, expected_list in zip(matches, expected):
        filtered_list = [match for match in match_list if match in expected_list]
        filtered_matches.append(filtered_list)

    return filtered_matches

def compare_graphs(transformed_graph, true_output_graph):
    errors = 0
    correct_matches, mismatch, all_nodes = [], [], []

    for iter in range(len(transformed_graph)):
        local_correct, local_mismatch, all = [], [], []
        actual = transformed_graph[iter]
        expected = true_output_graph[iter]
        reconstructed_transformed = task.train_input[iter].undo_abstraction(actual)
        reconstructed_expected = task.train_input[iter].undo_abstraction(expected)
        transformed_nodes = {node: data['color'] for node, data in reconstructed_transformed.graph.nodes(data=True)}
        expected_nodes = {node: data['color'] for node, data in reconstructed_expected.graph.nodes(data=True)}
        assert len(transformed_nodes.keys()) == len(expected_nodes.keys())
        for node, transformed_color in transformed_nodes.items():
            expected_color = expected_nodes.get(node)
            if expected_color is not None:
                if transformed_color != expected_color:
                    #print(f"Mismatch at node {node}: Transformed color {transformed_color}, Expected color {expected_color}")
                    errors += 1
                    local_mismatch.append(node)
                    all.append(node)
                else:
                    local_correct.append(node)
                    all.append(node)
            else:
                print(f"Node {node} not found in expected graph")
                errors += 1

        mismatch.append(local_mismatch)
        correct_matches.append(local_correct)
        all_nodes.append(all)
    return errors, correct_matches, mismatch, all_nodes

from itertools import combinations
def synthesize_transforms(task, vocab):
    enumerator = TSizeEnumerator(task, vocab, ValuesManager())
    partial_solutions = {}
    previous_programs = set()  # Set to track programs from previous iterations
    counter = 0
    new_programs = set()
    while enumerator.hasNext():
        program = enumerator.next()
        print("code:", program.code)
        print("count:", counter)
        counter += 1
        if program.values != []:
            transformed_graph, output_graph = task.apply_transformation(program, task.abstraction)
            if len(transformed_graph) != len(output_graph):
                continue
            else:
                new_programs.add(program)
                errors, matches, mismatches, total_pixels = compare_graphs(transformed_graph, output_graph)
                if program.nodeType == Types.NO_OP:
                    changed_pixels = mismatches
                    partial_solutions[program] = [list(match) for match in matches]

                if errors == 0:
                    print("All node colors match correctly.")
                    print("Program count:", counter)
                    print("matches:", matches)
                    return {program: matches}
                elif matches:
                    print("mismatches:", [list(match) for match in matches])
                    print("changed:", changed_pixels)
                    print(all(any(m in c for m in match_sublist) for match_sublist, c in zip(matches, changed_pixels)))
                    if [list(match) for match in matches] not in partial_solutions.values() and \
                    all(any(m in c for m in match_sublist) for match_sublist, c in zip(matches, changed_pixels)): # ignore programs that cover same subset of nodes
                        partial_solutions[program] = [list(match) for match in matches]

                    # Check combinations of partial solutions for complete coverage
                    current_programs = set(partial_solutions.keys())
                    new_programs = current_programs - previous_programs  # only consider subsets from newly added programs

                    for num in range(2, len(partial_solutions) + 1):
                        for subset in combinations(partial_solutions.keys(), num):
                            if not any(program in new_programs for program in subset):
                                continue
                            print("subset considered", [sub.code for sub in subset])
                            solution_sets = [list(partial_solutions[program]) for program in subset]
                            combined_solutions = [sorted(list(set().union(*tuples))) for tuples in zip(*solution_sets)]
                            if all(sol == exp for sol, exp in zip(combined_solutions, total_pixels)):
                                print("Collective solution found with subset of partial solutions.")
                                print("Program Count:", counter)
                                return {program: partial_solutions[program] for program in subset}
                                
                    previous_programs = current_programs  # Update for the next iteration

from itertools import combinations
def synthesize(task, vocab):
    enumerator = TSizeEnumerator(task, vocab, ValuesManager())
    partial_solutions = {}
    previous_programs = set()  # Set to track programs from previous iterations
    counter = 0
    new_programs = set()
    while enumerator.hasNext():
        program = enumerator.next()
        print("code:", program.code)
        print("count:", counter)
        counter += 1
        if program.values != []:
            transformed_graph, output_graph = task.apply_transformation(program, task.abstraction)
            if len(transformed_graph) != len(output_graph):
                continue
            else:
                new_programs.add(program)
                errors, matches, mismatches, total_pixels = compare_graphs(transformed_graph, output_graph)
                if program.nodeType == Types.NO_OP:
                    changed_pixels = mismatches
                    partial_solutions[program] = [list(match) for match in matches]

                if errors == 0:
                    print("All node colors match correctly.")
                    print("Program count:", counter)
                    print("matches:", matches)
                    
                    filtered_matches = filter_matches(matches, expected_nodes)
                    filter_solution = synthesize_filter(filtered_matches)
                    return {program: matches}
            
                elif matches:
                    print(all(any(m in c for m in match_sublist) for match_sublist, c in zip(matches, changed_pixels)))
                    if [list(match) for match in matches] not in partial_solutions.values() and \
                    all(any(m in c for m in match_sublist) for match_sublist, c in zip(matches, changed_pixels)): # ignore programs that cover same subset of nodes
                        partial_solutions[program] = [list(match) for match in matches]

                    # Check combinations of partial solutions for complete coverage
                    current_programs = set(partial_solutions.keys())
                    new_programs = current_programs - previous_programs  # only consider subsets from newly added programs

                    for num in range(2, len(partial_solutions) + 1):
                        for subset in combinations(partial_solutions.keys(), num):
                            if not any(program in new_programs for program in subset):
                                continue
                            print("subset considered", [sub.code for sub in subset])
                            solution_sets = [list(partial_solutions[program]) for program in subset]
                            combined_solutions = [sorted(list(set().union(*tuples))) for tuples in zip(*solution_sets)]
                            if all(sol == exp for sol, exp in zip(combined_solutions, total_pixels)):
                                print("Collective solution found with subset of partial solutions.")
                                print("Program Count:", counter)
                                solution = {program: partial_solutions[program] for program in subset}
                                all_filters_found = True

                                for prog in solution.items():
                                    print("Program:", prog)
                                    filtered_matches = filter_matches(prog[1], expected_nodes)
                                    filter_solution = synthesize_filter(filtered_matches)

                                    if not filter_solution:
                                        all_filters_found = False
                                        print("No filter solution found for this program")
                                        break
                                if all_filters_found:
                                    print("Filter solutions found for all programs")
                                    return solution
                    previous_programs = current_programs  # Update for the next iteration

def synthesize_filter(subset, timeout: int=0):
    #TODO: what is wrong with filter synthesis for 6455b5f5
    taskNumber = "08ed6ac7"
    task = Task("ARC/data/training/" + taskNumber + ".json")
    task.abstraction = "nbccg"
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                            input in task.train_input]
    task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                            output in task.train_output]
    task.get_static_inserted_objects()
    task.get_static_object_attributes(task.abstraction)
    setup_size_and_degree_based_on_task(task)
    vocabMakers = [FColor.black, FColor.blue, FColor.brown, FColor.cyan, FColor.fuchsia,
                   Size, Degree, FilterByColor, FilterBySize, FilterByDegree,
                   FilterByNeighborColor, FilterByNeighborSize,
                   FilterByNeighborDegree, Or, And, Not] # TODO: Add Not
    vocabMakers = [Degree, Size, FColor, FilterByColor, FilterBySize, FilterByDegree,
                   FilterByNeighborColor, FilterByNeighborSize,
                   FilterByNeighborDegree, Or, And, Not]
    vocab = VocabFactory.create(vocabMakers)
    enumerator = FSizeEnumerator(task, vocab, ValuesManager())
    i = 0
    # check correctness
    while enumerator.hasNext():
        program = enumerator.next()
        # TODO: check for all subset correctness!
        i += 1
        results = program.values
        print(f"Program: {program.code}: {results, program.size}")
        print("results:", results)
        print("subset:", subset)
        check = [sorted(sub1) == sorted(sub2) for sub1, sub2 in zip(results, subset)]
        if check != [] and all(check):
            print("Solution!", program.code, program.size)
            return program

solution_found = False
previous_subsets = []  # list to keep track of filters used in previous programs
from itertools import zip_longest

task = setUp("08ed6ac7", "nbccg")
synthesize(task, transform_vocab)
"""
while True:
    solution = synthesize_transforms(task, transform_vocab)  ## TODO: do not create a new enumerator!
    print("Transform Solution:", solution) # TODO: no need to synthesize solutions for NoOp
    all_filters_found = True
    pixels = [filter_matches(value, expected_nodes) for key, value in solution.items()]
    print("sss:", pixels)
    for idx, (prog_key, prog_value) in enumerate(solution.items()):
        print(idx)
        filtered_matches = filter_matches(prog_value, expected_nodes)
        print("filtered;", filtered_matches)
        print("Synthesizing filters for {prog}", prog_key.code)
        final_lst = []
        for iter in range(len(pixels[0])):
            print(iter)
            local_lst = []
            for pixel_lst in pixels[idx:]:
                local_lst.extend(pixel_lst[iter])
            final_lst.append(local_lst)
        print("finale:", final_lst)
        filter_solution = synthesize_filter(final_lst)

        if not filter_solution:
            all_filters_found = False
            print("No filter solution found for this program.")
            break
    if all_filters_found:
        print("Filter solutions found for all programs.")
        break
print("Finished process. Solution found:", all_filters_found)
"""

test_problems = \
[("08ed6ac7", "nbccg"), ("bb43febb", "nbccg"), ("25ff71a9", "nbccg"), ("3906de3d", "nbvcg"), ("4258a5f9", "nbccg"),
 ("50cb2852", "nbccg"), ("6455b5f5", "ccg"), ("d2abd087", "nbccg"), ("c8f0f002", "nbccg"), ("67385a82", "nbccg"),
 ("dc1df850", "nbccg"), ("b27ca6d3", "nbccg"), ("6e82a1ae", "nbccg"), ("aedd82e4", "nbccg"), ("b1948b0a", "nbccg"),
 ("ea32f347", "nbccg"), ("7f4411dc", "lrg")]

# TODO: check for ea32f347 if it is the smallest program :)
# a79310a0 (nbccg), 
# 543a7ed5 (mcccg), 
# 08ed6ac7 (nbccg) -- # no if finds the set of transforms, you just need to probably not look for Empty transform subsets? 
# don't try to search for them but do exclude them from the other program
# 694f12f3 (nbccg) -- for this you need to resynthesize transforms