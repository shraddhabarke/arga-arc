import time
from itertools import combinations, permutations
from task import *
import matplotlib.pyplot as plt
from collections import Counter
from inspect import signature
from transform import *
from filters import *
from typing import *
from pcfg.pcfg_compute import *
from OEValuesManager import *
from VocabMaker import *
from filter_synthesis import FSizeEnumerator
from transform_synthesis import TSizeEnumerator


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


def get_probability(vocab_maker):
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
    return transform_probabilities['Transform'].get(transform_key, 0)


def get_fprobability(vocab_maker):
    filter_mapping = {
        FilterByColor: ('FilterByColor', 'COLOR'),
        FilterBySize: ('FilterBySize', 'SIZE'),
        FilterByDegree: ('FilterByDegree', 'DEGREE'),
        FilterByNeighborColor: ('FilterByNeighborColor', 'COLOR'),
        FilterByNeighborSize: ('FilterByNeighborSize', 'SIZE'),
        FilterByNeighborDegree: ('FilterByNeighborDegree', 'DEGREE'),
        Not: ('Not', 'Filter'),
        And: ('And', 'Filter', 'Filter'),
        Or: ('Or', 'Filter', 'Filter'),
    }
    filter_key = filter_mapping.get(vocab_maker, ())
    category = 'Filters' if vocab_maker in [Not, And, Or] else 'Filter'
    return filter_probabilites[category].get(filter_key, 0)


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


# TODO: also account for Not, And, Or
def sort_filter_makers(filter_makers, filter_probs):
    def get_filter_probability(filter_maker):
        # Map filter maker instances to their corresponding keys in filter_probs
        filter_mapping = {
            FilterByColor: ('FilterByColor', 'COLOR'),
            FilterBySize: ('FilterBySize', 'SIZE'),
            FilterByDegree: ('FilterByDegree', 'DEGREE'),
            FilterByNeighborColor: ('FilterByNeighborColor', 'COLOR'),
            FilterByNeighborSize: ('FilterByNeighborSize', 'SIZE'),
            FilterByNeighborDegree: ('FilterByNeighborDegree', 'DEGREE'),
        }
        filter_key = filter_mapping.get(filter_maker, ())
        return filter_probs['Filter'].get(filter_key, 0)
    return sorted(filter_makers, key=get_filter_probability, reverse=True)


# Transform Vocab
tleaf_makers = [Color, NoOp(), Dir, Overlap, Rotation_Angle, Mirror_Axis, RelativePosition, ImagePoints,
                ObjectId, Symmetry_Axis, Variable("Var"), Insert, MoveNode, Transforms]
t_vocabMakers = [ExtendNodeVar]
# UpdateColor, MoveNode, RotateNode, AddBorder, FillRectangle, HollowRectangle, ExtendNode, MoveNodeMax, FillRectangle, Transforms
# AddBorder, ExtendNodeVar, RotateNode, FillRectangle, HollowRectangle, ExtendNode, MoveNodeMax, FillRectangle, Transform
expected_nodes = []
# "d43fd935" #"ae3edfdc" #"05f2a901" #"ddf7fa4f" #"d43fd935" "b27ca6d3", "67a3c6ac"
taskNumber = "3618c87e"
abstraction = "nbccg"
task = Task("ARC/data/training/" + taskNumber + ".json")
task.abstraction = abstraction
task.input_abstracted_graphs_original[task.abstraction] = [getattr(
    input, Image.abstraction_ops[task.abstraction])() for input in task.train_input]
task.output_abstracted_graphs_original[task.abstraction] = [getattr(
    output, Image.abstraction_ops[task.abstraction])() for output in task.train_output]
task.get_static_inserted_objects()
task.get_static_object_attributes(task.abstraction)
setup_objectids(task)
setup_size_and_degree_based_on_task(task)
transform_vocab = VocabFactory.create(tleaf_makers)

# Compute Expected Solution
for input_abstracted_graphs in task.input_abstracted_graphs_original[task.abstraction]:
    local_data = []
    for node, data in input_abstracted_graphs.graph.nodes(data=True):
        local_data.extend(data['nodes'])
    expected_nodes.append(local_data)
# Filter Vocab
f_vocabMakers = [FColor, Degree, Size, FilterByColor, FilterBySize, FilterByDegree,
                 FilterByNeighborColor, FilterByNeighborSize, FilterByNeighborDegree, Not, Or, And]
filter_vocab = VocabFactory.create(f_vocabMakers)

pcfg = False  # toggle this
if pcfg:
    transform_probabilities = laplace_smoothing()
    # TODO: check filter synthesis for ea32f347
    filter_probabilites = laplace_smoothing_for_filters()
    f_vocabMakers = [FilterByColor, FilterBySize, FilterByDegree,
                     FilterByNeighborColor, FilterByNeighborSize, FilterByNeighborDegree, Not, Or, And]
    # TODO: Set the initial costs for the leaves as well
    sorted_leaf_makers = sort_leaf_makers(
        tleaf_makers, transform_probabilities)
    sorted_vocab_makers = sort_vocab_makers(
        t_vocabMakers, transform_probabilities)
    sorted_filters = sort_filter_makers(f_vocabMakers, filter_probabilites)
    transform_values = [get_probability(vocab_maker)
                        for vocab_maker in t_vocabMakers]
    filter_values = [get_fprobability(vocab_maker)
                     for vocab_maker in f_vocabMakers]
    t_trans_number = max(transform_values) + 0.1
    t_filter_number = max(filter_values) + 0.1
    for vocab_maker in t_vocabMakers:
        prob = get_probability(vocab_maker)
        # Set the initial cost of transform vocabs
        vocab_maker.default_size = round(((t_trans_number - prob) * 100) / 2)
    for vocab_maker in f_vocabMakers:
        prob = get_fprobability(vocab_maker)
        vocab_maker.default_size = round(((t_filter_number - prob) * 100) / 2)
    transform_vocab = VocabFactory(sorted_leaf_makers, sorted_vocab_makers)
    filter_vocab = VocabFactory.create([FColor, Size, Degree] + sorted_filters)

# ----------------------------------------------------------------------------------------------------------------------------------------------


def filter_matches(matches, expected):
    filtered_matches = []
    for match_list, expected_list in zip(matches, expected):
        filtered_list = [
            match for match in match_list if match in expected_list]
        filtered_matches.append(filtered_list)
    return filtered_matches


def compare_graphs(transformed_nodes, true_output_graph):
    errors = 0
    correct_matches, mismatch, all_nodes = [], [], []

    for iter in range(len(true_output_graph)):
        local_correct, local_mismatch, all = [], [], []
        expected = true_output_graph[iter]
        reconstructed_expected = task.train_input[iter].undo_abstraction(
            expected)
        expected_nodes = {node: data['color'] for node,
                          data in reconstructed_expected.graph.nodes(data=True)}
        assert len(dict(transformed_nodes[iter]).keys()) == len(
            expected_nodes.keys())
        for node, transformed_color in dict(transformed_nodes[iter]).items():
            expected_color = expected_nodes.get(node)
            if expected_color is not None:
                if transformed_color != expected_color:
                    # print(f"Mismatch at node {node}: Transformed color {transformed_color}, Expected color {expected_color}")
                    errors += 1
                    local_mismatch.append(node)
                    all.append(node)
                else:
                    local_correct.append(node)
                    all.append(node)
            else:
                # print(f"Node {node} not found in expected graph")
                errors += 1

        mismatch.append(local_mismatch)
        correct_matches.append(local_correct)
        all_nodes.append(all)
    return errors, correct_matches, mismatch, all_nodes


def get_valid_nodes(transformed_values, pixels):
    def get_transformed_values_for_satisfied_nodes(satisfied_nodes, all_nodes, transformed_values):
        result = []
        for node in satisfied_nodes:
            nodes_list = next(value['nodes']
                              for key, value in all_nodes if key == node)
            values = [transformed_values[coord] for coord in nodes_list]
            if len(set(values)) == 1:
                result.append(values[0])
        return result

    can_see = []
    for input_graph, pixel, transform_val in zip(task.input_abstracted_graphs_original[task.abstraction], pixels, transformed_values):
        satisfied_nodes = [tup[0] for tup in input_graph.graph.nodes(
            data=True) if any(node in pixel for node in tup[1]['nodes'])]
        neighbors = [[neighbor for neighbor in input_graph.graph.neighbors(
            node) if neighbor not in satisfied_nodes] for node in satisfied_nodes]
        neighbors = [item for sublist in neighbors for item in sublist]
        neighbors_colors = [input_graph.get_color(
            node) for node in neighbors]  # todo: generalize
        print("color-2:", neighbors_colors)
        print("transform:", transformed_values)
        print(get_transformed_values_for_satisfied_nodes(
            satisfied_nodes, input_graph.graph.nodes(data=True), transform_val))
        can_see_nodes = [value['nodes'] for key_to_find in neighbors for key,
                         value in input_graph.graph.nodes(data=True) if key == key_to_find]
        can_see.append([inner_list
                       for sublist in can_see_nodes for inner_list in sublist])
    return can_see

# TODO: break up this function


def synthesize():
    enumerator = TSizeEnumerator(task, transform_vocab, ValuesManager())
    partial_solutions = {}
    # Set to track programs from previous iterations
    previous_programs = set()
    counter, filter_counter = 0, 0
    new_programs = set()
    while enumerator.hasNext():
        program = enumerator.next()
        print(program.code)
        print(program.values)
        counter += 1
        if program.values == []:
            continue
        new_programs.add(program)
        output_graph = task.output_abstracted_graphs_original[task.abstraction]
        if "Var" not in program.code:
            program.values = [program.values]
        for transformed_values in program.values:
            errors, matches, mismatches, total_pixels = compare_graphs(
                transformed_values, output_graph)
            if program.nodeType == Types.NO_OP:
                changed_pixels = mismatches
                partial_solutions[program] = [list(match) for match in matches]

            if errors == 0:  # Found a complete solution!
                # print("All node colors match correctly.")
                print("Transformation Solution:", program.code)
                print("Transformation Count:", counter)

                # Synthesizing filters...
                filtered_matches = filter_matches(matches, expected_nodes)
                _, filter_counter = synthesize_filter(filtered_matches)
                print("Filter Count:", filter_counter)
                if Types.VARIABLE in program.childTypes:
                    var_nodes = get_valid_nodes(
                        transformed_values, filtered_matches)
                    print("Synthesizing variable filters:")
                    filters = synthesize_filter(var_nodes)
                return {program: matches}

            # TODO: change this logic to look at objects covered instead of pixels
            elif matches:   # Optimization: Ignore partial programs that cover same subset of objects correctly
                if [list(match) for match in matches] not in partial_solutions.values() and \
                        all(any(m in c for m in match_sublist) for match_sublist, c in zip(matches, changed_pixels)):   # TODO: check why this is not sound for aabf363d.json
                    # Optimization: Check if in the pixels that are correct, at least one pixel is in pixels that are supposed to change
                    partial_solutions[program] = [
                        list(match) for match in matches]

                # Check combinations of partial solutions for complete coverage
                current_programs = set(partial_solutions.keys())
                # Optimization: Only consider subsets from newly added programs
                new_programs = current_programs - previous_programs

                for num in range(2, len(partial_solutions) + 1):
                    for subset in combinations(partial_solutions.keys(), num):
                        # TODO: how would this change for variable programs
                        # if not any(program in new_programs for program in subset):
                        # continue
                        solution_sets = [list(partial_solutions[program])
                                         for program in subset]

                        print("subset considered:", [
                            program.code for program in subset])
                        combined_solutions = [
                            sorted(list(set().union(*tuples))) for tuples in zip(*solution_sets)]

                        if all(sol == exp for sol, exp in zip(combined_solutions, total_pixels)):
                            print(
                                "Collective solution found with subset of partial solutions.")
                            # Consider different rule orderings
                            # for permuted_subset in permutations(subset):
                            solution = {
                                program: partial_solutions[program] for program in subset}
                            print("Solution considered", [
                                sub.code for sub in solution])
                            # TODO: do not search for filters of the same subset with different orderings
                            # Synthesizing filters...
                            all_filters_found = True
                            pixels = [filter_matches(
                                value, expected_nodes) for key, value in solution.items()]

                            for idx, (prog_key, prog_value) in enumerate(solution.items()):
                                filters = synthesize_filter(pixels[idx])
                                if Types.VARIABLE in prog_key.childTypes:
                                    var_nodes = get_valid_nodes(
                                        transformed_values, pixels[idx])
                                    print("Synthesizing variable filters:")
                                    filters = synthesize_filter(var_nodes)

                                if not filters:  # Filter not found so keep searching for transforms further
                                    all_filters_found = False
                                    break
                                else:
                                    filter_counter += filters[1]

                            if all_filters_found:
                                # print("Filter solutions found for all programs")
                                print("Transformation Counter:", counter)
                                print("Filter Counter:", filter_counter)
                                print("Transformation Solution:", [
                                    program.code for program in solution])
                                return solution
                            # Synthesizing filters...

                    previous_programs = current_programs  # Update for the next iteration


def synthesize_filter(subset, timeout: int = 0):
    enumerator = FSizeEnumerator(task, filter_vocab, ValuesManager())
    i = 0
    while enumerator.hasNext():
        program = enumerator.next()
        i += 1
        results = program.values
        # print(f"Program: {program.code}: {results, program.size}")
        check = [sorted(sub1) == sorted(sub2)
                 for sub1, sub2 in zip(results, subset)]
        if check != [] and all(check):
            print("Filter Solution!", program.code, program.size)
            return program, i


start_time = time.time()
synthesize()
print(f"Problem {taskNumber}: --- {(time.time() - start_time)} seconds ---")

test_problems = \
    [("08ed6ac7", "nbccg"), ("bb43febb", "nbccg"), ("25ff71a9", "nbccg"), ("3906de3d", "nbvcg"), ("4258a5f9", "nbccg"),
     ("50cb2852", "nbccg"), ("6455b5f5", "ccg"), ("d2abd087",
                                                  "nbccg"), ("c8f0f002", "nbccg"), ("67385a82", "nbccg"),
        ("dc1df850", "nbccg"), ("b27ca6d3", "nbccg"), ("6e82a1ae",
                                                       "nbccg"), ("aedd82e4", "nbccg"), ("b1948b0a", "nbccg"),
        ("ea32f347", "nbccg"), ("7f4411dc", "lrg"), ("a79310a0", "nbccg")]
