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
                Symmetry_Axis, ObjectId, Variable("Var"), MoveNode, MoveNodeMax]
# UpdateColor, MoveNode, RotateNode, AddBorder, FillRectangle, HollowRectangle, ExtendNode, MoveNodeMax, FillRectangle, Transforms
# AddBorder, ExtendNodeVar, RotateNode, FillRectangle, HollowRectangle, ExtendNode, MoveNodeMax, FillRectangle, Transform
# "d43fd935" #"ae3edfdc" #"05f2a901" #"ddf7fa4f" #"d43fd935" "b27ca6d3", "67a3c6ac"

taskNumber = "25ff71a9"
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


def compare_abstracted_graphs(actual_graphs, expected_graphs):
    correct_matches = []
    for input_graph, output_graph in zip(actual_graphs, expected_graphs):
        out_dict = output_graph.graph.nodes(data=True)
        matches = [
            in_node for in_node, in_props in input_graph
            if any(in_props['color'] == out_props['color'] and
                   set(in_props['nodes']) == set(out_props['nodes']) and
                   in_props['size'] == out_props['size']
                   for _, out_props in out_dict)
        ]  # nodes which are correct
        correct_matches.append(matches)
    return correct_matches


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


def synthesize_new():
    enumerator = TSizeEnumerator(task, transform_vocab, ValuesManager())
    counter, filter_counter, correct_table = 0, 0, [{node: None for node in input_graph.graph.nodes()}
                                                    for input_graph in task.input_abstracted_graphs_original[task.abstraction]]
    program_counts = {}  # Global state to maintain program counts
    output_graphs = task.output_abstracted_graphs_original[task.abstraction]

    while enumerator.hasNext():
        program = enumerator.next()
        print("program enumerated:", program.code)
        print("program values:", program.values)
        counter += 1
        if program.values == []:
            continue

        correct = compare_abstracted_graphs(program.values, output_graphs)
        if all(not sublist for sublist in correct):
            continue
        correct_sets = list(map(set, correct))

        for dict_, correct_set in zip(correct_table, correct_sets):  # per-task
            for key in correct_set:
                if dict_.get(key) is None or len(correct_set) > program_counts.get(dict_.get(key), 0):
                    dict_[key] = program.code
                    program_counts[program.code] = program_counts.get(
                        program.code, 0) + 1

        if all(value is not None for dict_ in correct_table for value in dict_.values()):
            # all objects are covered, synthesizing filters...

            transforms_data, all_filters_found = {}, True
            for dict_ in correct_table:
                for key, program in dict_.items():
                    transforms_data.setdefault(program, []).append([key])
            for _, objects in transforms_data.items():
                filters = synthesize_filter(objects)
                if not filters:
                    all_filters_found = False
                    break

            if all_filters_found:
                print("Transformation Solution:", [
                    program for program in transforms_data.keys()])
                return


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
synthesize_new()
print(f"Problem {taskNumber}: --- {(time.time() - start_time)} seconds ---")

test_problems = \
    [("08ed6ac7", "nbccg"), ("bb43febb", "nbccg"), ("25ff71a9", "nbccg"), ("3906de3d", "nbvcg"), ("4258a5f9", "nbccg"),
     ("50cb2852", "nbccg"), ("6455b5f5", "ccg"), ("d2abd087",
                                                  "nbccg"), ("c8f0f002", "nbccg"), ("67385a82", "nbccg"),
        ("dc1df850", "nbccg"), ("b27ca6d3", "nbccg"), ("6e82a1ae",
                                                       "nbccg"), ("aedd82e4", "nbccg"), ("b1948b0a", "nbccg"),
        ("ea32f347", "nbccg"), ("7f4411dc", "lrg"), ("a79310a0", "nbccg")]
