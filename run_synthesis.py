import time
from itertools import combinations, permutations
from task import *
import matplotlib.pyplot as plt
from collections import Counter
from inspect import signature
from transform import *
from filters import *
from typing import *
# from pcfg.pcfg_compute import *
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
                Symmetry_Axis, ObjectId, Variable(
                    "Var"), UpdateColor, MoveNode, MoveNodeMax, ExtendNode]  # FillRectangle,
# AddBorder, ExtendNode, RotateNode, HollowRectangle, Flip, Mirror, Transforms]
f_vocabMakers = [FColor, Degree, Size, Relation, FilterByColor, FilterBySize, FilterByDegree, FilterByNeighborColor, FilterByNeighborSize,
                 FilterByNeighborDegree, Not, And, Or]

pcfg = False  # toggle this
if pcfg:
    transform_probabilities = laplace_smoothing()
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
    correct_out_matches, correct_in_matches = [], []
    for input_graph, output_graph in zip(actual_graphs, expected_graphs):
        out_dict = output_graph.graph.nodes(data=True)
        in_matches, out_matches = [], []
        for out_node, out_props in out_dict:
            temp_keys, temp_vals = [], []
            for in_node, in_props in input_graph:
                if isinstance(in_props['color'], int) and isinstance(out_props['color'], int):
                    if in_props['color'] == out_props['color'] and \
                            set(in_props['nodes']).issubset(set(out_props['nodes'])):
                        temp_vals.append(in_props['nodes'])
                        temp_keys.append(in_node)
                        if (len(set(in_props['nodes'])) == len(set(out_props['nodes']))
                                and in_props['size'] == out_props['size']):
                            out_matches.append(out_node)
                            in_matches.append(in_node)
                            break
                        elif len(set(out_props['nodes'])) == len([item for sublist in temp_vals for item in sublist]):
                            out_matches.append(out_node)
                            in_matches.append(tuple(temp_keys))
                            break
                        else:
                            continue

                # for the na abstraction the entire graph is treated as one with a list of colors instead of one color
                elif len(in_props['color']) == len(out_props['color']) and len(out_props['color']) != 1 \
                        or len(in_props['color']) != 1:
                    in_dictionary = dict(
                        zip(in_props['nodes'], in_props['color']))
                    out_dictionary = dict(
                        zip(out_props['nodes'], out_props['color']))
                    if out_dictionary == in_dictionary:
                        out_matches.append(out_node)
                        in_matches.append(in_node)
                else:
                    break
        correct_out_matches.append(out_matches)
        correct_in_matches.append(in_matches)
    return correct_out_matches, correct_in_matches


def run_synthesis(taskNumber, abstraction):
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

    def synthesize_filter(subset, timeout: int = 0):
        filter_vocab = VocabFactory.create(f_vocabMakers)
        enumerator = FSizeEnumerator(task, filter_vocab, ValuesManager())
        i = 0
        while enumerator.hasNext():
            program = enumerator.next()
            i += 1
            results = program.values
            # print(f"Program: {program.code}: {results, program.size}")

            if not task.current_spec:
                check = [sorted(sub1) == sorted(sub2)
                         for sub1, sub2 in zip(results, subset)]
                if check != [] and all(check):
                    print("Filter Solution!", program.code,
                          program.size, program.values)
                    return program, i
            else:
                are_equal = all(sorted(sub1.items()) == sorted(sub2.items())
                                for sub1, sub2 in zip(subset, results))
                are_equal = (len(subset) == len(results)) and \
                    all(sorted(sub1.items()) == sorted(sub2.items()) for sub1, sub2
                        in zip(subset, results))
                if are_equal:
                    return program, i

    transform_vocab = VocabFactory.create(tleaf_makers)
    inputs = [input_graph.graph.nodes(
    ) for input_graph in task.input_abstracted_graphs_original[task.abstraction]]  # input-nodes

    enumerator = TSizeEnumerator(task, transform_vocab, ValuesManager())
    output_graphs = task.output_abstracted_graphs_original[task.abstraction]
    counter, correct_table, program_counts = 0, [
        {node: None for node in output_graph.graph.nodes()} for output_graph in output_graphs], {}
    all_correct_out_nodes = [[] for _ in range(
        len(output_graphs))]
    all_correct_in_nodes = [[] for _ in range(
        len(task.input_abstracted_graphs_original[task.abstraction]))]

    # need to make sure all objects in the output graphs have a transform
    while enumerator.hasNext():
        correct_table_updated = False  # Track changes to correct_table
        program = enumerator.next()
        # print("enumerator:", program.code)
        # print("enumerator-vals:", program.values)
        counter += 1
        if not program.values:
            continue

        correct_out_nodes, correct_in_nodes = compare_abstracted_graphs(
            program.values, output_graphs)
        for idx, (out_nodes, in_nodes) in enumerate(zip(correct_out_nodes, correct_in_nodes)):
            all_correct_out_nodes[idx].extend(out_nodes)
            all_correct_in_nodes[idx].extend(in_nodes)

        correct_sets = [set(correct) for correct in correct_out_nodes]
        if all(not correct_set for correct_set in correct_sets):
            continue

        for dict_, correct_set in zip(correct_table, correct_sets):  # per-task
            for key in correct_set:
                if dict_.get(key) is None or sum(len(set_) for set_ in correct_sets) > program_counts.get(dict_.get(key), 0):
                    dict_[key] = program.code
                    program_counts[program.code] = program_counts.get(
                        program.code, 0) + 1
                    correct_table_updated = True  # Set to True if any changes are made

        if correct_table_updated and all(value for dict_ in correct_table for value in dict_.values()):
            updated_correct_table = []
            for table, old_key_set, new_key_set in zip(correct_table, all_correct_out_nodes, all_correct_in_nodes):
                key_map = dict(zip(old_key_set, new_key_set))
                updated_table = {key_map.get(
                    key, None): value for key, value in table.items()}
                updated_correct_table.append(updated_table)

            transforms_data = {program: [] for program in set().union(
                *[d.values() for d in updated_correct_table])}
            for program in transforms_data:
                for idx, dict_ in enumerate(updated_correct_table):
                    transforms_data[program].append(
                        [key for key, prog in dict_.items() if prog == program])
            # all objects are covered, synthesizing filters...
            all_filters_found, filters_sol = True, []

            for program, objects in transforms_data.items():
                print("Enumerating filters:", program, objects)
                task.current_spec = []
                filters = synthesize_filter([[obj for obj in object_lst if obj in set(
                    input_lst)] for input_lst, object_lst in zip(inputs, objects)])  # todo: fix!
                # filters = synthesize_filter([obj[0] for obj in objects])

                if filters:
                    if program not in task.spec.keys():
                        filters_sol.append(filters[0])
                        continue
                    # it is a variable program so synthesizing filters for that!
                    if task.spec[program]:
                        task.current_spec = task.spec[program]
                        variable_filters = synthesize_filter(task.current_spec)
                        if not variable_filters:
                            all_filters_found = False
                            break
                else:
                    all_filters_found = False
                    break

            if all_filters_found:
                print("Transformation Solution:", [
                    program for program in transforms_data.keys()])
                print("Filter Solution", [
                      program.code for program in filters_sol])
                break


test_problems = [("7f4411dc", "lrg"), ("a79310a0", "nbccg")]

evaluation = {"08ed6ac7": "nbccg", "25ff71a9": "nbccg", "3906de3d": "nbvcg", "4258a5f9": "nbccg", "50cb2852": "nbccg",
              "6455b5f5": "ccg", "67385a82": "nbccg", "694f12f3": "nbccg", "6e82a1ae": "nbccg", "aedd82e4": "nbccg",
              "b1948b0a": "nbccg", "b27ca6d3": "nbccg", "bb43febb": "nbccg", "c8f0f002": "nbccg", "d2abd087": "nbccg",
              "dc1df850": "nbccg", "ea32f347": "nbccg", "00d62c1b": "ccgbr", "9565186b": "nbccg", "810b9b61": "ccgbr",
              "a5313dff": "ccgbr", "b230c067": "nbccg", "d5d6de2d": "ccg", "67a3c6ac": "na", "3c9b0459": "na",
              "9dfd6313": "na", "ed36ccf7": "na", "6150a2bd": "na",  "68b16354": "na", "5582e5ca": "ccg", "05f2a901": "nbccg"
              }
# "ddf7fa4f": "nbccg", "d43fd935": "nbccg", "ae3edfdc": "nbccg", "dc433765": nbccg? moveNode(Variable)

# todo: add evaluation over test problem
evals = {"d43fd935": "nbccg"}
more_problems = {"a48eeaf7": "nbccg", "dc433765": "nbccg"}
for task, abstraction in evals.items():
    start_time = time.time()
    print("taskNumber:", task)
    run_synthesis(task, abstraction)
    print(f"Problem {task}: --- {(time.time() - start_time)} seconds ---")
