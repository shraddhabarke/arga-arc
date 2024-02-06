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
                Variable('Var'), Symmetry_Axis, ObjectId, UpdateColor, MoveNode, MoveNodeMax, ExtendNode, Flip, FillRectangle,
                HollowRectangle, RotateNode, AddBorder, Mirror, Transforms]

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


def compare_abstracted_graphs(actual_graphs, expected_graphs):
    correct_out_matches, correct_in_matches = [], []
    for input_graph, output_graph in zip(actual_graphs, expected_graphs):
        out_dict = output_graph.graph.nodes(data=True)
        in_matches, out_matches = [], []
        for out_node, out_props in out_dict:
            sub_keys, sub_vals = [], []
            for in_node, in_props in input_graph:
                if isinstance(in_props['color'], int) and isinstance(out_props['color'], int): # not multi-color
                    if in_props['color'] == out_props['color'] and \
                            set(in_props['nodes']).issubset(set(out_props['nodes'])):
                        sub_vals.append(in_props['nodes'])
                        sub_keys.append(in_node)
                        if (len(set(in_props['nodes'])) == len(set(out_props['nodes']))
                                and in_props['size'] == out_props['size']):
                            out_matches.append(out_node)
                            in_matches.append(in_node)
                            break
                        elif len(set(out_props['nodes'])) == len([item for sublist in sub_vals for item in sublist]):
                            # to handle cases where multiple abstracted nodes in the one graph match
                            # one output abstracted node (due to the way addBorder works in networkx)
                            out_matches.append(out_node)
                            in_matches.append(sub_keys)
                            break
                        else:
                            continue

                # for the na abstraction, the entire graph is treated as one with a list of colors instead of one color
                elif len(in_props['color']) == len(out_props['color']):
                    if dict(zip(in_props['nodes'], in_props['color'])) == dict(
                        zip(out_props['nodes'], out_props['color'])):
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
            #print(f"Program: {program.code}: {results, program.size}")

            if not task.current_spec:
                if all(isinstance(elem, list) for elem in results):
                    check = [sorted(sub1) == sorted(sub2) for sub1, sub2 in zip(results, subset)]
                    if check != [] and all(check):
                        print("Filter Solution!", program.code, program.size, program.values)
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
    input_graphs = task.input_abstracted_graphs_original[task.abstraction]
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
        counter += 1
        if not program.values:
            continue
        correct_out_nodes, correct_in_nodes = compare_abstracted_graphs(
            program.values, output_graphs) # todo: could compute part of this only once
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
                updated_table = {}
                for key, value in table.items():
                    new_key = key_map.get(tuple(key) if isinstance(key, list) else key)
                    if isinstance(new_key, list):
                        for element in new_key:
                            updated_table[element] = value
                    else:
                        updated_table[new_key] = value
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
                filters = synthesize_filter([[obj for obj in object_lst if obj in set(input_lst)]
                                             for input_lst, object_lst in zip(inputs, objects)])

                if filters:
                    if program not in task.spec.keys():
                        filters_sol.append(filters[0])
                        continue
                    # it is a variable program, synthesize filters for that!
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
    return [program for program in transforms_data.keys()], [
                      program.code for program in filters_sol]

evals = {}
for task, abstraction in evals.items():
    start_time = time.time()
    print("taskNumber:", task)
    transformations, filters = run_synthesis(task, abstraction)
    print("transformations:", transformations)
    print("filters:", filters)
    print(f"Problem {task}: --- {(time.time() - start_time)} seconds ---")

class TestEvaluation(unittest.TestCase):
    def test_all_problems(self):
        t1, f1 = run_synthesis("00d62c1b", "ccgbr")
        self.assertCountEqual(['NoOp', 'updateColor(Color.yellow)'], t1)
        self.assertCountEqual(['FilterByColor(FColor.green)', 'FilterByColor(FColor.black)'], f1)
        t2, f2 = run_synthesis("9565186b", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t2)
        self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterByNeighborSize(SIZE.MAX)'], f2)
        t3, f3 = run_synthesis("08ed6ac7", "nbccg")
        self.assertCountEqual(['updateColor(Color.yellow)', 'updateColor(Color.green)', 'updateColor(Color.red)', 'updateColor(Color.blue)'], t3)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)', 'And(FilterByDegree(DEGREE.2), FilterByNeighborSize(SIZE.8))', 'And(FilterByNeighborSize(SIZE.MAX), Or(FilterBySize(SIZE.5), FilterBySize(SIZE.8)))', 'FilterBySize(SIZE.MAX)'], f3)

        t27, f27 = run_synthesis("6455b5f5", "ccg")
        self.assertCountEqual(['updateColor(Color.blue)', 'updateColor(Color.cyan)', 'NoOp'], t27)
        self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)', 'Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))'], f27)

        t4, f4 = run_synthesis("67385a82", "nbccg")
        self.assertCountEqual(['updateColor(Color.cyan)', 'NoOp'], t4)
        self.assertCountEqual(['Not(FilterBySize(SIZE.MIN))', 'FilterBySize(SIZE.MIN)'], f4)
        t5, f5 = run_synthesis("810b9b61", "ccgbr")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], t5)
        self.assertCountEqual(['FilterByNeighborDegree(DEGREE.1)', 'Not(FilterByNeighborDegree(DEGREE.1))'], f5)
        t6, f6 = run_synthesis("a5313dff", "ccgbr")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t6)
        self.assertCountEqual(['Or(FilterBySize(SIZE.6), FilterBySize(SIZE.8))', 'Or(FilterByColor(FColor.red), FilterBySize(SIZE.ODD))'], f6)
        t7, f7 = run_synthesis("d2abd087", "nbccg")
        self.assertCountEqual(['updateColor(Color.red)', 'updateColor(Color.blue)'], t7)
        self.assertCountEqual(['FilterBySize(SIZE.6)', 'Not(FilterBySize(SIZE.6))'], f7)
        t8, f8 = run_synthesis("6e82a1ae", "nbccg")
        self.assertCountEqual( ['updateColor(Color.red)', 'updateColor(Color.blue)', 'updateColor(Color.green)'], t8)
        self.assertCountEqual(['FilterBySize(SIZE.ODD)', 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f8)
        t9, f9 = run_synthesis("ea32f347", "nbccg")
        self.assertCountEqual(['updateColor(Color.yellow)', 'updateColor(Color.blue)', 'updateColor(Color.red)'], t9)
        self.assertCountEqual(['Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))', 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f9)

        #t10, f10 = run_synthesis("868de0fa", "nbccg") # todo
        #self.assertCountEqual(['fillRectangle(Color.orange, Overlap.TRUE)', 'fillRectangle(Color.red, Overlap.TRUE)'], t7)
        #self.assertCountEqual(['FilterByColor(FColor.blue)', 'FilterByColor(FColor.black)'], f7)
        t12, f12 = run_synthesis("aedd82e4", "nbccg")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t12)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)', 'Not(FilterBySize(SIZE.MIN))'], f12)
        t13, f13 = run_synthesis("c8f0f002", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t13)
        self.assertCountEqual(['Not(FilterByColor(FColor.orange))', 'FilterByColor(FColor.orange)'], f13)
        t14, f14 = run_synthesis("5582e5ca", "ccg")
        self.assertCountEqual(['updateColor(Color.most)'], t14)
        self.assertCountEqual(['FilterByNeighborSize(SIZE.MIN)'], f14)
        t15, f15 = run_synthesis("b1948b0a", "nbccg")
        self.assertCountEqual(['updateColor(Color.red)', 'NoOp'], t15)
        self.assertCountEqual(['FilterByColor(FColor.fuchsia)', 'FilterByColor(FColor.orange)'], f15)

        t11, f11 = run_synthesis("1e0a9b12", "nbccg")
        self.assertCountEqual(['[moveNodeMax(Dir.DOWN), moveNodeMax(Dir.DOWN)]'], t11)
        self.assertCountEqual(['Not(FilterByColor(FColor.black))'], f11)
        t16, f16 = run_synthesis("3c9b0459", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], t16)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f16)
        t17, f17 = run_synthesis("6150a2bd", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], t17)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f17)
        t18, f18 = run_synthesis("9dfd6313", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], t18)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f18)

        t19, f19 = run_synthesis("25ff71a9", "nbccg")
        self.assertCountEqual(['moveNode(Dir.DOWN)'], t19)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f19)
        t20, f20 = run_synthesis("67a3c6ac", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.HORIZONTAL)'], t20)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f20)
        t21, f21 = run_synthesis("74dd1130", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], t21)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f21)
        t22, f22 = run_synthesis("ed36ccf7", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CCW)'], t22)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f22)

        t24, f24 = run_synthesis("3906de3d", "nbvcg")
        self.assertCountEqual(['moveNodeMax(Dir.UP)'], t24)
        self.assertCountEqual(['Not(FilterByColor(FColor.black))'], f24)
        t23, f23 = run_synthesis("68b16354", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.VERTICAL)'], t23)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f23)
        t25, f25 = run_synthesis("a79310a0", "nbccg")
        self.assertCountEqual(['[updateColor(Color.red), moveNode(Dir.DOWN)]'], t25)
        self.assertCountEqual(['FilterByColor(FColor.cyan)'], f25)

        t26, f26 = run_synthesis("d037b0a7", "nbccg")
        self.assertCountEqual(['extendNode(Dir.DOWN, Overlap.TRUE)'], t26)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f26)
        t28, f28 = run_synthesis("b230c067", "nbccg")
        self.assertCountEqual(['updateColor(Color.red)', 'updateColor(Color.blue)'], t28)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)', 'FilterBySize(SIZE.MAX)'], f28)

        # next is t29, f29

if __name__ == "__main__":
    unittest.main()