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

# Transform Vocab
# tleaf_makers = [Color, NoOp(), Dir, Overlap, Rotation_Angle, RelativePosition, ImagePoints,
# Symmetry_Axis, ObjectId, Variable("Var"), UpdateColor, MoveNode, MoveNodeMax, AddBorder, ExtendNode, Mirror,
# HollowRectangle, Flip, Insert, RotateNode, FillRectangle, Transforms]
tleaf_makers = [Color, NoOp(), Dir, Overlap, Rotation_Angle, RelativePosition, ImagePoints,
                Symmetry_Axis, ObjectId, UpdateColor, MoveNode, MoveNodeMax, AddBorder, ExtendNode, Mirror,
                HollowRectangle, RotateNode, Flip, FillRectangle, Transforms]  # todo: add variable back after sequences fix!
f_vocabMakers = [FColor, Degree, Size, Relation, FilterByColor, FilterBySize, FilterByDegree, FilterByNeighborColor, FilterByNeighborSize,
                 FilterByNeighborDegree, Not, And, Or]


def run_synthesis(taskNumber, abstraction):
    store = {}
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
    input_graphs = [input_graph.graph.nodes(
        data=True) for input_graph in task.input_abstracted_graphs_original[task.abstraction]]
    enumerator = TSizeEnumerator(task, transform_vocab, ValuesManager())
    # has the entire unabstracted output graphs
    expected_graphs = [output.graph.nodes(
        data=True) for output in task.train_output]
    output_graphs = [output_graph.graph.nodes(
        data=True) for output_graph in task.output_abstracted_graphs_original[task.abstraction]]
    input_graph_dicts, output_graphs_dicts = [], []
    for node_data_view in input_graphs:  # per-task
        graph_dict = {}
        for node_key, node_info in node_data_view:
            for node in node_info['nodes']:
                graph_dict[node] = node_info['color']
        # only has the abstracted nodes from the input graphs
        input_graph_dicts.append(graph_dict)

    for node_data_view in output_graphs:  # per-task
        graph_dict = {}
        for node_key, node_info in node_data_view:
            for node in node_info['nodes']:
                graph_dict[node] = node_info['color']
        # only has the abstracted nodes from the output graphs
        output_graphs_dicts.append(graph_dict)

    aggregated_correct_nodes_per_task = [set() for _ in output_graphs_dicts]
    correct_nodes_per_transform = {}  # the dictionary from transforms to correct-nodes

    while enumerator.hasNext():
        program = enumerator.next()
        print("enumerator:", program.code)
        print("enumerator-vals:", [val.graph.nodes(data=True)
            for val in program.values])
        # transformed graph values of the current program enumerated
        blue_prints = [val.graph.nodes(data=True) for val in program.values]
        actual_values = program.values
        # correct nodes for the current program
        correct_nodes_for_this_transform = [set() for _ in actual_values]

        for task_idx, (train_in, actual_value, expected_graph, blue_print) in \
                enumerate(zip(task.train_input, actual_values, expected_graphs, blue_prints)):
            correct_nodes = set()  # to collect all correct nodes contributed by this transform

            # todo: enforce the invariant that only one operation can act on an object
            node_correctness_map = {}
            # check the blueprint for each transformed object
            for node_key, node_info in blue_print:
                node_set = set(node_info['nodes'])
                if isinstance(node_info['color'], list):
                    # zip pixels with their corresponding colors for multi-color objects
                    nodes_with_colors = zip(node_info['nodes'], node_info['color'])

                    # Check if the blueprint for the entire transformed object is correct
                    all_nodes_correct = all(
                        node in expected_graph and expected_graph[node]['color'] == color
                        for node, color in nodes_with_colors
                    )
                else:
                    all_nodes_correct = all(
                        node in expected_graph and expected_graph[node]['color'] == node_info['color']
                        for node in node_set
                )
                for node in node_set:
                    node_correctness_map[node] = all_nodes_correct

            actual_value = train_in.undo_abstraction(
                actual_value).graph.nodes(data=True)

            correct_nodes = [
                node for node, node_info in actual_value if (node in expected_graph and expected_graph[node]['color'] == node_info['color']) and
                (not node in node_correctness_map or node_correctness_map[node])]

            if correct_nodes:
                correct_nodes_for_this_transform[task_idx] = correct_nodes
                aggregated_correct_nodes_per_task[task_idx].update(
                    correct_nodes)

        if any(correct_nodes_for_this_transform):
            correct_nodes_per_transform[program.code] = correct_nodes_for_this_transform
        full_coverage_per_task = []
        full_coverage_per_task = [set(aggregated_correct_nodes) == set(dict(expected_graphs[task_idx]).keys())
                                for task_idx, aggregated_correct_nodes in enumerate(aggregated_correct_nodes_per_task)]

        # & each object in the input node has a unique transformation mapped to it!
        if all(full_coverage_per_task):
            print(correct_nodes_per_transform)
            minimal_transforms = set()
            for task_index in range(len(actual_values)):
                uncovered_nodes = set(dict(expected_graphs[task_index]).keys())
                # maps transform to its coverage for this specific task
                task_specific_transforms = {}

                # fill task-specific coverage based on correct_nodes_per_transform
                for transform, coverage_lists in correct_nodes_per_transform.items():
                    # check if there's coverage for this task
                    if coverage_lists[task_index]:
                        task_specific_transforms[transform] = set(
                            coverage_lists[task_index])

                # greedy selection for this task
                while uncovered_nodes:
                    best_transform = None
                    best_coverage = set()
                    for transform, coverage in task_specific_transforms.items():
                        current_coverage = coverage & uncovered_nodes
                        if len(current_coverage) > len(best_coverage):
                            best_transform = transform
                            best_coverage = current_coverage

                    if not best_transform:
                        break  # all nodes covered or no suitable transform found

                    # add the best transform to the minimal set
                    minimal_transforms.add(best_transform)
                    uncovered_nodes -= best_coverage  # update uncovered nodes

            print("Minimal Transforms:", minimal_transforms)
            return minimal_transforms, []

        # todo: transform synthesis: we have to save the input object id + output pixels cos of movenode :/
        # todo: filter synthesis over pairs
        # todo: filter synthesis over subsets


# {"ded97339": "nbccg"} #{"4093f84a": "nbccg"} #{"ae3edfdc": "nbccg"} # 3618c87e, 868de0fa 1
evals = {"ed36ccf7": "na"} #{"3c9b0459": "na"} {"1e0a9b12": "nbccg"}

for task, abstraction in evals.items():
    start_time = time.time()
    print("taskNumber:", task)
    transformations, filters = run_synthesis(task, abstraction)
    print("transformations:", transformations)
    print("filters:", filters)
    print(f"Problem {task}: --- {(time.time() - start_time)} seconds ---")


class TestEvaluation(unittest.TestCase):
    def all_problems(self):
        print("==================================================COLORING PROBLEMS==================================================")
        print("Solving problem d511f180")
        t0, f0 = run_synthesis("d511f180", "nbccg")
        self.assertCountEqual(
            ['NoOp', 'updateColor(Color.grey)', 'updateColor(Color.cyan)'], t0)
        # self.assertCountEqual(
        # ['Not(Or(FilterByColor(FColor.grey), FilterByColor(FColor.cyan)))', 'FilterByColor(FColor.cyan)', 'FilterByColor(FColor.grey)'], f0)

        print("Solving problem 00d62c1b")
        t1, f1 = run_synthesis("00d62c1b", "ccgbr")
        self.assertCountEqual(['NoOp', 'updateColor(Color.yellow)'], t1)
        # self.assertCountEqual(
        # ['FilterByColor(FColor.green)', 'FilterByColor(FColor.black)'], f1)

        t2, f2 = run_synthesis("9565186b", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t2)
        # self.assertCountEqual(
        # ['FilterByColor(FColor.most)', 'FilterByNeighborSize(SIZE.MAX)'], f2)

        print("Solving problem b230c067")
        t3, f3 = run_synthesis("b230c067", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t3)
        # self.assertCountEqual(
        # ['FilterBySize(SIZE.MIN)', 'FilterBySize(SIZE.MAX)'], f3)

        print("Solving problem 08ed6ac7")
        t4, f4 = run_synthesis("08ed6ac7", "nbccg")
        self.assertCountEqual(['updateColor(Color.yellow)', 'updateColor(Color.green)',
                            'updateColor(Color.red)', 'updateColor(Color.blue)'], t4)
        # self.assertCountEqual(['FilterBySize(SIZE.MIN)', 'And(FilterByDegree(DEGREE.2), FilterByNeighborSize(SIZE.8))',
        # 'And(FilterByNeighborSize(SIZE.MAX), Or(FilterBySize(SIZE.5), FilterBySize(SIZE.8)))', 'FilterBySize(SIZE.MAX)'], f4)

        print("Solving problem 6455b5f5")
        t5, f5 = run_synthesis("6455b5f5", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.blue)', 'updateColor(Color.cyan)', 'NoOp'], t5)
        # self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)',
        # 'Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))'], f5)

        print("Solving problem 67385a82")
        t7, f7 = run_synthesis("67385a82", "nbccg")
        self.assertCountEqual(['updateColor(Color.cyan)', 'NoOp'], t7)
        # self.assertCountEqual(
        # ['Not(FilterBySize(SIZE.MIN))', 'FilterBySize(SIZE.MIN)'], f7)

        print("Solving problem 810b9b61")
        t8, f8 = run_synthesis("810b9b61", "ccgbr")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], t8)
        # self.assertCountEqual(
        # ['FilterByNeighborDegree(DEGREE.1)', 'Not(FilterByNeighborDegree(DEGREE.1))'], f8)

        print("Solving problem a5313dff")
        t9, f9 = run_synthesis("a5313dff", "ccgbr")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t9)
        # self.assertCountEqual(['Or(FilterBySize(SIZE.6), FilterBySize(SIZE.8))',
        # 'Or(FilterByColor(FColor.red), FilterBySize(SIZE.ODD))'], f9)

        print("Solving problem d2abd087")
        t10, f10 = run_synthesis("d2abd087", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t10)
        # self.assertCountEqual(
        # ['FilterBySize(SIZE.6)', 'Not(FilterBySize(SIZE.6))'], f10)

        print("Solving problem 6e82a1ae")
        t11, f11 = run_synthesis("6e82a1ae", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)', 'updateColor(Color.green)'], t11)
        # self.assertCountEqual(
        # ['FilterBySize(SIZE.ODD)', 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f11)

        print("Solving problem 67385a82")
        t7, f7 = run_synthesis("67385a82", "nbccg")
        self.assertCountEqual(['updateColor(Color.cyan)', 'NoOp'], t7)
        # self.assertCountEqual(
        # ['Not(FilterBySize(SIZE.MIN))', 'FilterBySize(SIZE.MIN)'], f7)

        print("Solving problem 810b9b61")
        t8, f8 = run_synthesis("810b9b61", "ccgbr")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], t8)
        # self.assertCountEqual(
        #    ['FilterByNeighborDegree(DEGREE.1)', 'Not(FilterByNeighborDegree(DEGREE.1))'], f8)

        print("Solving problem a5313dff")
        t9, f9 = run_synthesis("a5313dff", "ccgbr")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t9)
        # self.assertCountEqual(['Or(FilterBySize(SIZE.6), FilterBySize(SIZE.8))',
        # 'Or(FilterByColor(FColor.red), FilterBySize(SIZE.ODD))'], f9)

        print("Solving problem d2abd087")
        t10, f10 = run_synthesis("d2abd087", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t10)
        # self.assertCountEqual(
        # ['FilterBySize(SIZE.6)', 'Not(FilterBySize(SIZE.6))'], f10)

        print("Solving problem ea32f347")
        t12, f12 = run_synthesis("ea32f347", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.yellow)', 'updateColor(Color.blue)', 'updateColor(Color.red)'], t12)
        # self.assertCountEqual(['Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))',
        # 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f12)

        print("Solving problem aabf363d")
        t14, f14 = run_synthesis("aabf363d", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.yellow)', 'updateColor(Color.fuchsia)'], t14)
        # self.assertCountEqual(['Not(FilterBySize(SIZE.12))', 'FilterByColor(FColor.red)', 'FilterByColor(FColor.green)'], f14)

        print("Solving problem c0f76784")
        t15, f15 = run_synthesis("c0f76784", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.orange, Overlap.TRUE)',
                            'fillRectangle(Color.fuchsia, Overlap.TRUE)', 'fillRectangle(Color.cyan, Overlap.TRUE)'], t15)
        # self.assertCountEqual(['FilterBySize(SIZE.12)', 'FilterBySize(SIZE.8)', 'FilterBySize(SIZE.MAX)'], f15)

        print("Solving problem d5d6de2d")
        t16, f16 = run_synthesis("d5d6de2d", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.green)'], t16)
        # self.assertCountEqual(['Or(FilterByColor(FColor.red), FilterBySize(SIZE.MAX))', 'Not(Or(FilterByColor(FColor.red), FilterBySize(SIZE.MAX)))'], f16)

        print("Solving problem 25d8a9c8")
        t17, f17 = run_synthesis("25d8a9c8", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.grey)', 'updateColor(Color.black)'], t17)
        # self.assertCountEqual(['And(FilterBySize(SIZE.3), Not(FilterByColor(FColor.green)))',
        # 'Or(FilterByColor(FColor.green), Not(FilterBySize(SIZE.3)))'], f17)

        print("Solving problem aedd82e4")
        t18, f18 = run_synthesis("aedd82e4", "nbccg")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t18)
        # self.assertCountEqual(
        # ['FilterBySize(SIZE.MIN)', 'Not(FilterBySize(SIZE.MIN))'], f18)

        print("Solving problem c8f0f002")
        t19, f19 = run_synthesis("c8f0f002", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t19)
        # self.assertCountEqual(
        # ['Not(FilterByColor(FColor.orange))', 'FilterByColor(FColor.orange)'], f19)

        print("Solving problem 5582e5ca")  # todo-eusolver
        t21, f21 = run_synthesis("5582e5ca", "ccg")
        # self.assertCountEqual(['updateColor(Color.most)'], t21)
        # self.assertCountEqual(['FilterByNeighborSize(SIZE.MIN)'], f21)

        print("Solving problem b1948b0a")
        t22, f22 = run_synthesis("b1948b0a", "nbccg")
        self.assertCountEqual(['updateColor(Color.red)', 'NoOp'], t22)
        # self.assertCountEqual(
        # ['FilterByColor(FColor.fuchsia)', 'FilterByColor(FColor.orange)'], f22)

        print("Solving problem a61f2674")
        t23, f23 = run_synthesis("a61f2674", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.blue)', 'updateColor(Color.red)'], t23)
        # self.assertCountEqual(
        # ['Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))', 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f23)

        print("Solving problem 7f4411dc")
        t24, f24 = run_synthesis("7f4411dc", "lrg")
        self.assertCountEqual(['updateColor(Color.black)', 'NoOp'], t24)

        # 868de0fa -- nbccg
        # ddf7fa4f -- nbccg
        # 1e0a9b12 -- nbccg
        print("==================================================MOVEMENT PROBLEMS==================================================")
        print("Solving problem 25ff71a9")
        mt7, mf7 = run_synthesis("25ff71a9", "nbccg")
        self.assertCountEqual(['moveNode(Dir.DOWN)'], mt7)
        # self.assertCountEqual(['FilterByColor(FColor.least)'], mf7)

        print("Solving problem 3c9b0459")
        mt1, mf1 = run_synthesis("3c9b0459", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt1)
        #self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf1)

        print("Solving problem 6150a2bd")
        mt2, mf2 = run_synthesis("6150a2bd", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt2)
        #self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf2)

        print("Solving problem 9dfd6313")
        mt3, mf3 = run_synthesis("9dfd6313", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt3)
        #self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf3)

        print("Solving problem 67a3c6ac")
        mt8, mf8 = run_synthesis("67a3c6ac", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.HORIZONTAL)'], mt8)
        #self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf8)

        print("Solving problem 74dd1130")
        mt9, mf9 = run_synthesis("74dd1130", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt9)
        #self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf9)

        print("Solving problem ed36ccf7")
        mt10, mf10 = run_synthesis("ed36ccf7", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CCW)'], mt10)
        #self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf10)

        print("Solving problem 68b16354")
        t23, f23 = run_synthesis("68b16354", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.VERTICAL)'], t23)
        #self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f23)

        print("Solving problem a79310a0")
        mt3, mf3 = run_synthesis("a79310a0", "nbccg")
        self.assertCountEqual(
            ['[updateColor(Color.red), moveNode(Dir.DOWN)]'], mt3)
        # self.assertCountEqual(['FilterByColor(FColor.cyan)'], mf3)

        print("Solving problem 3906de3d")
        t24, f24 = run_synthesis("3906de3d", "nbvcg")
        self.assertCountEqual(['moveNodeMax(Dir.UP)'], t24)
        # self.assertCountEqual(['Not(FilterByColor(FColor.black))'], f24)

        print("==================================================AUGMENTATION PROBLEMS==================================================")
        print("Solving problem bb43febb")
        at0, af0 = run_synthesis("bb43febb", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.red)'], at0)
        # self.assertCountEqual(['FilterByColor(FColor.grey)'], af0)

        print("Solving problem 4258a5f9")
        at1, af1 = run_synthesis("4258a5f9", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)'], at1)
        # self.assertCountEqual(['FilterByColor(FColor.grey)'], af1)

        print("Solving problem b27ca6d3")
        at5, af5 = run_synthesis("b27ca6d3", "nbccg")
        self.assertCountEqual(['addBorder(Color.green)', 'NoOp'], at5)
        # self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], af5)

        print("Solving problem d037b0a7")  # todo-prob!
        # at6, af6 = run_synthesis("d037b0a7", "nbccg")
        # self.assertCountEqual(['extendNode(Dir.DOWN, Overlap.TRUE)'], at6)
        # self.assertCountEqual(['FilterBySize(SIZE.MIN)'], af6)

        print("Solving problem dc1df850")
        at12, af12 = run_synthesis("dc1df850", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)', 'NoOp'], at12)
        # self.assertCountEqual(['FilterByColor(FColor.red)', 'Not(FilterByColor(FColor.red))'], af12)

        print("Solving problem 4347f46a")
        at4, af4 = run_synthesis("4347f46a", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.black)'], at4)
        # self.assertCountEqual(['Not(FilterByColor(FColor.blue))'], af4)

        print("Solving problem 3aa6fb7a")
        at10, af10 = run_synthesis("3aa6fb7a", "nbccg")
        self.assertCountEqual(
            ['fillRectangle(Color.blue, Overlap.TRUE)'], at10)
        # self.assertCountEqual(['FilterByColor(FColor.cyan)'], af10)

        print("Solving problem 6d75e8bb")
        at13, af13 = run_synthesis("6d75e8bb", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.red, Overlap.TRUE)'], at13)
        # self.assertCountEqual(['FilterByColor(FColor.cyan)'], af13)

        print("Solving problem 913fb3ed")
        at14, af14 = run_synthesis("913fb3ed", "nbccg")
        self.assertCountEqual(
            ['addBorder(Color.blue)', 'addBorder(Color.yellow)', 'addBorder(Color.fuchsia)'], at14)
        # self.assertCountEqual(['FilterByColor(FColor.red)', 'FilterByColor(FColor.cyan)', 'FilterByColor(FColor.green)'], af14)

        print("Solving problem e8593010")
        at15, af15 = run_synthesis("e8593010", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.green)', 'updateColor(Color.blue)', 'updateColor(Color.red)', 'NoOp'], at15)
        # self.assertCountEqual(['And(FilterByColor(FColor.black), FilterBySize(SIZE.MIN))', 'FilterBySize(SIZE.3)', 'And(FilterByColor(FColor.black), FilterBySize(SIZE.2))', 'FilterByColor(FColor.grey)'], af15)

        print("Solving problem 50cb2852")
        at16, af16 = run_synthesis("50cb2852", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.cyan)'], at16)

        print("Solving problem 694f12f3")
        at17, af17 = run_synthesis("694f12f3", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.red)', 'hollowRectangle(Color.blue)'], at17)

if __name__ == "__main__":
    unittest.main()
