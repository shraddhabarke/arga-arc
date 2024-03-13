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
tleaf_makers = [Color, NoOp(), Dir, Overlap, Rotation_Angle, RelativePosition, ImagePoints,
                Symmetry_Axis, ObjectId, Variable('Var'), UpdateColor, MoveNode, MoveNodeMax, ExtendNode, AddBorder, Mirror,
                HollowRectangle, RotateNode, Flip, FillRectangle, Transforms]
# todo: add variable back after sequences fix! Insert
f_vocabMakers = [FColor, Int, Degree, Height, Width, Size, Shape, Column, IsDirectNeighbor, IsDiagonalNeighbor, IsAnyNeighbor, FilterByColor, FilterBySize, FilterByShape, FilterByDegree, FilterByHeight,
                FilterByColumns, FilterByNeighborColor, FilterByNeighborSize, FilterByNeighborDegree, Not, And, Or, VarAnd]

#f_vocabMakers = [Column, FilterByColumns, Not]
def filter_compare(results, subset):
    if len(results) != len(subset):
        return False

    for res_dict, sub_dict in zip(results, subset):
        if set(res_dict.keys()) != set(sub_dict.keys()):
            return False
        for key in sub_dict:
            # Ensure res_dict[key] values are a superset or equal to sub_dict[key] values
            if not res_dict[key] and not sub_dict[key]:
                continue  # Both are empty, so this is fine
            elif res_dict[key] and sub_dict[key]:
                if not set(res_dict[key]).issubset(set(sub_dict[key])):
                    return False
            else:
                return False
    return True

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

    transform_vocab = VocabFactory.create(tleaf_makers)
    input_graphs = [input_graph.graph.nodes(
        data=True) for input_graph in task.input_abstracted_graphs_original[task.abstraction]]
    print("input-graphs:", [input_graph.graph.nodes() for input_graph in task.input_abstracted_graphs_original[task.abstraction]])
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

    def find_input_nodes(input_graphs, correct_pixels):
        """
        Collect the input objects corresponding to the pixels covered by the transform
        """
        input_nodes = []
        for input_graph, correct_pixel in zip(input_graphs, correct_pixels):
            task_input_nodes = []
            for key, value in input_graph:
                if any(node in value['nodes'] for node in correct_pixel):
                #if all(node in abstract_list for node in value['nodes']):
                    task_input_nodes.append(key)
            input_nodes.append(task_input_nodes)
        return input_nodes

    def synthesize_filter(subset, timeout: int = 2):
        filter_vocab = VocabFactory.create(f_vocabMakers)
        enumerator = FSizeEnumerator(task, filter_vocab, ValuesManager())
        i = 0
        while enumerator.hasNext():
            program = enumerator.next()
            i += 1
            results = program.values
            #print(f"Program: {program.code}: {results, program.size}")
            print("subset:", subset)
            print("results:", results)
            if filter_compare(results, subset):
                return program.code
            
    correct_transforms = set()
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
            correct_nodes = set()  # to collect all correct pixels contributed by this transform
            node_correctness_map, all_blue_print = {}, []
            blue_print = {key: value for key, value in blue_print}

            for node_key, node_info in blue_print.items():
                if isinstance(node_info['color'], list):
                    graph_dict = dict(
                        zip(node_info['nodes'], node_info['color']))
                    all_blue_print.append(graph_dict)
                else:
                    graph_dict = {}
                    for node in node_info['nodes']:
                        graph_dict[node] = node_info['color']
                    node_id = (node_key[0], node_key[1], 0)
                    if node_id in blue_print.keys():
                        for node in blue_print[node_id]['nodes']:
                            graph_dict[node] = blue_print[node_id]['color']
                    all_blue_print.append(graph_dict)

            # check the blueprint for each transformed object
            for blue_print in all_blue_print:
                node_set = set(blue_print.keys())
                for node, color in blue_print.items():
                    all_nodes_correct = all(
                        node in expected_graph and expected_graph[node]['color'] == blue_print[node]
                        for node in node_set)

                for node in node_set:
                    node_correctness_map[node] = all_nodes_correct
            print("node_correctness_map:", node_correctness_map)
            actual_value = train_in.undo_abstraction(
                actual_value).graph.nodes(data=True)

            correct_nodes = [
                node for node, node_info in actual_value if (node in expected_graph and expected_graph[node]['color'] == node_info['color']) and
                (not node in node_correctness_map or node_correctness_map[node])]
            print("correct_nodes:", correct_nodes)
            if correct_nodes:
                correct_nodes_for_this_transform[task_idx] = correct_nodes
                aggregated_correct_nodes_per_task[task_idx].update(correct_nodes)

        if any(val for val in correct_nodes_for_this_transform):
            correct_nodes_per_transform[program.code] = correct_nodes_for_this_transform
        full_coverage_per_task = [set(aggregated_correct_nodes) == set(dict(expected_graphs[task_idx]).keys())
                                for task_idx, aggregated_correct_nodes in enumerate(aggregated_correct_nodes_per_task)]

        if all(full_coverage_per_task):
            print("New Coverage being Considered")
            minimal_transforms = set()
            # Calculate total coverage size for each transform across all tasks
            total_coverage_size_per_transform = {}
            for transform, coverage_lists in correct_nodes_per_transform.items():
                total_size = sum(len(set(coverage)) for coverage in coverage_lists if coverage)
                total_coverage_size_per_transform[transform] = total_size

            sorted_transforms_by_total_coverage = sorted(total_coverage_size_per_transform.items(), key=lambda x: x[1], reverse=True)
            for task_index in range(len(actual_values)):
                uncovered_nodes = set(dict(expected_graphs[task_index]).keys())

                # Greedy selection for this task, considering transforms in order of their overall coverage
                task_specific_transforms = {}
                for transform, _ in sorted_transforms_by_total_coverage:
                    coverage_lists = correct_nodes_per_transform[transform]
                    if coverage_lists[task_index]:
                        task_specific_transforms[transform] = set(coverage_lists[task_index])

                # greedy selection for this task
                while uncovered_nodes:
                    best_transform = None
                    best_coverage = set()
                    for transform, _ in sorted_transforms_by_total_coverage:
                        if transform in total_coverage_size_per_transform and correct_nodes_per_transform[transform][task_index]:
                            current_coverage = set(correct_nodes_per_transform[transform][task_index]) & uncovered_nodes
                            if len(current_coverage) > len(best_coverage):
                                best_transform = transform
                                best_coverage = current_coverage

                    if not best_transform:
                        break  # All nodes covered or no suitable transform found

                    # Add the best transform to the minimal set
                    minimal_transforms.add(best_transform)
                    uncovered_nodes -= best_coverage  # Update uncovered nodes

            print("minimal-transform:", minimal_transforms)
            minimal_transforms = list(minimal_transforms)
            if minimal_transforms != correct_transforms:
                correct_transforms = minimal_transforms
                # Candidate set of transforms found: initiating filter synthesis...
                all_filters_found, filters_sol = True, []
                for program in minimal_transforms:
                    print("Enumerating filters for:", program)
                    if "Var" not in program:
                        input_nodes = find_input_nodes(input_graphs, correct_nodes_per_transform[program])
                        print("input-nodes:", input_nodes)
                        subset = [{tup: [] for tup in subset} for subset in input_nodes]
                        print("Subset:", subset)
                    else:
                        print("SPEC!", task.spec)
                        subset = task.spec
                    filters = synthesize_filter(subset)
                    if filters:
                        filters_sol.append(filters)
                    else:
                        all_filters_found = False
                        break
                if all_filters_found:
                    print("Transformation Solution:", [
                        program for program in minimal_transforms])
                    print("Filter Solution", [
                        program for program in filters_sol])
                    return [program for program in minimal_transforms], [
                    program for program in filters_sol]

#4093f84a, 7e0986d6, ExtendNode --> 7ddcd7ec, dbc1a6ce

evals = {}
# todo: add insert 3618c87e

for task, abstraction in evals.items():
    start_time = time.time()
    print("taskNumber:", task)
    transformations, filters = run_synthesis(task, abstraction)
    print("transformations:", transformations)
    print("filters:", filters)
    print(f"Problem {task}: --- {(time.time() - start_time)} seconds ---")

class TestEvaluation(unittest.TestCase):
    def test_all_problems(self):
        print("==================================================VARIABLE PROBLEMS==================================================")
        # there is one correct assignment for the variables and the filters should convey that

        print("Solving problem 6855a6e4")
        vt0, vf0 = run_synthesis("6855a6e4", "nbccg")
        self.assertCountEqual(['mirror(Var.mirror_axis)'], vt0)
        self.assertCountEqual(['And(FilterByColor(FColor.grey), VarAnd(Var.IsDirectNeighbor, Var.FilterByColor(FColor.red)))'], vf0)

        print("Solving problem ddf7fa4f")
        vt1, vf1 = run_synthesis("ddf7fa4f", "nbccg")
        self.assertCountEqual(['updateColor(Var.color)'], vt1)
        self.assertCountEqual(['And(FilterByColor(FColor.grey), VarAnd(Var.IsDirectNeighbor, Var.FilterBySize(SIZE.MIN)))'], vf1)

        print("Solving problem f8a8fe49")
        #vt2, vf2 = run_synthesis("f8a8fe49", "nbccg")
        #self.assertCountEqual(['mirror(Var.mirror_axis)'], vt2)
        #self.assertCountEqual(['And(FilterByColor(FColor.grey), VarAnd(Var.IsNeighbor, Var.FilterByColor(FColor.red)))'], vf2)

        print("Solving problem dc433765")
        vt3, vf3 = run_synthesis("dc433765", "nbccg")
        self.assertCountEqual(['moveNode(Var.direction)'], vt3)
        self.assertCountEqual(['And(FilterByColor(FColor.green), VarAnd(Var.IsAnyNeighbor, Var.FilterByColor(FColor.yellow)))'], vf3)

        print("Solving problem a48eeaf7")
        vt6, vf6 = run_synthesis("a48eeaf7", "nbccg")
        self.assertCountEqual(['moveNodeMax(Var.direction)'], vt6)
        self.assertCountEqual(['And(FilterByColor(FColor.grey), VarAnd(Var.IsAnyNeighbor, Var.FilterByColor(FColor.red)))'], vf6)

        print("Solving problem ae3edfdc")
        vt4, vf4 = run_synthesis("ae3edfdc", "nbccg")
        self.assertCountEqual(['moveNodeMax(Var.direction)'], vt4)
        self.assertCountEqual(['And(Not(FilterByNeighborDegree(DEGREE.1)), VarAnd(Var.IsDirectNeighbor, Var.FilterByNeighborDegree(DEGREE.1)))'], vf4)

        print("Solving problem d43fd935")
        vt5, vf5 = run_synthesis("d43fd935", "nbccg")
        self.assertCountEqual(['extendNode(Var.direction, Overlap.TRUE)'], vt5)
        self.assertCountEqual(['And(FilterByNeighborSize(SIZE.MAX), VarAnd(Var.IsDirectNeighbor, Var.FilterByColor(FColor.green)))'], vf5)

        print("Solving problem 2c608aff")
        vt6, vf6 = run_synthesis("2c608aff", "ccgbr")
        self.assertCountEqual(['extendNode(Var.direction, Overlap.TRUE)'], vt5)
        self.assertCountEqual(['And(FilterByNeighborSize(SIZE.MAX), VarAnd(Var.IsDirectNeighbor, Var.FilterBySize(SIZE.MAX)))'], vf6)

        print("Solving problem ded97339")
        #vt8, vf8 = run_synthesis("ded97339", "nbccg") #todo
        #self.assertCountEqual(['extendNode(Var.direction, Overlap.TRUE)'], vt8)

        print("Solving problem 05f2a901")
        #vt7, vf7 = run_synthesis("05f2a901", "nbccg")
        #self.assertCountEqual(['moveNodeMax(Var.direction)'], vt7) #todo - eusolver
        #self.assertCountEqual(['moveNodeMax(Dir.UP)', 'moveNodeMax(Dir.DOWN)', 'NoOp', 'moveNodeMax(Dir.RIGHT)'], vt7)
        #self.assertCountEqual(['FilterByColor(FColor.red)', 'FilterByColor(FColor.red)', 'FilterByColor(FColor.cyan)', 'FilterByColor(FColor.red)'], vf7)
        print("==================================================COLORING PROBLEMS==================================================")
        print("Solving problem d23f8c26")
        ct0, cf0 = run_synthesis("d23f8c26", "nbccg")
        self.assertCountEqual(['updateColor(Color.black)', 'NoOp'], ct0)
        self.assertCountEqual(['Not(FilterByColumns(COLUMN.CENTER))', 'FilterByColumns(COLUMN.CENTER)'], cf0)

        print("Solving problem a5f85a15") # even columns
        ct1, cf1 = run_synthesis("a5f85a15", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.yellow)'], ct1)
        self.assertCountEqual(['FilterByColumns(COLUMN.EVEN)', 'Not(FilterByColumns(COLUMN.EVEN))'], cf1)

        print("Solving problem ba26e723")

        print("Solving problem b2862040")
        #ct2, cf2 = run_synthesis("b2862040", "nbccg")
        #self.assertCountEqual([], ct1)
        #self.assertCountEqual([], cf1)

        print("Solving problem 810b9b61")
        ct2, cf2 = run_synthesis("810b9b61", "nbccg")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], ct2)
        self.assertCountEqual(['FilterByShape(Shape.enclosed)', 'Not(FilterByShape(Shape.enclosed))'], cf2)

        print("Solving problem f76d97a5")
        t0, f0 = run_synthesis("f76d97a5", "nbccg")
        self.assertCountEqual(['updateColor(Color.yellow)', 'updateColor(Color.black)', 'updateColor(Color.most)'], t0)
        self.assertCountEqual(['FilterBySize(SIZE.5)', 'Not(FilterByColor(FColor.grey))', 'And(FilterByColor(FColor.grey), FilterByColor(FColor.least))'], f0)

        print("Solving problem d511f180")
        t0, f0 = run_synthesis("d511f180", "nbccg")
        self.assertCountEqual(
            ['NoOp', 'updateColor(Color.grey)', 'updateColor(Color.cyan)'], t0)
        self.assertCountEqual(
        ['Not(Or(FilterByColor(FColor.grey), FilterByColor(FColor.cyan)))', 'FilterByColor(FColor.cyan)', 'FilterByColor(FColor.grey)'], f0)

        print("Solving problem 00d62c1b")
        t1, f1 = run_synthesis("00d62c1b", "ccgbr")
        self.assertCountEqual(['NoOp', 'updateColor(Color.yellow)'], t1)
        self.assertCountEqual(
        ['FilterByColor(FColor.green)', 'FilterByColor(FColor.black)'], f1)

        print("Solving problem 9565186b")
        t2, f2 = run_synthesis("9565186b", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t2)
        self.assertCountEqual(['FilterByColor(FColor.most)', 'FilterByNeighborSize(SIZE.MAX)'], f2)

        print("Solving problem b230c067")
        t3, f3 = run_synthesis("b230c067", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t3)
        self.assertCountEqual(
        ['FilterBySize(SIZE.MIN)', 'FilterBySize(SIZE.MAX)'], f3)

        print("Solving problem 08ed6ac7")
        t4, f4 = run_synthesis("08ed6ac7", "nbccg")
        self.assertCountEqual(['updateColor(Color.yellow)', 'updateColor(Color.green)',
                            'updateColor(Color.red)', 'updateColor(Color.blue)'], t4)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)',
        'And(FilterByDegree(DEGREE.2), FilterByNeighborSize(SIZE.8))',
        'And(FilterByNeighborSize(SIZE.MAX), Or(FilterBySize(SIZE.5), FilterBySize(SIZE.8)))',
        'FilterBySize(SIZE.MAX)'], f4)

        print("Solving problem 6455b5f5")
        t5, f5 = run_synthesis("6455b5f5", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.blue)', 'updateColor(Color.cyan)', 'NoOp'], t5)
        self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)',
        'Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))'], f5)

        print("Solving problem 67385a82")
        t7, f7 = run_synthesis("67385a82", "nbccg")
        self.assertCountEqual(['updateColor(Color.cyan)', 'NoOp'], t7)
        self.assertCountEqual(
        ['Not(FilterBySize(SIZE.MIN))', 'FilterBySize(SIZE.MIN)'], f7)

        print("Solving problem 810b9b61")
        t8, f8 = run_synthesis("810b9b61", "ccgbr")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], t8)
        self.assertCountEqual(
        ['FilterByShape(Shape.enclosed)', 'Not(FilterByShape(Shape.enclosed))'], f8)

        print("Solving problem a5313dff")
        t9, f9 = run_synthesis("a5313dff", "ccgbr")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t9)
        self.assertCountEqual(['FilterByHeight(HEIGHT.3)', 'Not(FilterByHeight(HEIGHT.3))'], f9)

        print("Solving problem d2abd087")
        t10, f10 = run_synthesis("d2abd087", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t10)
        self.assertCountEqual(['FilterBySize(SIZE.6)', 'Not(FilterBySize(SIZE.6))'], f10)

        print("Solving problem 6e82a1ae")
        t11, f11 = run_synthesis("6e82a1ae", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)', 'updateColor(Color.green)'], t11)
        self.assertCountEqual(['FilterBySize(SIZE.ODD)', 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f11)

        print("Solving problem ea32f347")
        t12, f12 = run_synthesis("ea32f347", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.yellow)', 'updateColor(Color.blue)', 'updateColor(Color.red)'], t12)
        self.assertCountEqual(['Or(FilterBySize(SIZE.5), FilterByHeight(HEIGHT.4))',
        'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f12)

        print("Solving problem aabf363d")
        t14, f14 = run_synthesis("aabf363d", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.yellow)', 'updateColor(Color.fuchsia)'], t14)
        self.assertCountEqual(['Not(FilterBySize(SIZE.12))', 'FilterByColor(FColor.red)', 'FilterByColor(FColor.green)'], f14)

        print("Solving problem c0f76784")
        t15, f15 = run_synthesis("c0f76784", "nbccg")
        self.assertCountEqual(['NoOp', 'fillRectangle(Color.orange, Overlap.TRUE)',  # todo: post-process
                            'fillRectangle(Color.fuchsia, Overlap.TRUE)', 'fillRectangle(Color.cyan, Overlap.TRUE)'], t15)
        self.assertCountEqual(['FilterByColor(FColor.grey)', 'FilterBySize(SIZE.12)', 'FilterBySize(SIZE.8)', 'FilterBySize(SIZE.MAX)'], f15)

        print("Solving problem d5d6de2d")
        t16, f16 = run_synthesis("d5d6de2d", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.green)'], t16)
        self.assertCountEqual(['Or(FilterByColor(FColor.red), FilterBySize(SIZE.MAX))', 'Not(Or(FilterByColor(FColor.red), FilterBySize(SIZE.MAX)))'], f16)

        print("Solving problem aedd82e4")
        t18, f18 = run_synthesis("aedd82e4", "nbccg")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t18)
        self.assertCountEqual(
        ['FilterBySize(SIZE.MIN)', 'Not(FilterBySize(SIZE.MIN))'], f18)

        print("Solving problem c8f0f002")
        t19, f19 = run_synthesis("c8f0f002", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t19)
        self.assertCountEqual(
        ['Not(FilterByColor(FColor.orange))', 'FilterByColor(FColor.orange)'], f19)

        print("Solving problem 5582e5ca")  # todo-eusolver
        t21, f21 = run_synthesis("5582e5ca", "ccg")
        # self.assertCountEqual(['updateColor(Color.most)'], t21)
        # self.assertCountEqual(['FilterByNeighborSize(SIZE.MIN)'], f21)

        print("Solving problem b1948b0a")
        t22, f22 = run_synthesis("b1948b0a", "nbccg")
        self.assertCountEqual(['updateColor(Color.red)', 'NoOp'], t22)
        self.assertCountEqual(['FilterByColor(FColor.fuchsia)', 'FilterByColor(FColor.orange)'], f22)

        print("Solving problem a61f2674")
        t23, f23 = run_synthesis("a61f2674", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.blue)', 'updateColor(Color.red)'], t23)
        self.assertCountEqual(
        ['Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))', 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f23)

        print("Solving problem 25d8a9c8")
        t17, f17 = run_synthesis("25d8a9c8", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.grey)', 'updateColor(Color.black)'], t17)
        self.assertCountEqual(['And(FilterBySize(SIZE.3), FilterByHeight(HEIGHT.MIN))',
        'Not(And(FilterBySize(SIZE.3), FilterByHeight(HEIGHT.MIN)))'], f17)
        print("==================================================MOVEMENT PROBLEMS==================================================")
        print("Solving problem 25ff71a9")
        mt0, mf0 = run_synthesis("25ff71a9", "nbccg")
        self.assertCountEqual(['moveNode(Dir.DOWN)'], mt0)
        self.assertCountEqual(['FilterByColor(FColor.least)'], mf0)

        print("Solving problem 1e0a9b12")
        #mt0, mf0 = run_synthesis("1e0a9b12", "nbccg")
        #self.assertCountEqual(['[moveNodeMax(Dir.DOWN), moveNodeMax(Dir.DOWN)]'], mt1)
        #self.assertCountEqual(['FilterByColor(FColor.least)'], mf1)

        print("Solving problem 3c9b0459")
        mt2, mf2 = run_synthesis("3c9b0459", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt2)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf2)

        print("Solving problem 6150a2bd")
        mt3, mf3 = run_synthesis("6150a2bd", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt3)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf3)

        print("Solving problem 9dfd6313")
        mt4, mf4 = run_synthesis("9dfd6313", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt4)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf4)

        print("Solving problem 67a3c6ac")
        mt5, mf5 = run_synthesis("67a3c6ac", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.HORIZONTAL)'], mt5)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf5)

        print("Solving problem 74dd1130")
        mt6, mf6 = run_synthesis("74dd1130", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt6)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf6)

        print("Solving problem ed36ccf7")
        mt7, mf7 = run_synthesis("ed36ccf7", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CCW)'], mt7)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf7)

        print("Solving problem 68b16354")
        t8, f8 = run_synthesis("68b16354", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.VERTICAL)'], t8)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f8)

        print("Solving problem a79310a0")
        mt9, mf9 = run_synthesis("a79310a0", "nbccg")
        self.assertCountEqual(
        ['[updateColor(Color.red), moveNode(Dir.DOWN)]'], mt9) # todo-eusolver
        self.assertCountEqual(['FilterByColor(FColor.cyan)'], mf9)

        print("Solving problem 3906de3d")
        mt10, mf10 = run_synthesis("3906de3d", "nbvcg")
        self.assertCountEqual(['moveNodeMax(Dir.UP)'], mt10)
        self.assertCountEqual(['Not(FilterByColor(FColor.black))'], mf10)

        print("Solving problem ce22a75a")
        mt11, mf11 = run_synthesis("ce22a75a", "nbccg")
        self.assertCountEqual(
            ['addBorder(Color.blue)', 'updateColor(Color.blue)'], mt11)
        print("==================================================AUGMENTATION PROBLEMS==================================================")

        print("Solving problem bb43febb")
        at0, af0 = run_synthesis("bb43febb", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.red)'], at0)
        self.assertCountEqual(['FilterByColor(FColor.grey)'], af0)

        print("Solving problem 4258a5f9")
        at1, af1 = run_synthesis("4258a5f9", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)'], at1)
        self.assertCountEqual(['FilterByColor(FColor.grey)'], af1)

        print("Solving problem b27ca6d3")
        at2, af2 = run_synthesis("b27ca6d3", "nbccg")
        self.assertCountEqual(['addBorder(Color.green)', 'NoOp'], at2)
        self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterByColor(FColor.red)'], af2)

        print("Solving problem d037b0a7")
        at3, af3 = run_synthesis("d037b0a7", "nbccg")
        self.assertCountEqual(['extendNode(Dir.DOWN, Overlap.TRUE)'], at3)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], af3)

        print("Solving problem dc1df850")
        at4, af4 = run_synthesis("dc1df850", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)', 'NoOp'], at4)
        self.assertCountEqual(['FilterByColor(FColor.red)', 'FilterBySize(SIZE.MIN)'], af4)

        print("Solving problem 4347f46a")
        at5, af5 = run_synthesis("4347f46a", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.black)'], at5)
        self.assertCountEqual(['Not(FilterByColor(FColor.black))'], af5)

        print("Solving problem 3aa6fb7a")
        at6, af6 = run_synthesis("3aa6fb7a", "nbccg")
        self.assertCountEqual(
            ['fillRectangle(Color.blue, Overlap.TRUE)'], at6)
        self.assertCountEqual(['FilterByColor(FColor.cyan)'], af6)

        print("Solving problem 6d75e8bb")
        at7, af7 = run_synthesis("6d75e8bb", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.red, Overlap.TRUE)'], at7)
        self.assertCountEqual(['FilterByColor(FColor.cyan)'], af7)

        print("Solving problem 913fb3ed")
        at8, af8 = run_synthesis("913fb3ed", "nbccg")
        self.assertCountEqual(
            ['addBorder(Color.blue)', 'addBorder(Color.yellow)', 'addBorder(Color.fuchsia)'], at8)
        self.assertCountEqual(['FilterByColor(FColor.red)', 'FilterByColor(FColor.cyan)', 'FilterByColor(FColor.green)'], af8)

        print("Solving problem e8593010")
        at9, af9 = run_synthesis("e8593010", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.green)', 'updateColor(Color.blue)', 'updateColor(Color.red)', 'NoOp'], at9)
        self.assertCountEqual(['And(FilterByColor(FColor.black), FilterBySize(SIZE.MIN))', 'FilterBySize(SIZE.3)', 'And(FilterByColor(FColor.black), FilterBySize(SIZE.2))', 'FilterByColor(FColor.grey)'], af9)

        print("Solving problem 50cb2852")
        at10, af10 = run_synthesis("50cb2852", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.cyan)'], at10)
        self.assertCountEqual(['Not(FilterByColor(FColor.black))'], af10)

        print("Solving problem 44d8ac46")
        at10, af10 = run_synthesis("44d8ac46", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.red, Overlap.TRUE)', 'NoOp'], at10)
        self.assertCountEqual(['FilterByShape(Shape.square)', 'FilterByColor(FColor.grey)'], af10)

        print("Solving problem 694f12f3")
        at11, af11 = run_synthesis("694f12f3", "nbccg")
        self.assertCountEqual(
            ['hollowRectangle(Color.red)', 'hollowRectangle(Color.blue)', 'hollowRectangle(Color.black)'], at11)  # todo: post-process
        self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)', 'FilterByColor(FColor.yellow)'], af11)

        print("Solving problem 868de0fa")
        st2, sf2 = run_synthesis("868de0fa", "nbccg")
        self.assertCountEqual(
            ['NoOp', 'fillRectangle(Color.red, Overlap.TRUE)',
            'fillRectangle(Color.orange, Overlap.TRUE)'], st2)  # todo: post-process
        self.assertCountEqual(['FilterByColor(FColor.blue)', 'Not(FilterByHeight(HEIGHT.ODD))', 'FilterByHeight(HEIGHT.ODD)'], sf2)

        print("Solving problem 7f4411dc")
        t24, f24 = run_synthesis("7f4411dc", "lrg")
        self.assertCountEqual(['updateColor(Color.black)', 'NoOp'], t24)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)', 'Not(FilterBySize(SIZE.MIN))'], f24)

if __name__ == "__main__":
    unittest.main()
