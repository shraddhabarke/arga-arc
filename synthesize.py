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
                Symmetry_Axis, ObjectId, UpdateColor, MoveNode, MoveNodeMax, AddBorder, ExtendNode, 
                HollowRectangle, Flip, RotateNode, FillRectangle, Insert, Transforms]
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
                if all(isinstance(elem, list) for elem in results):
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
                #print("are-equal:", program.code, are_equal)
                if are_equal:
                    return program, i

    transform_vocab = VocabFactory.create(tleaf_makers)
    input_graphs = [input_graph for input_graph in task.input_abstracted_graphs_original[task.abstraction]]
    enumerator = TSizeEnumerator(task, transform_vocab, ValuesManager())
    expected_graphs = [output.graph.nodes(data=True) for output in task.train_output]
    output_graphs = [output_graph.graph.nodes(data=True) for output_graph in task.output_abstracted_graphs_original[task.abstraction]]
    print("output-graphs:", output_graphs)
    grid_sizes = [inp_graph.grid_size for inp_graph in input_graphs]

    counter, pixel_counts, object_counts = 0, {}, {}
    correct_counts = [set() for _ in range(len(expected_graphs))]
    correct_table = [{node: None for node in input_graph.graph.nodes()} for input_graph in 
        task.input_abstracted_graphs_original[task.abstraction]] # nodes in the input graphs

    def check_termination(program, expected_graphs): # updates the correct_counts for the current filled table
        count = 0
        reconstructed = [task.train_input[iter].undo_abstraction(store[program][iter]) 
                        for iter in range(len(store[program]))]
        for expected_graph, recons in zip(expected_graphs, reconstructed):
            for key in dict(recons.graph.nodes(data=True)):
                if dict(recons.graph.nodes(data=True))[key]['color'] == dict(expected_graph)[key]['color']:
                    correct_counts[count].add(key) 
            count += 1

    def get_correct_pixels(program, output_graphs):
        count = 0
        correct_pixels = [set() for i in range(len(output_graphs))] 
        reconstructed = [task.train_input[iter].undo_abstraction(program[iter]) 
                        for iter in range(len(program))]
        for output_graph, recons in zip(output_graphs, reconstructed):
            for node_id, data in output_graph:   
                if isinstance(output_graph[node_id]['color'], list):
                    colors = dict(zip(data['nodes'], output_graph[node_id]['color']))             
                    for node in data['nodes']:
                        if dict(recons.graph.nodes(data=True))[node]['color'] == colors[node]:
                            correct_pixels[count].add(node)    
                else:
                    for node in data['nodes']:
                        if dict(recons.graph.nodes(data=True))[node]['color'] == output_graph[node_id]['color']:
                            correct_pixels[count].add(node)
            count += 1
        total_pixels = sum([len(correct_count) for correct_count in correct_pixels])
        return (correct_pixels, total_pixels)
    
    def get_correct_objects(actual_values):
        # Compares the abstracted actual ARCGraph with the unabstracted expected ARCGraph!
        correct_objects = []
        input_graphs = [input_graph for input_graph in [getattr(input, Image.abstraction_ops[task.abstraction])() for input in task.train_input]]
        
        for actual_graph, expected_graph, input_graph in zip(actual_values, expected_graphs, input_graphs):
            cobjs_local = set()
            if len(actual_graph.graph.nodes) == len(input_graph.graph.nodes):  # no new objects added
                for node, data in actual_graph.graph.nodes(data=True):
                    if isinstance(actual_graph.graph.nodes(data=True)[node]['color'], list):
                        colors = dict(zip(data['nodes'], actual_graph.graph.nodes(data=True)[node]['color']))
                        if data['nodes'] != [] and \
                            all([expected_graph[pixel]['color'] == colors[pixel] for pixel in data['nodes']]):
                            cobjs_local.add(node)
                    else:
                        if data['nodes'] != [] and \
                        all([expected_graph[pixel]['color'] == \
                            actual_graph.graph.nodes(data=True)[node]['color'] for pixel in data['nodes']]):
                            cobjs_local.add((node[0], node[1]))
            
            elif len(actual_graph.graph.nodes) > len(input_graph.graph.nodes): # check the insertions
                for node, data in actual_graph.graph.nodes(data=True):
                    if node not in input_graph.graph.nodes() and data['nodes'] != [] and \
                    all([expected_graph[pixel]['color'] == \
                        actual_graph.graph.nodes(data=True)[node]['color'] for pixel in data['nodes']]):            
                        cobjs_local.add((node[0], node[1]))
            correct_objects.append(cobjs_local)
        total_objects = sum([len(correct_count) for correct_count in correct_objects])
        return (correct_objects, total_objects)

    while enumerator.hasNext():
        correct_table_updated = False
        correct_counts, unique_values = [set() for _ in range(len(expected_graphs))], set()
        program = enumerator.next()
        counter += 1
        if not program.values:
            continue
        store[program.code] = program.values
        print("enumerator:", program.code)
        print("enumerator-vals:", [val.graph.nodes(data=True) for val in program.values])
        
        pixel_counts[program.code] = get_correct_pixels(program.values, output_graphs)
        object_counts[program.code] = get_correct_objects(program.values) # correct-objects!
        print("program--counts:", pixel_counts[program.code])
        print("program--sum:", pixel_counts[program.code][1])
        if all(not correct_set for correct_set in object_counts[program.code][0]):
            continue

        for dict_, correct_set in zip(correct_table, object_counts[program.code][0]):  # per-task
            for key in correct_set:
                if key in dict_.keys() and (dict_.get(key) is None or \
                pixel_counts[program.code][1] > pixel_counts.get(dict_.get(key), 0)[1] or \
                object_counts[program.code][1] > object_counts.get(dict_.get(key), 0)[1]):
                    dict_[key] = program.code # todo: when values are equal
                    correct_table_updated = True # set true if changes are made

        print("correct-table:", correct_table)
        if correct_table_updated and all(value for dict_ in correct_table for value in dict_.values()):
            correct_counts = [set() for i in range(len(expected_graphs))]
            for dict_ in correct_table:
                unique_values.update(dict_.values())
            for program in list(unique_values):
                check_termination(program, expected_graphs)
        
        # Initiating filter synthesis!----------------------------------------------------------------------
        transforms_data = {program: [] for program in set().union(*[d.values() for d in correct_table])}
        if grid_sizes == [len(correct_count) for correct_count in correct_counts]:
            print("correct-table", correct_table)
            for program in transforms_data:
                for _, dict_ in enumerate(correct_table):
                    transforms_data[program].append(
                    [key for key, prog in dict_.items() if prog == program])
            
        # all objects are covered, synthesizing filters...
            all_filters_found, filters_sol = True, []   
            for program, objects in transforms_data.items():
                print("Enumerating filters:", program, objects)
                task.current_spec = []
                filters = synthesize_filter(objects)
                if filters:
                    filters_sol.append(filters[0])
                    if program not in task.spec.keys():
                        continue
                # it is a variable program, synthesize filters for that!
                    if task.spec[program]:
                        print("task-spec:", task.spec[program])
                        task.current_spec = task.spec[program]
                        variable_filters = synthesize_filter(task.current_spec)
                        if not variable_filters:
                            all_filters_found = False
                            break
                        else:
                            filters_sol.append(variable_filters[0])
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

evals = {"1e0a9b12": "nbccg"} #1e0a9b12 3618c87e, 868de0fa

for task, abstraction in evals.items():
    start_time = time.time()
    print("taskNumber:", task)
    transformations, filters = run_synthesis(task, abstraction)
    print("transformations:", transformations)
    print("filters:", filters)
    print(f"Problem {task}: --- {(time.time() - start_time)} seconds ---")

class TestEvaluation(unittest.TestCase):
    def test_all_problems(self):
        print("==================================================COLORING PROBLEMS==================================================")
        #"""
        print("Solving problem 00d62c1b")
        t1, f1 = run_synthesis("00d62c1b", "ccgbr")
        self.assertCountEqual(['NoOp', 'updateColor(Color.yellow)'], t1)
        self.assertCountEqual(
            ['FilterByColor(FColor.green)', 'FilterByColor(FColor.black)'], f1)

        t2, f2 = run_synthesis("9565186b", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t2)
        self.assertCountEqual(
            ['FilterByColor(FColor.most)', 'FilterByNeighborSize(SIZE.MAX)'], f2)
        
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
        self.assertCountEqual(['FilterBySize(SIZE.MIN)', 'And(FilterByDegree(DEGREE.2), FilterByNeighborSize(SIZE.8))',
                            'And(FilterByNeighborSize(SIZE.MAX), Or(FilterBySize(SIZE.5), FilterBySize(SIZE.8)))', 'FilterBySize(SIZE.MAX)'], f4)
        
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
            ['FilterByNeighborDegree(DEGREE.1)', 'Not(FilterByNeighborDegree(DEGREE.1))'], f8)
        
        print("Solving problem a5313dff")
        t9, f9 = run_synthesis("a5313dff", "ccgbr")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t9)
        self.assertCountEqual(['Or(FilterBySize(SIZE.6), FilterBySize(SIZE.8))',
                            'Or(FilterByColor(FColor.red), FilterBySize(SIZE.ODD))'], f9)
        
        print("Solving problem d2abd087")
        t10, f10 = run_synthesis("d2abd087", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t10)
        self.assertCountEqual(
            ['FilterBySize(SIZE.6)', 'Not(FilterBySize(SIZE.6))'], f10)
        
        print("Solving problem 6e82a1ae")
        t11, f11 = run_synthesis("6e82a1ae", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)', 'updateColor(Color.green)'], t11)
        self.assertCountEqual(
            ['FilterBySize(SIZE.ODD)', 'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f11)

        t12, f12 = run_synthesis("ea32f347", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.yellow)', 'updateColor(Color.blue)', 'updateColor(Color.red)'], t12)
        self.assertCountEqual(['Not(Or(FilterBySize(SIZE.MIN), FilterBySize(SIZE.MAX)))',
                            'FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], f12)

        print("Solving problem aabf363d")
        t14, f14 = run_synthesis("aabf363d", "ccg")
        self.assertCountEqual(['updateColor(Color.most)', 'updateColor(Color.least)'], t14)
        self.assertCountEqual(['Not(FilterBySize(SIZE.12))', 'FilterBySize(SIZE.12)'], f14)

        print("Solving problem c0f76784")
        t15, f15 = run_synthesis("c0f76784", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.orange, Overlap.TRUE)', 'fillRectangle(Color.fuchsia, Overlap.TRUE)', 'fillRectangle(Color.cyan, Overlap.TRUE)'], t15)
        self.assertCountEqual(['FilterBySize(SIZE.12)', 'FilterBySize(SIZE.8)', 'FilterBySize(SIZE.MAX)'], f15)

        print("Solving problem d5d6de2d")
        t16, f16 = run_synthesis("d5d6de2d", "ccg")
        self.assertCountEqual(['updateColor(Color.most)', 'updateColor(Color.green)'], t16)
        self.assertCountEqual(['Or(FilterByColor(FColor.red), FilterBySize(SIZE.MAX))', 'Not(Or(FilterByColor(FColor.red), FilterBySize(SIZE.MAX)))'], f16)

        print("Solving problem 25d8a9c8")
        t17, f17 = run_synthesis("25d8a9c8", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.grey)', 'updateColor(Color.black)'], t17)
        self.assertCountEqual(['And(FilterBySize(SIZE.3), Not(FilterByColor(FColor.green)))',
                            'Or(FilterByColor(FColor.green), Not(FilterBySize(SIZE.3)))'], f17)

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

        print("Solving problem 5582e5ca")
        t21, f21 = run_synthesis("5582e5ca", "ccg")
        self.assertCountEqual(['updateColor(Color.most)'], t21)
        self.assertCountEqual(['FilterByNeighborSize(SIZE.MIN)'], f21)

        print("Solving problem b1948b0a")
        t22, f22 = run_synthesis("b1948b0a", "nbccg")
        self.assertCountEqual(['updateColor(Color.red)', 'NoOp'], t22)
        self.assertCountEqual(
            ['FilterByColor(FColor.fuchsia)', 'FilterByColor(FColor.orange)'], f22)
        
        print("==================================================MOVEMENT PROBLEMS==================================================")
        print("Solving problem 3c9b0459")
        mt1, mf1 = run_synthesis("3c9b0459", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt1)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf1)

        print("Solving problem 6150a2bd")
        mt2, mf2 = run_synthesis("6150a2bd", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt2)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf2)

        print("Solving problem 9dfd6313")
        mt3, mf3 = run_synthesis("9dfd6313", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt3)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf3)

        print("Solving problem 25ff71a9")
        mt7, mf7 = run_synthesis("25ff71a9", "nbccg")
        self.assertCountEqual(['moveNode(Dir.DOWN)'], mt7)
        self.assertCountEqual(['FilterByColor(FColor.least)'], mf7)

        print("Solving problem 67a3c6ac")
        mt8, mf8 = run_synthesis("67a3c6ac", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.HORIZONTAL)'], mt8)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf8)
        
        print("Solving problem 74dd1130")
        mt9, mf9 = run_synthesis("74dd1130", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt9)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf9)

        print("Solving problem ed36ccf7")
        mt10, mf10 = run_synthesis("ed36ccf7", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CCW)'], mt10)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], mf10)

        print("Solving problem 3906de3d")
        t24, f24 = run_synthesis("3906de3d", "nbvcg")
        self.assertCountEqual(['moveNodeMax(Dir.UP)'], t24)
        self.assertCountEqual(['Not(FilterByColor(FColor.black))'], f24)

        print("Solving problem 68b16354")
        t23, f23 = run_synthesis("68b16354", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.VERTICAL)'], t23)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], f23)

        print("Solving problem a79310a0")
        mt3, mf3 = run_synthesis("a79310a0", "nbccg")
        self.assertCountEqual(['[updateColor(Color.red), moveNode(Dir.DOWN)]'], mt3)
        self.assertCountEqual(['FilterByColor(FColor.cyan)'], mf3)
        print("==================================================AUGMENTATION PROBLEMS==================================================")
        print("Solving problem bb43febb")
        at0, af0 = run_synthesis("bb43febb", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.red)'], at0)
        self.assertCountEqual(['FilterByColor(FColor.grey)'], af0)

        print("Solving problem 4258a5f9")
        at1, af1 = run_synthesis("4258a5f9", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)'], at1)
        self.assertCountEqual(['FilterByColor(FColor.grey)'], af1)

        print("Solving problem ce22a75a")
        at3, af3 = run_synthesis("ce22a75a", "nbccg")
        self.assertCountEqual(
            ['[updateColor(Color.blue), addBorder(Color.blue)]'], at3) # todo
        self.assertCountEqual(['FilterByColor(FColor.grey)'], af3)

        print("Solving problem 4347f46a")
        at4, af4 = run_synthesis("4347f46a", "ccg")
        self.assertCountEqual(['hollowRectangle(Color.most)'], at4)
        self.assertCountEqual(['Not(FilterByColor(FColor.blue))'], af4)

        print("Solving problem b27ca6d3")
        at5, af5 = run_synthesis("b27ca6d3", "nbccg")
        self.assertCountEqual(['addBorder(Color.green)', 'NoOp'], at5)
        self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], af5)

        print("Solving problem d037b0a7")
        at6, af6 = run_synthesis("d037b0a7", "nbccg")
        self.assertCountEqual(['extendNode(Dir.DOWN, Overlap.TRUE)'], at6)
        self.assertCountEqual(['FilterBySize(SIZE.MIN)'], af6)

        print("Solving problem 694f12f3")
        at8, af8 = run_synthesis("694f12f3", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.red)', 'hollowRectangle(Color.blue)'], at8)
        self.assertCountEqual(['FilterBySize(SIZE.MAX)', 'FilterBySize(SIZE.MIN)'], af8)

        print("Solving problem 3aa6fb7a")
        at10, af10 = run_synthesis("3aa6fb7a", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.blue, Overlap.TRUE)'], at10)
        self.assertCountEqual(['FilterByColor(FColor.cyan)'], af10)

        print("Solving problem 50cb2852")
        at11, af11 = run_synthesis("50cb2852", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.cyan)'], at11)
        self.assertCountEqual(['Not(FilterByColor(FColor.black))'], af11)

        print("Solving problem dc1df850")
        at12, af12 = run_synthesis("dc1df850", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)', 'NoOp'], at12)
        self.assertCountEqual(['FilterByColor(FColor.red)', 'Not(FilterByColor(FColor.red))'], af12)

        print("Solving problem 6d75e8bb")
        at13, af13 = run_synthesis("6d75e8bb", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.red, Overlap.TRUE)'], at13)
        self.assertCountEqual(['FilterByColor(FColor.cyan)'], af13)

        print("Solving problem 913fb3ed")
        at14, af14 = run_synthesis("913fb3ed", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)', 'addBorder(Color.yellow)', 'addBorder(Color.fuchsia)'], at14)
        self.assertCountEqual(['FilterByColor(FColor.red)', 'FilterByColor(FColor.cyan)', 'FilterByColor(FColor.green)'], af14)

if __name__ == "__main__":
    unittest.main()