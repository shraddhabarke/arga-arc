import time
from itertools import combinations, permutations
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
from compute_pcfg import *
import math, json, csv

# Transform Vocab

tleaf_makers = [NoOp, Color, Dir, Overlap, Rotation_Angle, RelativePosition, ImagePoints,
                Symmetry_Axis, Mirror_Axis, ObjectId, MoveNode, MoveNodeMax, AddBorder, ExtendNode,
                HollowRectangle, RotateNode, Flip, FillRectangle, Insert, UpdateColor, Transforms]

fleaf_makers = [Object, FColor, Size, Degree, Height, Width, Shape, Row, Column, Neighbor_Of, Direct_Neighbor_Of,
                Color_Equals, Size_Equals, Degree_Equals, Shape_Equals, Height_Equals, Row_Equals, Not, Column_Equals, And, Or]

# Initializing terminal sizes!
Color._sizes = {color.name: 1 for color in Color}
Dir._sizes = {dir.name: 1 for dir in Dir}
Overlap._sizes = {overlap.name: 1 for overlap in Overlap}
Rotation_Angle._sizes = {overlap.name: 1 for overlap in Rotation_Angle}
Symmetry_Axis._sizes = {overlap.name: 1 for overlap in Symmetry_Axis}
RelativePosition._sizes = {relativepos.name: 1 for relativepos in RelativePosition}
ImagePoints._sizes = {imagepts.name: 1 for imagepts in ImagePoints}
ObjectId._sizes = {objid.value: 1 for objid in ObjectId._all_values}
Mirror_Axis._sizes = {axis.name: 1 for axis in Mirror_Axis}
FColor._sizes = {color.name: 1 for color in FColor}
Object._sizes = {obj.name: 1 for obj in Object}
Shape._sizes = {shape.name: 1 for shape in Shape}
Degree._sizes = {degree.value: 1 for degree in Degree._all_values}
Size._sizes = {s.value: 1 for s in Size._all_values}
Column._sizes = {col.value: 1 for col in Column._all_values}
Row._sizes = {row.value: 1 for row in Row._all_values}
Height._sizes = {height.value: 1 for height in Height._all_values}
Width._sizes = {width.value: 1 for width in Width._all_values}

def get_t_probability(transform_probabilities, item, category=None):
    if category:
        return transform_probabilities[category].get(item, 0)
    else:
        return transform_probabilities['Transform'].get(item, 0)

def get_all_t_probabilities(vocab_makers, transform_probabilities):
    all_probs = {}
    for vocab_maker in vocab_makers:
        class_name = vocab_maker.__name__
        prob = get_t_probability(transform_probabilities, class_name, 'Transform')  # Now using 'Transform' as category
        all_probs[class_name] = prob

    # Handle token probabilities
    for category in ['Color', 'Direction', 'Overlap', 'Rotation_Angle', 'Symmetry_Axis', 'ImagePoints', 'RelativePosition', 'ObjectId']:
        if category in transform_probabilities:
            for token in transform_probabilities[category]:
                prob = get_t_probability(transform_probabilities, token, category)
                all_probs[f"{category}.{token}"] = prob
    return all_probs

def get_f_probability(filter_probabilites, item, category=None):
    if category:
        return filter_probabilites[category].get(item, 0)
    else:
        prob = filter_probabilites.get('Filters', {}).get(item, 0)
        return prob if prob > 0 else filter_probabilites.get('Filter', {}).get(item, 0)

def get_all_f_probabilities(vocab_makers, filter_probabilites):
    all_probs = []
    for vocab_maker in vocab_makers:
        class_name = vocab_maker.__name__
        prob = get_f_probability(filter_probabilites, class_name, None)
        all_probs.append((class_name, prob))

    # Handle token probabilities
    for category in ['FColor', 'Shape', 'Degree', 'Size', 'Column', 'Row', 'Width', 'Height']:
        if category in filter_probabilites:
            for token in filter_probabilites[category]:
                prob = get_f_probability(filter_probabilites, token, category)
                all_probs.append((f"{category}.{token}", prob))
    return all_probs

def compute_costs(taskNumber):
    from compute_pcfg import compute_transform_costs, compute_filter_costs
    transform_probabilities = compute_transform_costs(taskNumber)
    filter_probabilites = compute_filter_costs(taskNumber)
    #print("transform_probabilities:", transform_probabilities)
    #print("filter_probabilites:", filter_probabilites)

    t_vocabMakers = [NoOp, MoveNode, MoveNodeMax, AddBorder, ExtendNode, Mirror,
                HollowRectangle, RotateNode, Flip, FillRectangle, Insert, UpdateColor, Transforms]
    f_vocabMakers = [Object, FColor, Size, Degree, Height, Width, Shape, Row, Column, Neighbor_Of, Direct_Neighbor_Of,
                Color_Equals, Size_Equals, Degree_Equals, Shape_Equals, Height_Equals, Row_Equals, Not, Column_Equals, And, Or]
    transform_values = get_all_t_probabilities(t_vocabMakers, transform_probabilities)
    filter_values = get_all_f_probabilities(f_vocabMakers, filter_probabilites)
    print("transform_values", transform_values)
    print("filter_values:", filter_values)
    # Computing Real Costs
    f_real_costs, t_real_costs = {}, {}
    for trans, probability in transform_values.items():
        num = -math.log(probability)
        t_real_costs[trans] = int(math.ceil(num)) if (num - int(num) > 0.5) else int(math.floor(num))
    for trans, probability in filter_values:
        num = -math.log(probability) if probability > 0 else 1
        f_real_costs[trans] = int(math.ceil(num)) if (num - int(num) > 0.5) else int(math.floor(num))
    print("t_real_costs", t_real_costs)
    print("f_real_costs:", f_real_costs)
    Color.set_sizes(t_real_costs)
    Dir.set_sizes(t_real_costs)
    Overlap.set_sizes(t_real_costs)
    Rotation_Angle.set_sizes(t_real_costs)
    Symmetry_Axis.set_sizes(t_real_costs)
    RelativePosition.set_sizes(t_real_costs)
    ImagePoints.set_sizes(t_real_costs)

    FColor.set_sizes(f_real_costs)
    Shape.set_sizes(f_real_costs)
    DegreeValue.set_sizes(f_real_costs)
    ColumnValue.set_sizes(f_real_costs)
    RowValue.set_sizes(f_real_costs)
    SizeValue.set_sizes(f_real_costs)
    HeightValue.set_sizes(f_real_costs)
    SizeValue.set_sizes(f_real_costs)
    WidthValue.set_sizes(f_real_costs)
    for vocab_maker in t_vocabMakers:
        if vocab_maker.__name__ in t_real_costs:
            vocab_maker.default_size = t_real_costs[vocab_maker.__name__]
    f_real_costs['Direct_Neighbor_Of'] = f_real_costs['Neighbor_Of']

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
    #compute_costs(taskNumber)
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
    print("input_graphs:", input_graphs)
    print("input-graphs:", [input_graph.graph.nodes() for input_graph in task.input_abstracted_graphs_original[task.abstraction]])
    enumerator = TSizeEnumerator(task, transform_vocab, ValuesManager())
    # has the entire unabstracted output graphs
    expected_graphs = [output.graph.nodes(
        data=True) for output in task.train_output]
    output_graphs = [output_graph.graph.nodes(
        data=True) for output_graph in task.output_abstracted_graphs_original[task.abstraction]]
    input_graph_dicts, output_graphs_dicts = [], []
    filter_cache = {}
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
        filter_vocab = VocabFactory.create(fleaf_makers)
        enumerator = FSizeEnumerator(task, filter_vocab, ValuesManager())
        i = 0
        while enumerator.hasNext():
            program = enumerator.next()
            i += 1
            results = program.values
            #print(f"Filter-Program: {program.code}: {program.size, results}")
            filter_compare(results, subset)
            if filter_compare(results, subset):
                return program.code
            
    correct_transforms = set()
    while enumerator.hasNext():
        program = enumerator.next()
        print("enumerator:", program.code, program.size)
        if program.values:  # Check if program.values is not empty
            if isinstance(program.values[0], list):
                progvalues = program.values[0]
            else:
                progvalues = program.values
        else:
            progvalues = None

        if progvalues is None:
            continue
        # transformed graph values of the current program enumerated
        blue_prints = [val.graph.nodes(data=True) for val in progvalues]
        actual_values = progvalues
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
            actual_value = train_in.undo_abstraction(
                actual_value).graph.nodes(data=True)

            correct_nodes = [
                node for node, node_info in actual_value if (node in expected_graph and expected_graph[node]['color'] == node_info['color']) and
                (not node in node_correctness_map or node_correctness_map[node])]
            if correct_nodes:
                correct_nodes_for_this_transform[task_idx] = correct_nodes
                aggregated_correct_nodes_per_task[task_idx].update(correct_nodes)

        if any(val for val in correct_nodes_for_this_transform):
            correct_nodes_per_transform[program.code] = correct_nodes_for_this_transform
        full_coverage_per_task = [set(aggregated_correct_nodes) == set(dict(expected_graphs[task_idx]).keys())
                                for task_idx, aggregated_correct_nodes in enumerate(aggregated_correct_nodes_per_task)]
        if all(full_coverage_per_task):
            #print("New Coverage being Considered")
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

            minimal_transforms = list(minimal_transforms)
            if minimal_transforms != correct_transforms:
                correct_transforms = minimal_transforms
                print("Minimal-transforms:", correct_transforms)
                # Candidate set of transforms found: initiating filter synthesis...
                all_filters_found, filters_sol = True, []
                for program in minimal_transforms:
                    filters = None
                    print("Enumerating filters for:", program)
                    if program in filter_cache:
                        if filter_cache[program] is None:
                            all_filters_found = False
                            break
                        filters = filter_cache[program]
                    else:
                        if "Var" not in program:
                            input_nodes = find_input_nodes(input_graphs, correct_nodes_per_transform[program])
                            #print("program:", program, input_nodes)
                            subset = [{tup: [] for tup in subset} for subset in input_nodes]
                        else:
                            if len(task.all_specs) == 1:
                                subset = task.all_specs[0]
                            else:
                                subset = task.all_specs[int(program.split("_")[-1])]
                            task.current_spec = subset
                        filters = synthesize_filter(subset)
                        filter_cache[program] = filters
                    if filters:
                        filters_sol.append(filters)
                    else:
                        all_filters_found = False
                        break
                if all_filters_found:
                    print("Transformation Solution:", [program for program in minimal_transforms])
                    print("Filter Solution", [
                        program for program in filters_sol])
                    return [program for program in minimal_transforms], [
                    program for program in filters_sol]

# todo: add insert 3618c87e, 6c434453, 88a10436, 67a423a3
evals = {"e73095fd": "ccgbr2"}
evals = {"25d487eb": "ccgbr"}
evals = {"29c11459": "nbccg"}
# ARGA Problems --
# Augmentation: 29c11459, 67a423a3, 88a10436, 22168020
# 4093f84a -- [filterbySize(Size.1) -> updateColor(gray), moveNodeMax(Variable)]
# ExtendNode -->  dbc1a6ce, 7ddcd7ec, 25d487eb
# moveNode by height --> 5521c0d9
evals = {"dbc1a6ce": "nbccg"}
evals = {"7ddcd7ec": "nbccg"} # todo -- blue print for extendNode
evals = {"88a10436": "mcccg"}
evals = {"67a423a3": "nbccg"}
evals = {"ded97339": "nbccg"}
evals = {"63613498": "nbccg"}
evals = {"3618c87e": "nbccg"}
evals = {"4258a5f9": "nbccg"}
evals = {"3906de3d": "nbvcg"}
evals = {"f8a8fe49": "nbccg"}
evals = {"2c608aff": "ccgbr"}
evals = {"91714a58": "lrg"}
evals = {"63613498": "nbccg"}
evals = {"f8a8fe49": "nbccg"}
evals = {"25d487eb": "lrg"}
evals = {"6c434453": "nbccg"}
evals = {"ddf7fa4f": "nbccg"}

allevals = {"bb43febb": "nbccg", "4258a5f9": "nbccg", "b27ca6d3": "nbccg", "d037b0a7": "nbccg", "dc1df850": "nbccg", "4347f46a": "nbccg",
            "3aa6fb7a": "nbccg", "6d75e8bb": "nbccg", "913fb3ed": "nbccg", "50cb2852": "nbccg", "44d8ac46": "nbccg", "694f12f3": "nbccg",
            "868de0fa": "nbccg", "60b61512": "nbccg", "e8593010": "ccg", "7f4411dc": "lrg"
            }
results = {}
start_time = time.time()
print(run_synthesis("ddf7fa4f", "nbccg"))
execution = (time.time() - start_time)
print(execution)
"""
with open("unguided_arc.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Task Number', 'Abstraction', 'Transformations', 'Filters', 'Time'])
    for task, abstraction in allevals.items():
        start_time = time.time()
        print("taskNumber:", task)
        transformations, filters = run_synthesis(task, abstraction)
        print("transformations:", transformations)
        print("filters:", filters)
        execution = (time.time() - start_time)
        print(f"Problem {task}: --- {execution} seconds ---")
        writer.writerow([task, abstraction, transformations, filters, execution])   
"""
class TestEvaluation(unittest.TestCase):
    def all_problems(self):
        print("==================================================AUGMENTATION PROBLEMS==================================================")
        print("Solving problem bb43febb")
        at0, af0 = run_synthesis("bb43febb", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.red)'], at0)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], af0)

        print("Solving problem 4258a5f9")
        at1, af1 = run_synthesis("4258a5f9", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)'], at1)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], af1)

        print("Solving problem b27ca6d3")
        at2, af2 = run_synthesis("b27ca6d3", "nbccg")
        self.assertCountEqual(['addBorder(Color.green)', 'NoOp'], at2)
        self.assertCountEqual(['Size_Of(Object.this) == SIZE.MAX', 'Equals(FColor.black, FColor.black)'], af2)

        print("Solving problem d037b0a7")
        at3, af3 = run_synthesis("d037b0a7", "nbccg")
        self.assertCountEqual(['extendNode(Dir.DOWN, Overlap.TRUE)'], at3)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], af3)

        print("Solving problem dc1df850")
        at4, af4 = run_synthesis("dc1df850", "nbccg")
        self.assertCountEqual(['addBorder(Color.blue)', 'NoOp'], at4)
        self.assertCountEqual(['Color_Of(Object.this) == FColor.red', 'Equals(FColor.black, FColor.black)'], af4)

        print("Solving problem 4347f46a")
        at5, af5 = run_synthesis("4347f46a", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.black)'], at5)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], af5)

        print("Solving problem 3aa6fb7a")
        at6, af6 = run_synthesis("3aa6fb7a", "nbccg")
        self.assertCountEqual(
            ['fillRectangle(Color.blue, Overlap.TRUE)'], at6)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], af6)

        print("Solving problem 6d75e8bb")
        at7, af7 = run_synthesis("6d75e8bb", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.red, Overlap.TRUE)'], at7)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], af7)

        print("Solving problem 913fb3ed")
        at8, af8 = run_synthesis("913fb3ed", "nbccg")
        self.assertCountEqual(
            ['addBorder(Color.blue)', 'addBorder(Color.yellow)', 'addBorder(Color.fuchsia)'], at8)
        self.assertCountEqual(['Color_Of(Object.this) == FColor.red', 'Color_Of(Object.this) == FColor.cyan', 'Color_Of(Object.this) == FColor.green'], af8)

        print("Solving problem 50cb2852")
        at10, af10 = run_synthesis("50cb2852", "nbccg")
        self.assertCountEqual(['hollowRectangle(Color.cyan)'], at10)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], af10)

        print("Solving problem 44d8ac46")
        at10, af10 = run_synthesis("44d8ac46", "nbccg")
        self.assertCountEqual(['fillRectangle(Color.red, Overlap.TRUE)', 'NoOp'], at10)
        self.assertCountEqual(['Shape_Of(Object.this) == Shape.square', 'Equals(FColor.black, FColor.black)'], af10)

        print("Solving problem 694f12f3")
        at11, af11 = run_synthesis("694f12f3", "nbccg")
        self.assertCountEqual(
            ['hollowRectangle(Color.red)', 'hollowRectangle(Color.blue)', 'hollowRectangle(Color.black)'], at11)  # todo: post-process
        self.assertCountEqual(['Size_Of(Object.this) == SIZE.MAX', 'Size_Of(Object.this) == SIZE.MIN', 'Equals(FColor.black, FColor.black)'], af11)

        print("Solving problem 868de0fa")
        st2, sf2 = run_synthesis("868de0fa", "nbccg")
        self.assertCountEqual(
            ['NoOp', 'fillRectangle(Color.red, Overlap.TRUE)',
            'fillRectangle(Color.orange, Overlap.TRUE)'], st2)  # todo: post-process
        self.assertCountEqual(['Equals(FColor.black, FColor.black)', 'Not(Height_Of(Object.this) == HEIGHT.ODD)', 'Height_Of(Object.this) == HEIGHT.ODD'], sf2)

        print("Solving problem 60b61512")
        st3, sf3 = run_synthesis("60b61512", "nbccg")
        self.assertCountEqual(
            ['fillRectangle(Color.orange, Overlap.FALSE)'], st3
        )
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], sf3)

        print("Solving problem e8593010")
        at9, af9 = run_synthesis("e8593010", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.green)', 'updateColor(Color.blue)', 'updateColor(Color.red)', 'NoOp'], at9)
        self.assertCountEqual(['And(Color_Of(Object.this) == FColor.black, Size_Of(Object.this) == SIZE.MIN)', 'Color_Of(Object.this) == FColor.grey', 'Size_Of(Object.this) == SIZE.3', 'And(Color_Of(Object.this) == FColor.black, Size_Of(Object.this) == SIZE.2)'], af9)

        #print("Solving problem 7f4411dc")
        #t24, f24 = run_synthesis("7f4411dc", "lrg")
        #self.assertCountEqual(['updateColor(Color.black)', 'NoOp'], t24)
        #self.assertCountEqual(['Size_Of(Object.this) == SIZE.MIN', 'Not(Size_Of(Object.this) == SIZE.MIN)'], f24)

        print("==================================================COLORING PROBLEMS==================================================")
        print("Solving problem 0d3d703e")
        ct0, cf0 = run_synthesis("0d3d703e", "nbccg")
        self.assertCountEqual(['updateColor(Color.green)', 'updateColor(Color.yellow)', 'updateColor(Color.brown)', 'updateColor(Color.red)', 'updateColor(Color.blue)', 'updateColor(Color.grey)', 'updateColor(Color.cyan)', 'updateColor(Color.fuchsia)'], ct0)
        self.assertCountEqual(['Color_Of(Object.this) == FColor.yellow', 'Color_Of(Object.this) == FColor.green', 'Color_Of(Object.this) == FColor.cyan', 
                            'Color_Of(Object.this) == FColor.fuchsia', 'Color_Of(Object.this) == FColor.grey', 'Color_Of(Object.this) == FColor.blue', 'Color_Of(Object.this) == FColor.brown', 'Color_Of(Object.this) == FColor.red'], cf0)

        print("Solving problem 810b9b61")
        ct2, cf2 = run_synthesis("810b9b61", "ccgbr")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], ct2)
        self.assertCountEqual(['Not(Shape_Of(Object.this) == Shape.enclosed)', 'Shape_Of(Object.this) == Shape.enclosed'], cf2)

        print("Solving problem d23f8c26")
        ct0, cf0 = run_synthesis("d23f8c26", "nbccg")
        self.assertCountEqual(['updateColor(Color.black)', 'NoOp'], ct0)
        self.assertCountEqual(['Column_Of(Object.this) == COLUMN.CENTER', 'Not(Column_Of(Object.this) == COLUMN.CENTER)'], cf0)

        print("Solving problem a5f85a15") # even columns
        ct1, cf1 = run_synthesis("a5f85a15", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.yellow)'], ct1)
        self.assertCountEqual(['Column_Of(Object.this) == COLUMN.EVEN', 'Column_Of(Object.this) == COLUMN.ODD'], cf1)

        print("Solving problem d406998b") # even columns from right
        ct1, cf1 = run_synthesis("d406998b", "nbvcg")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], ct1)
        self.assertCountEqual(['Not(Column_Of(Object.this) == COLUMN.EVEN_FROM_RIGHT)', 'Column_Of(Object.this) == COLUMN.EVEN_FROM_RIGHT'], cf1)

        print("Solving problem b2862040")
        ct2, cf2 = run_synthesis("b2862040", "ccgbr")
        self.assertCountEqual(['updateColor(Color.cyan)', 'NoOp'], ct2)
        self.assertCountEqual(['Shape_Of(Object.this) == Shape.enclosed', 'Not(Shape_Of(Object.this) == Shape.enclosed)'], cf2)

        print("Solving problem ba26e723")
        ct3, cf3 = run_synthesis("ba26e723", "nbvcg")
        self.assertCountEqual(['updateColor(Color.fuchsia)', 'NoOp'], ct3)
        self.assertCountEqual(['Column_Of(Object.this) == COLUMN.MOD3', 'Not(Column_Of(Object.this) == COLUMN.MOD3)'], cf3)

        #print("Solving problem 7b6016b9") # todo - filter size: 12
        #ct3, cf3 = run_synthesis("7b6016b9", "ccg")
        #self.assertCountEqual(['updateColor(Color.red)', 'NoOp', 'updateColor(Color.green)'], ct3)
        #self.assertCountEqual(['Not(Or(Color_Of(Obj) == FColor.least, Or(Size_Of(Obj) == SIZE.MAX, Or(FilterBySize(SIZE.99), FilterBySize(SIZE.71)))))',
                            #'Color_Of(Obj) == FColor.least', 'Or(Size_Of(Obj) == SIZE.MAX, Or(FilterBySize(SIZE.99), FilterBySize(SIZE.71)))'], cf3)

        print("Solving problem f76d97a5")
        t0, f0 = run_synthesis("f76d97a5", "nbccg")
        self.assertCountEqual(['updateColor(Color.yellow)', 'updateColor(Color.black)', 'updateColor(Color.most)'], t0)
        self.assertCountEqual(['Size_Of(Object.this) == SIZE.5', 'Not(Color_Of(Object.this) == FColor.grey)', 'And(Equals(FColor.least, FColor.grey), Color_Of(Object.this) == FColor.grey)'], f0)

        print("Solving problem d511f180")
        t0, f0 = run_synthesis("d511f180", "nbccg")
        self.assertCountEqual(
            ['NoOp', 'updateColor(Color.grey)', 'updateColor(Color.cyan)'], t0)
        self.assertCountEqual(
        ['Not(Or(Color_Of(Object.this) == FColor.grey, Color_Of(Object.this) == FColor.cyan))', 'Color_Of(Object.this) == FColor.cyan', 'Color_Of(Object.this) == FColor.grey'], f0)
        
        print("Solving problem 00d62c1b")
        t1, f1 = run_synthesis("00d62c1b", "ccgbr")
        self.assertCountEqual(['NoOp', 'updateColor(Color.yellow)'], t1)
        self.assertCountEqual(
        ['Color_Of(Object.this) == FColor.black', 'Color_Of(Object.this) == FColor.green'], f1)

        print("Solving problem 9565186b")
        t2, f2 = run_synthesis("9565186b", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t2)
        self.assertCountEqual(['Color_Of(Object.this) == FColor.most', 'Not(Color_Of(Object.this) == FColor.most)'], f2)

        print("Solving problem b230c067")
        t3, f3 = run_synthesis("b230c067", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t3)
        self.assertCountEqual(
        ['Size_Of(Object.this) == SIZE.MAX', 'Size_Of(Object.this) == SIZE.MIN'], f3)

        print("Solving problem 6455b5f5")
        t5, f5 = run_synthesis("6455b5f5", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.blue)', 'updateColor(Color.cyan)', 'NoOp'], t5)
        self.assertCountEqual(['Size_Of(Object.this) == SIZE.MIN', 'Not(Or(Size_Of(Object.this) == SIZE.MIN, Size_Of(Object.this) == SIZE.MAX))', 'Size_Of(Object.this) == SIZE.MAX'], f5)

        print("Solving problem 67385a82")
        t7, f7 = run_synthesis("67385a82", "nbccg")
        self.assertCountEqual(['updateColor(Color.cyan)', 'NoOp'], t7)
        self.assertCountEqual(
        ['Size_Of(Object.this) == SIZE.MIN', 'Not(Size_Of(Object.this) == SIZE.MIN)'], f7)

        print("Solving problem 810b9b61")
        t8, f8 = run_synthesis("810b9b61", "ccgbr")
        self.assertCountEqual(['updateColor(Color.green)', 'NoOp'], t8)
        self.assertCountEqual(['Not(Shape_Of(Object.this) == Shape.enclosed)', 'Shape_Of(Object.this) == Shape.enclosed'], f8)

        print("Solving problem a5313dff")
        t9, f9 = run_synthesis("a5313dff", "ccgbr")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t9)
        self.assertCountEqual(['Height_Of(Object.this) == HEIGHT.3', 'Not(Height_Of(Object.this) == HEIGHT.3)'], f9)

        print("Solving problem d2abd087")
        t10, f10 = run_synthesis("d2abd087", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)'], t10)
        self.assertCountEqual(['Size_Of(Object.this) == SIZE.6', 'Not(Size_Of(Object.this) == SIZE.6)'], f10)

        print("Solving problem 6e82a1ae")
        t11, f11 = run_synthesis("6e82a1ae", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.blue)', 'updateColor(Color.green)'], t11)
        self.assertCountEqual(['Size_Of(Object.this) == SIZE.ODD', 'Size_Of(Object.this) == SIZE.MAX', 'Size_Of(Object.this) == SIZE.MIN'], f11)

        print("Solving problem ea32f347")
        t12, f12 = run_synthesis("ea32f347", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.yellow)', 'updateColor(Color.blue)', 'updateColor(Color.red)'], t12)
        self.assertCountEqual(['Or(Size_Of(Object.this) == SIZE.5, Height_Of(Object.this) == HEIGHT.4)',
        'Size_Of(Object.this) == SIZE.MAX', 'Size_Of(Object.this) == SIZE.MIN'], f12)

        print("Solving problem aabf363d")
        t14, f14 = run_synthesis("aabf363d", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.yellow)', 'updateColor(Color.fuchsia)'], t14)
        self.assertCountEqual(['Not(Size_Of(Object.this) == SIZE.12)', 'Color_Of(Object.this) == FColor.red', 'Color_Of(Object.this) == FColor.green'], f14)

        print("Solving problem c0f76784")
        t15, f15 = run_synthesis("c0f76784", "nbccg")
        self.assertCountEqual(['NoOp', 'fillRectangle(Color.orange, Overlap.TRUE)',  # todo: post-process
                            'fillRectangle(Color.fuchsia, Overlap.TRUE)', 'fillRectangle(Color.cyan, Overlap.TRUE)'], t15)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)', 'Size_Of(Object.this) == SIZE.12', 'Size_Of(Object.this) == SIZE.8', 'Size_Of(Object.this) == SIZE.MAX'], f15)

        print("Solving problem d5d6de2d")
        t16, f16 = run_synthesis("d5d6de2d", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.black)', 'updateColor(Color.green)'], t16)
        self.assertCountEqual(['Or(Color_Of(Object.this) == FColor.red, Size_Of(Object.this) == SIZE.MAX)', 'Not(Or(Color_Of(Object.this) == FColor.red, Size_Of(Object.this) == SIZE.MAX))'], f16)

        print("Solving problem aedd82e4")
        t18, f18 = run_synthesis("aedd82e4", "nbccg")
        self.assertCountEqual(['updateColor(Color.blue)', 'NoOp'], t18)
        self.assertCountEqual(
        ['Size_Of(Object.this) == SIZE.MIN', 'Not(Size_Of(Object.this) == SIZE.MIN)'], f18)

        print("Solving problem c8f0f002")
        t19, f19 = run_synthesis("c8f0f002", "nbccg")
        self.assertCountEqual(['NoOp', 'updateColor(Color.grey)'], t19)
        self.assertCountEqual(
        ['Color_Of(Object.this) == FColor.orange', 'Not(Color_Of(Object.this) == FColor.orange)'], f19)

        print("Solving problem 5582e5ca")
        t21, f21 = run_synthesis("5582e5ca", "ccg")
        self.assertCountEqual(['updateColor(Color.most)'], t21)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], f21)

        print("Solving problem b1948b0a")
        t22, f22 = run_synthesis("b1948b0a", "nbccg")
        self.assertCountEqual(['updateColor(Color.red)', 'NoOp'], t22)
        self.assertCountEqual(['Color_Of(Object.this) == FColor.orange', 'Color_Of(Object.this) == FColor.fuchsia'], f22)

        print("Solving problem a61f2674")
        t23, f23 = run_synthesis("a61f2674", "nbccg")
        self.assertCountEqual(
            ['updateColor(Color.red)', 'updateColor(Color.black)', 'updateColor(Color.blue)'], t23)
        self.assertCountEqual(
        ['Or(Size_Of(Object.this) == SIZE.4, Column_Of(Object.this) == COLUMN.MOD3)', 'Size_Of(Object.this) == SIZE.MAX', 'Size_Of(Object.this) == SIZE.MIN'], f23)

        print("Solving problem 25d8a9c8")
        t17, f17 = run_synthesis("25d8a9c8", "ccg")
        self.assertCountEqual(
            ['updateColor(Color.grey)', 'updateColor(Color.black)'], t17)
        self.assertCountEqual(['And(Size_Of(Object.this) == SIZE.3, Height_Of(Object.this) == HEIGHT.MIN)',
        'Not(And(Size_Of(Object.this) == SIZE.3, Height_Of(Object.this) == HEIGHT.MIN))'], f17)

        print("Solving problem 91714a58")
        ct11, cf11 = run_synthesis("91714a58", "lrg")
        self.assertCountEqual(['updateColor(Color.black)', 'NoOp'], ct11)
        self.assertCountEqual(['Not(Size_Of(Object.this) == SIZE.MAX)', 'Size_Of(Object.this) == SIZE.MAX'], cf11)

        print("Solving problem 08ed6ac7")
        t4, f4 = run_synthesis("08ed6ac7", "nbccg")
        self.assertCountEqual(['updateColor(Color.yellow)', 'updateColor(Color.green)',
                            'updateColor(Color.red)', 'updateColor(Color.blue)'], t4)
        self.assertCountEqual(['And(Not(Size_Of(Object.this) == SIZE.MAX), Or(Size_Of(Object.this) == SIZE.5, Size_Of(Object.this) == SIZE.8))',
        'Size_Of(Object.this) == SIZE.MAX', 'Or(Size_Of(Object.this) == SIZE.4, Size_Of(Object.this) == SIZE.6)', 'Size_Of(Object.this) == SIZE.MIN'], f4)

        # todo
        print("Solving problem 63613498") # doesn't satisfy test likely
        ct0, cf0 = run_synthesis("63613498", "nbccg")
        self.assertCountEqual(['updateColor(Color.grey)', 'NoOp'], ct0)
        self.assertCountEqual(['Not(And(Neighbor_Size_Of(Obj) == SIZE.4, Or(Color_Of(Obj) == FColor.fuchsia, Degree_Of(Obj) == DEGREE.2)))',
        'And(Neighbor_Size_Of(Obj) == SIZE.4, Not(Color_Of(Obj) == FColor.green))'], cf0)

        print("==================================================MOVEMENT PROBLEMS==================================================")
        print("Solving problem 25ff71a9")
        mt0, mf0 = run_synthesis("25ff71a9", "nbccg")
        self.assertCountEqual(['moveNode(Dir.DOWN)'], mt0)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf0)

        #print("Solving problem 1e0a9b12")
        #mt1, mf1 = run_synthesis("1e0a9b12", "nbccg")
        #self.assertCountEqual(['[moveNodeMax(Dir.DOWN), moveNodeMax(Dir.DOWN)]'], mt1)
        #self.assertCountEqual(['FColor.black == FColor.black)'], mf1)

        print("Solving problem e9afcf9a")
        mt1, mf1 = run_synthesis("e9afcf9a", "nbvcg")
        self.assertCountEqual(['moveNode(Dir.DOWN)', 'NoOp'], mt1)
        self.assertCountEqual(['Column_Of(Object.this) == COLUMN.EVEN', 
                            'Or(Column_Of(Object.this) == COLUMN.CENTER, Column_Of(Object.this) == COLUMN.ODD)'], mf1)

        print("Solving problem 3906de3d")
        mt10, mf10 = run_synthesis("3906de3d", "nbvcg")
        self.assertCountEqual(['moveNodeMax(Dir.UP)'], mt10)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf10)

        print("Solving problem ce22a75a")
        mt11, mf11 = run_synthesis("ce22a75a", "nbccg")
        self.assertCountEqual(
            ['addBorder(Color.blue)', 'updateColor(Color.blue)'], mt11)
        self.assertCountEqual(
            ['Equals(FColor.black, FColor.black)', 'Equals(FColor.blue, FColor.black)'], mf11)

        print("Solving problem 42a50994 with diagonal abstraction")
        vt01, vf01 = run_synthesis("42a50994", "nbccgm")
        self.assertCountEqual(['NoOp', 'updateColor(Color.black)'], vt01)
        self.assertCountEqual(['Not(Size_Of(Object.this) == SIZE.MIN)', 'Size_Of(Object.this) == SIZE.MIN'], vf01)

        print("Solving problem 3c9b0459")
        mt2, mf2 = run_synthesis("3c9b0459", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt2)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf2)

        print("Solving problem 6150a2bd")
        mt3, mf3 = run_synthesis("6150a2bd", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CW2)'], mt3)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf3)

        print("Solving problem 9dfd6313")
        mt4, mf4 = run_synthesis("9dfd6313", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt4)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf4)

        print("Solving problem 67a3c6ac")
        mt5, mf5 = run_synthesis("67a3c6ac", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.HORIZONTAL)'], mt5)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf5)

        print("Solving problem 74dd1130")
        mt6, mf6 = run_synthesis("74dd1130", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.DIAGONAL_LEFT)'], mt6)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf6)

        print("Solving problem ed36ccf7")
        mt7, mf7 = run_synthesis("ed36ccf7", "na")
        self.assertCountEqual(['rotateNode(Rotation_Angle.CCW)'], mt7)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], mf7)

        print("Solving problem 68b16354")
        t8, f8 = run_synthesis("68b16354", "na")
        self.assertCountEqual(['flip(Symmetry_Axis.VERTICAL)'], t8)
        self.assertCountEqual(['Equals(FColor.black, FColor.black)'], f8)

        #print("Solving problem a79310a0")
        #mt9, mf9 = run_synthesis("a79310a0", "nbccg")
        #self.assertCountEqual(
        #['[updateColor(Color.red), moveNode(Dir.DOWN)]'], mt9)
        #self.assertCountEqual(['Color_Of(Obj) == FColor.cyan'], mf9)

        print("==================================================VARIABLE PROBLEMS==================================================")
        # there is one correct assignment for the variables and the filters should convey that
        print("Solving problem 6855a6e4")
        vt0, vf0 = run_synthesis("6855a6e4", "nbccg")
        self.assertCountEqual(['mirror(Var.mirror_axis)'], vt0)
        self.assertCountEqual(['And(Color_Of(Obj) == FColor.grey, And(Direct_Neighbor_Of(Obj) == X, Color_Of(X) == FColor.red))'], vf0)

        print("Solving problem dc433765")
        vt3, vf3 = run_synthesis("dc433765", "nbccg")
        self.assertCountEqual(['moveNode(Dir.Variable)'], vt3)
        self.assertCountEqual(['And(Color_Of(Obj) == FColor.green, And(Neighbor_Of(Obj) == X, Color_Of(X) == FColor.yellow))'], vf3)

        print("Solving problem a48eeaf7")
        vt6, vf6 = run_synthesis("a48eeaf7", "nbccg")
        self.assertCountEqual(['moveNodeMax(Dir.Variable)'], vt6)
        self.assertCountEqual(['And(Color_Of(Obj) == FColor.grey, And(Neighbor_Of(Obj) == X, Color_Of(X) == FColor.red))'], vf6)

        print("Solving problem ddf7fa4f")
        vt1, vf1 = run_synthesis("ddf7fa4f", "nbccg")
        self.assertCountEqual(['updateColor(Var.color)_137'], vt1)
        self.assertCountEqual(['And(Color_Of(Obj) == FColor.grey, And(Direct_Neighbor_Of(Obj) == X, Size_Of(X) == SIZE.MIN))'], vf1)

        print("Solving problem d43fd935")
        vt5, vf5 = run_synthesis("d43fd935", "nbccg")
        self.assertCountEqual(['extendNode(Var.direction, Overlap.TRUE)_64'], vt5)
        self.assertCountEqual(['And(Direct_Neighbor_Of(Obj) == X, Color_Of(Object.var) == FColor.green)'], vf5)

        print("Solving problem 2c608aff")
        vt6, vf6 = run_synthesis("2c608aff", "ccgbr")
        self.assertCountEqual(['extendNode(Var.direction, Overlap.TRUE)_2'], vt6)
        self.assertCountEqual(['And(Direct_Neighbor_Of(Obj) == X, Size_Of(Object.var) == SIZE.MAX'], vf6)

        print("Solving problem 05f2a901")
        vt7, vf7 = run_synthesis("05f2a901", "nbccg")
        self.assertCountEqual(['moveNodeMax(Var.direction)'], vt7) #todo - eusolver
        self.assertCountEqual(['moveNodeMax(Dir.DOWN)', 'moveNodeMax(Dir.UP)', 'moveNode(Dir.Variable)_0', 'moveNodeMax(Dir.RIGHT)'], vt7)
        self.assertCountEqual(['Color_Of(Object.this) == FColor.red', 'Color_Of(Object.this) == FColor.red', 'And(Neighbor_Of(Obj) == X, Color_Of(Object.this) == FColor.red)', 'Color_Of(Object.this) == FColor.red'], vf7)

        print("Solving problem ae3edfdc")
        vt4, vf4 = run_synthesis("ae3edfdc", "nbccg")
        self.assertCountEqual(['moveNodeMax(Var.direction)_421'], vt4)
        self.assertCountEqual(['And(Direct_Neighbor_Of(Obj) == X, Or(Color_Of(Object.var) == FColor.blue, Color_Of(Object.var) == FColor.red))'], vf4)

        print("Solving problem f8a8fe49")
        #9edfc990 -- ccg with blue neighbor
        #vt2, vf2 = run_synthesis("f8a8fe49", "nbccg")
        #self.assertCountEqual(['mirror(Var.mirror_axis)'], vt2)
        #self.assertCountEqual(['And(Color_Of(Obj) == FColor.grey, VarAnd(Var.IsNeighbor, Var.Color_Of(Obj) == FColor.red))'], vf2)

        print("Solving problem ded97339")
        #vt8, vf8 = run_synthesis("ded97339", "nbccg") #todo
        #self.assertCountEqual(['extendNode(Var.direction, Overlap.TRUE)'], vt8)

#3618c87e: nbccg
#transformations: ['[flip(Symmetry_Axis.DIAGONAL_LEFT), moveNode(Dir.DOWN_LEFT)]', "insert(('OBJECT_ID.0', 'ImagePoints.BOTTOM_RIGHT', 'RelativePosition.TARGET'))", '[rotateNode(Rotation_Angle.CW2), moveNodeMax(Dir.DOWN)]', '[moveNode(Dir.DOWN), updateColor(Color.grey)]', "insert(('OBJECT_ID.1', 'ImagePoints.BOTTOM', 'RelativePosition.MIDDLE'))", "insert(('OBJECT_ID.1', 'ImagePoints.BOTTOM_RIGHT', 'RelativePosition.MIDDLE'))", '[flip(Symmetry_Axis.DIAGONAL_RIGHT), moveNodeMax(Dir.DOWN)]', '[flip(Symmetry_Axis.VERTICAL), addBorder(Color.grey)]']
#filters: ['Not(FilterBySize(SIZE.7))', 'Size_Of(Obj) == SIZE.6', 'Not(Color_Of(Obj) == FColor.Black)', 'Not(Color_Of(Obj) == FColor.Black)', 'Color_Of(Obj) == FColor.grey', 'Color_Of(Obj) == FColor.grey', 'Not(And(Degree_Of(Obj) == DEGREE.2, Or(Column_Of(Obj) == COLUMN.EVEN, Column_Of(Obj) == COLUMN.MOD3)))', 'FilterBySize(SIZE.7)']

#6c434453: nbccg # todo
#transformations: ['mirror(Var.mirror_axis)_0', "insert(('OBJECT_ID.0', 'ImagePoints.TOP', 'RelativePosition.SOURCE'))"]
#filters: ['And(Size_Of(Obj) == SIZE.MAX, VarAnd(Var.IsDirectNeighbor, Var.Or(Neighbor_Size(Obj) == SIZE.MIN, FilterByNeighborSize(SIZE.ODD))))', 'Or(Size_Of(Obj) == SIZE.MAX, FilterBySize(SIZE.4))']

if __name__ == "__main__":
    unittest.main()