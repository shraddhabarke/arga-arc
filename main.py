from task import *
import matplotlib.pyplot as plt
from collections import Counter
from inspect import signature
from transform import *
from filter import *

if __name__ == "__main__":
    taskNumber = "bb43febb"
    task = Task("dataset/" + taskNumber + ".json")
    task.abstraction = "nbccg"
    move_node_instance = MoveNode(Direction.LEFT)    
    update_color_instance = UpdateColor(Color.C2)
    add_border_instance = AddBorder(FillColor.C1)
    filterinstance = FilterByColor(Color.C5, Exclude.FALSE)
    transformed_graph, output_graph = task.apply_transformation(update_color_instance, task.abstraction) # applying here
    
    # testing transforms
    for iter in range(len(transformed_graph)):
        print("iter!", iter)
        actual = transformed_graph[iter]
        expected = output_graph[iter]
        for a, e in zip(actual.graph.nodes(data=True), expected.graph.nodes(data=True)):
            print("actual:", a[0], a[1])
            print("expected:", e[0], e[1])

    for abstracted_graph in task.input_abstracted_graphs_original[task.abstraction]:
        for trans in abstracted_graph.graph.nodes(data=True):
            print("trans-after:", trans[0], trans[1])

    # testing filters
    training_in = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                   input in task.train_input]
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                              input in task.train_input]
    task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                               output in task.train_output]
    for graph in task.input_abstracted_graphs_original[task.abstraction][0].graph.nodes(data=True):
        print("this one:", graph[0], graph[1])

    # Example usage:
    subsets = task.input_abstracted_graphs_original[task.abstraction][0].get_all_subsets()
    for subset in subsets:
        print("subset", subset)
    task.get_static_inserted_objects()
    task.get_static_object_attributes(task.abstraction)
    filtered_nodes = []
    for input_abstracted_graph in task.input_abstracted_graphs_original[task.abstraction]:
        filtered_nodes_i = []
        for node in input_abstracted_graph.graph.nodes():
            if input_abstracted_graph.apply_filters(node, filterinstance):
                filtered_nodes_i.append(node)
        filtered_nodes.extend(filtered_nodes_i)
        print("iter:", filtered_nodes_i)
    
    # testing full rule
    print("Testing full rule:")
    transformed_graph, output_graph = task.apply_rule(filterinstance, add_border_instance, task.abstraction)
    for iter in range(len(transformed_graph)):
        print("iter!", iter)
        actual = transformed_graph[iter]
        expected = output_graph[iter]
        for a, e in zip(actual.graph.nodes(data=True), expected.graph.nodes(data=True)):
            print("actual:", a[0], a[1])
            print("expected:", e[0], e[1])

    # Testing a sequence of transforms and filters

    transforms_list = [UpdateColor(Color.C3), MoveNode(Direction.UP)]
    trans_sequence = Transforms(transforms_list)
    print(trans_sequence.code)