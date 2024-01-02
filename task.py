import json
import os
from inspect import signature
from itertools import product

from utils import *
from image import Image
from ARCGraph import ARCGraph
from transform import *
from filters import *
from transform import *


class Task:
    # all_possible_abstractions = Image.abstractions
    # all_possible_transformations = ARCGraph.transformation_ops

    def __init__(self, filepath):
        """
        contains all information related to an ARC task
        """
        # get task id from filepath
        self.task_id = filepath.split("/")[-1].split(".")[0]
        # input output images given
        self.train_input = []
        self.train_output = []
        self.test_input = []
        self.test_output = []
        # abstracted graphs from input output images
        # a dictionary of ARCGraphs, where the keys are the abstraction name and
        self.input_abstracted_graphs = dict()
        # values are lists of ARCGraphs with the abs name for all inputs/outputs
        self.output_abstracted_graphs = dict()
        # a dictionary of ARCGraphs, where the keys are the abstraction name and
        self.input_abstracted_graphs_original = dict()
        self.output_abstracted_graphs_original = dict()
        self.abstraction = None  # which type of abstraction the search is currently working with
        # static objects used for the "insert" transformation
        self.static_objects_for_insertion = dict()
        self.object_sizes = dict()  # object sizes to use for filters
        self.object_degrees = dict()  # object degrees to use for filters
        self.load_task_from_file(filepath)

    def load_task_from_file(self, filepath):
        """
        loads the task from a json file
        """
        with open(filepath) as f:
            data = json.load(f)
        for i, data_pair in enumerate(data["train"]):
            self.train_input.append(
                Image(self, grid=data_pair["input"], name=self.task_id + "_" + str(i + 1) + "_train_in"))
            self.train_output.append(
                Image(self, grid=data_pair["output"], name=self.task_id + "_" + str(i + 1) + "_train_out"))
        for i, data_pair in enumerate(data["test"]):
            self.test_input.append(
                Image(self, grid=data_pair["input"], name=self.task_id + "_" + str(i + 1) + "_test_in"))
            self.test_output.append(
                Image(self, grid=data_pair["output"], name=self.task_id + "_" + str(i + 1) + "_test_out"))

    # --------------------------------- Utility Functions ---------------------------------
    def get_static_inserted_objects(self):
        """
        populate self.static_objects_for_insertion, which contains all static objects detected in the images.
        """
        self.static_objects_for_insertion[self.abstraction] = []
        existing_objects = []

        for i, output_abstracted_graph in enumerate(self.output_abstracted_graphs_original[self.abstraction]):
            # difference_image = self.train_output[i].copy()
            input_abstracted_nodes = self.input_abstracted_graphs_original[self.abstraction][i].graph.nodes(
            )
            for abstracted_node, data in output_abstracted_graph.graph.nodes(data=True):
                if abstracted_node not in input_abstracted_nodes:
                    new_object = data.copy()
                    min_x = min([subnode[1]
                                for subnode in new_object["nodes"]])
                    min_y = min([subnode[0]
                                for subnode in new_object["nodes"]])
                    adjusted_subnodes = []
                    for subnode in new_object["nodes"]:
                        adjusted_subnodes.append(
                            (subnode[0] - min_y, subnode[1] - min_x))
                    adjusted_subnodes.sort()
                    if adjusted_subnodes not in existing_objects:
                        existing_objects.append(adjusted_subnodes)
                        self.static_objects_for_insertion[self.abstraction].append(
                            new_object)

    def get_static_object_attributes(self, abstraction):
        """
        populate self.object_sizes and self.object_degrees, which contains all sizes and degrees existing objects
        """
        self.object_sizes[abstraction] = set()
        self.object_degrees[abstraction] = set()
        for abs_graph in self.input_abstracted_graphs_original[abstraction]:
            for node, size in abs_graph.graph.nodes(data="size"):
                self.object_sizes[abstraction].add(size)
            for node, degree in abs_graph.graph.degree():
                self.object_degrees[abstraction].add(degree)

    def apply_transformation(self, transform: TransformASTNode, abstraction):
        """
        apply transformation rule to training images without filtering
        """
        self.abstraction = abstraction
        self.input_abstracted_graphs_original[abstraction] = [getattr(input, Image.abstraction_ops[abstraction])() for
                                                              input in self.train_input]
        self.output_abstracted_graphs_original[abstraction] = [getattr(output, Image.abstraction_ops[abstraction])() for
                                                               output in self.train_output]
        self.get_static_inserted_objects()
        self.get_static_object_attributes(self.abstraction)
        children = transform.childTypes
        if len(children) > 0 and children[0] == Types.TRANSFORMS:
            transformations = [transform.children[0], transform.children[1]]
        else:
            transformations = [transform]
        if transformations is not None:
            for abstracted_graph in self.input_abstracted_graphs_original[abstraction]:
                for transformation in transformations:
                    abstracted_graph.apply_transform(transformation)
        return self.input_abstracted_graphs_original[abstraction], self.output_abstracted_graphs_original[abstraction]

    def output_matches(self, filter: FilterASTNode, transformation: TransformASTNode, abstraction):
        """
        Returns whether the output of the filter, transform pair matches the expected output
        """
        self.abstraction = abstraction
        test_input = self.test_input[0]
        test_abstracted_graph = getattr(
            test_input, Image.abstraction_ops[abstraction])()
        test_abstracted_graph.apply_all(filter, transformation)
        reconstructed = test_input.undo_abstraction(test_abstracted_graph)
        # check if the solution found the correct test output
        error = 0
        for node, data in self.test_output[0].graph.nodes(data=True):
            if data["color"] != reconstructed.graph.nodes[node]["color"]:
                error += 1
        if error == 0:
            return True
        print("The solution found predicted {} out of {} pixels incorrectly".format(
            error, len(self.test_output[0].graph.nodes())))
        return False

    def transform_values(self, filter: FilterASTNode, transformations: list):
        """
        Returns the values of the transformed grid
        """
        self.input_abstracted_graphs_original[self.abstraction] = [getattr(
            input, Image.abstraction_ops[self.abstraction])() for input in self.train_input]

        if not isinstance(transformations, list):
            transformations = [transformations]
        transformed_values = []

        # TODO: some issue here for na, mcccg
        for train_input, input_abstracted_graph in zip(self.train_input, self.input_abstracted_graphs_original[self.abstraction]):
            for transformation in transformations:
                input_abstracted_graph.apply_all(filter, transformation)
            reconstructed = train_input.undo_abstraction(
                input_abstracted_graph)
            transformed_values.append(
                {node: data['color'] for node, data in reconstructed.graph.nodes(data=True)})
        return transformed_values

    def filter_values(self, filter: FilterASTNode):
        filtered_nodes = []
        self.input_abstracted_graphs_original[self.abstraction] = [getattr(
            input, Image.abstraction_ops[self.abstraction])() for input in self.train_input]
        for input_abstracted_graph in self.input_abstracted_graphs_original[self.abstraction]:
            filtered_nodes_i = []
            for node in input_abstracted_graph.graph.nodes(data=True):
                if input_abstracted_graph.apply_filters(node[0], filter):
                    filtered_nodes_i.extend(node[1]['nodes'])
            filtered_nodes.append(filtered_nodes_i)
        return filtered_nodes

    def var_transform_values(self, filter: FilterASTNode, transformations: list):
        """
        Returns the values of the transformed grid with different possibilities for variable transformation
        """
        self.input_abstracted_graphs_original[self.abstraction] = [getattr(
            input, Image.abstraction_ops[self.abstraction])() for input in self.train_input]
        self.output_abstracted_graphs_original[self.abstraction] = [getattr(
            input, Image.abstraction_ops[self.abstraction])() for input in self.train_output]
        if not isinstance(transformations, list):
            transformations = [transformations]

        def generate_cartesian_product(input_graph, output_graph):
            in_dict = input_graph.graph.nodes(data=True)
            out_dict = {node[0]: node[1]
                        for node in output_graph.graph.nodes(data=True)}
            matching_nodes = [node[0] for node in in_dict if node[0]
                              in out_dict and node[1]['color'] == out_dict[node[0]]['color']]
            # extract all colors from the input graph
            colors = set(item[1]['color'] for item in in_dict)
            # all possible valuations of color assignments for nodes which undergo change
            cartesian_product = (combo for combo in product(colors, repeat=len(
                in_dict) - len(matching_nodes)) if len(set(combo)) > 1)

            diff_nodes = set(input_graph.graph.nodes) - set(matching_nodes)
            # Generator that yields dictionaries for each combination of colors

            def combo_dicts_generator():
                for combo in cartesian_product:
                    yield {node: color for node, color in zip(diff_nodes, combo)}
            return combo_dicts_generator()

        transformed_values = []
        for train_input, input_graph, output_graph in zip(self.train_input, self.input_abstracted_graphs_original[self.abstraction], self.output_abstracted_graphs_original[self.abstraction]):
            color_combo = generate_cartesian_product(input_graph, output_graph)
            temp_transformed_values = []
            for color in color_combo:
                for transformation in transformations:
                    input_graph.varcolor_apply_all(
                        color, filter, transformation)
                reconstructed = train_input.undo_abstraction(input_graph)
                temp_transformed_values.append(
                    {node: data['color'] for node, data in reconstructed.graph.nodes(data=True)})
            transformed_values.append(temp_transformed_values)
        return list(product(*transformed_values))
