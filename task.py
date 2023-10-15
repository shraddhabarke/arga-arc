import json
import os
from inspect import signature
from itertools import product

from utils import *
from image import Image
from ARCGraph import ARCGraph
from transform import *
from filters import *

class Task:
    #all_possible_abstractions = Image.abstractions
    #all_possible_transformations = ARCGraph.transformation_ops

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

    def get_candidate_filters(self):
        """
        return list of candidate filters
        """
        ret_apply_filter_calls = []  # final list of filter calls
        # use this list to avoid filters that return the same set of nodes
        filtered_nodes_all = []

        self.input_abstracted_graphs_original[self.abstraction] = [getattr(
            input, Image.abstraction_ops[self.abstraction])() for input in self.train_input]

        for filter_op in ARCGraph.filter_ops:
            # first, we generate all possible values for each parameter
            sig = signature(getattr(ARCGraph, filter_op))
            generated_params = []
            for param in sig.parameters:
                param_name = sig.parameters[param].name
                param_type = sig.parameters[param].annotation
                param_default = sig.parameters[param].default
                if param_name == "self" or param_name == "node":
                    continue
                if param_name == "color":
                    generated_params.append(
                        [c for c in range(10)] + ["most", "least"])
                elif param_name == "size":
                    generated_params.append(
                        [w for w in self.object_sizes[self.abstraction]] + ["min", "max", "odd"])
                elif param_name == "degree":
                    generated_params.append(
                        [d for d in self.object_degrees[self.abstraction]] + ["min", "max", "odd"])
                elif param_type == bool:
                    generated_params.append([True, False])
                elif issubclass(param_type, Enum):
                    generated_params.append([value for value in param_type])

            # then, we combine all generated values to get all possible combinations of parameters
            for item in product(*generated_params):

                # generate dictionary, keys are the parameter names, values are the corresponding values
                param_vals = {}
                # skip "self", "node"
                for i, param in enumerate(list(sig.parameters)[2:]):
                    param_vals[sig.parameters[param].name] = item[i]
                candidate_filter = {"filters": [
                    filter_op], "filter_params": [param_vals]}

                #  do not include if the filter result in empty set of nodes (this will be the majority of filters)
                filtered_nodes = []
                applicable_to_all = True
                for input_abstracted_graph in self.input_abstracted_graphs[self.abstraction]:
                    filtered_nodes_i = []
                    for node in input_abstracted_graph.graph.nodes():
                        if input_abstracted_graph.apply_filters(node, **candidate_filter):
                            filtered_nodes_i.append(node)
                    if len(filtered_nodes_i) == 0:
                        applicable_to_all = False
                    filtered_nodes.extend(filtered_nodes_i)
                filtered_nodes.sort()
                # does not result in empty or duplicate set of nodes
                if applicable_to_all and filtered_nodes not in filtered_nodes_all:
                    ret_apply_filter_calls.append(candidate_filter)
                    filtered_nodes_all.append(filtered_nodes)

        return ret_apply_filter_calls

    def parameters_generation(self, transform_sig):
        """
        given a transformation, generate parameters to be passed to the transformation
        :param all_calls: all apply filter calls, this is used to generate the dynamic parameters
        :param transform_sig: signature for a transformation
        :return: parameters to be passed to the transformation
        """
        generated_params = []
        for param in transform_sig.parameters:
            param_name = transform_sig.parameters[param].name
            param_type = transform_sig.parameters[param].annotation
            param_default = transform_sig.parameters[param].default
            if param_name == "self" or param_name == "node":  # nodes are already generated using the filters
                continue

            # first we generate the static values
            if param_name == "color":
                all_possible_values = [
                    c for c in range(10)] + ["most", "least"]
            elif param_name == "fill_color" or param_name == "border_color":
                all_possible_values = [c for c in range(10)]
            elif param_name == "object_id":
                all_possible_values = [id for id in range(len(self.static_objects_for_insertion[self.abstraction]))] + [
                    -1]
            # for insertion, could be ImagePoints or a coordinate on image (tuple)
            elif param_name == "point":
                all_possible_values = [value for value in ImagePoints]
            elif issubclass(param_type, Enum):
                all_possible_values = [value for value in param_type]
            elif param_type == bool:
                all_possible_values = [True, False]
            elif param_default is None:
                all_possible_values = [None]
            else:
                all_possible_values = []
            # TODO: Currently not dealing with dynamic parameter values
            generated_params.append(all_possible_values)

        return generated_params

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
        [abstracted_graph.apply_transform(transform)
         for abstracted_graph in self.input_abstracted_graphs_original[abstraction]]
        
        return self.input_abstracted_graphs_original[abstraction], self.output_abstracted_graphs_original[abstraction]

    def apply_rule(self, filter: FilterASTNode, transformation: TransformASTNode, abstraction, save_images=False):
        """
        apply solution abstraction and apply_call to test image
        """
        self.abstraction = abstraction
        self.input_abstracted_graphs_original[abstraction] = [getattr(input, Image.abstraction_ops[abstraction])() for
                                                              input in self.train_input]
        self.output_abstracted_graphs_original[abstraction] = [getattr(output, Image.abstraction_ops[abstraction])() for
                                                               output in self.train_output]
        self.get_static_inserted_objects()
        [abstracted_graph.apply_all(filter, transformation)
         for abstracted_graph in self.input_abstracted_graphs_original[abstraction]]
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
        print("The solution found predicted {} out of {} pixels incorrectly".format(error, len(self.test_output[0].graph.nodes())))
        return False

    def transform_values(self, filter: FilterASTNode, transformation: TransformASTNode, abstraction):
        """
        Returns the values of the transformed grid
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
        return reconstructed.graph.nodes(data=True)

    def filter_values(self, filter: FilterASTNode):
        filtered_nodes = []
        for input_abstracted_graph in self.input_abstracted_graphs_original[self.abstraction]:
            filtered_nodes_i = []
            for node in input_abstracted_graph.graph.nodes():
                if input_abstracted_graph.apply_filters(node, filter):
                    filtered_nodes_i.append(node)
            filtered_nodes.append(filtered_nodes_i)
        return filtered_nodes