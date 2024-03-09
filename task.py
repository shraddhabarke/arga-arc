import json
import copy
from inspect import signature
from itertools import product
import typing as t

from image import Image
from ARCGraph import ARCGraph
from transform import *
from filters import *
from transform import *
from collections import defaultdict


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
        self.abstraction = (
            None  # which type of abstraction the search is currently working with
        )
        # static objects used for the "insert" transformation
        self.static_objects_for_insertion = dict()
        self.object_sizes = dict()  # node_object sizes to use for filters
        self.object_heights = dict()
        self.object_degrees = dict()  # node_object degrees to use for filters
        self.object_heights = dict()
        self.object_widths = dict()
        self.load_task_from_file(filepath)
        self.spec = None  # for variable filter synthesis
        self.current_spec = []

    def load_task_from_file(self, filepath):
        """
        loads the task from a json file
        """
        with open(filepath) as f:
            data = json.load(f)
        for i, data_pair in enumerate(data["train"]):
            self.train_input.append(
                Image(
                    self,
                    grid=data_pair["input"],
                    name=self.task_id + "_" + str(i + 1) + "_train_in",
                )
            )
            self.train_output.append(
                Image(
                    self,
                    grid=data_pair["output"],
                    name=self.task_id + "_" + str(i + 1) + "_train_out",
                )
            )
        for i, data_pair in enumerate(data["test"]):
            self.test_input.append(
                Image(
                    self,
                    grid=data_pair["input"],
                    name=self.task_id + "_" + str(i + 1) + "_test_in",
                )
            )
            self.test_output.append(
                Image(
                    self,
                    grid=data_pair["output"],
                    name=self.task_id + "_" + str(i + 1) + "_test_out",
                )
            )

    # --------------------------------- Utility Functions ---------------------------------
    def get_static_inserted_objects(self):
        """
        populate self.static_objects_for_insertion, which contains all static objects detected in the images.
        """
        self.static_objects_for_insertion[self.abstraction] = []
        existing_objects = []

        for i, output_abstracted_graph in enumerate(
            self.output_abstracted_graphs_original[self.abstraction]
        ):
            # difference_image = self.train_output[i].copy()
            input_abstracted_nodes = self.input_abstracted_graphs_original[
                self.abstraction
            ][i].graph.nodes()
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
                            (subnode[0] - min_y, subnode[1] - min_x)
                        )
                    adjusted_subnodes.sort()
                    if adjusted_subnodes not in existing_objects:
                        existing_objects.append(adjusted_subnodes)
                        self.static_objects_for_insertion[self.abstraction].append(
                            new_object
                        )

    def get_static_object_attributes(self, abstraction):
        """
        populate self.object_sizes and self.object_degrees, which contains all sizes and degrees existing objects
        """
        self.object_sizes[abstraction] = set()
        self.object_degrees[abstraction] = set()
        self.object_heights[abstraction] = set()
        self.object_widths[abstraction] = set()
        for abs_graph in self.input_abstracted_graphs_original[abstraction]:
            for node, size in abs_graph.graph.nodes(data="size"):
                self.object_sizes[abstraction].add(size)
            for node, degree in abs_graph.graph.degree():
                self.object_degrees[abstraction].add(degree)
            for node, height in abs_graph.graph.nodes(data="height"):
                self.object_heights[abstraction].add(height)
            for node, width in abs_graph.graph.nodes(data="width"):
                self.object_widths[abstraction].add(width)

    def evaluate_program(
        self,
        program: t.List[t.Tuple[FilterASTNode, t.List[TransformASTNode]]],
        abstraction: str,
    ) -> t.List[ARCGraph]:
        self.abstraction = abstraction
        test_inputs = self.test_input
        transformed_inputs = []
        for idx, test_input in enumerate(test_inputs):
            test_abstracted_graph: ARCGraph = getattr(
                test_input, Image.abstraction_ops[abstraction]
            )()
            for filter, transformations in program:
                for transformation in transformations:
                    test_abstracted_graph.apply_all(filter, transformation)
            reconstructed = test_input.undo_abstraction(test_abstracted_graph)

            transformed_inputs.append(reconstructed)
        return transformed_inputs

    def test_program(
        self,
        program: t.List[t.Tuple[FilterASTNode, TransformASTNode]],
        abstraction: str,
    ) -> bool:
        """applies all filters and transformations to the test set and checks if the output matches the expected output"""
        transformed_inputs = self.evaluate_program(program, abstraction)
        for idx, transformed_input in enumerate(transformed_inputs):
            error = self.__graph_diff_px(
                transformed_input, self.test_output[idx])
            if error != 0:
                # print(
                #     f"Test failed for {self.task_id} test {idx + 1} with error {error}"
                # )
                return False
        return True

    def output_matches(
        self, filter: FilterASTNode, transformation: TransformASTNode, abstraction
    ):
        """
        Returns whether the output of the filter, transform pair matches the expected output
        """
        self.abstraction = abstraction
        test_input = self.test_input[0]
        test_abstracted_graph = getattr(
            test_input, Image.abstraction_ops[abstraction]
        )()
        test_abstracted_graph.apply_all(filter, transformation)
        reconstructed = test_input.undo_abstraction(test_abstracted_graph)
        # check if the solution found the correct test output
        error = self.__graph_diff_px(reconstructed, self.test_output[0])
        if error == 0:
            return True
        print(
            "The solution found predicted {} out of {} pixels incorrectly".format(
                error, len(self.test_output[0].graph.nodes())
            )
        )
        return False

    def __graph_diff_px(self, g1, g2) -> int:
        error = 0
        for node, data in g1.graph.nodes(data=True):
            if data["color"] != g2.graph.nodes[node]["color"]:
                error += 1
        return error

    def flatten_transforms(self, transform) -> t.List:
        """
        Recursively flattens nested transform.Transforms objects into a single list of transformations.
        """
        all_transforms = []
        if transform.childTypes == [Types.TRANSFORMS, Types.TRANSFORMS]:
            for sub_transform in transform.children:
                # If the child is also a transform.Transforms, recurse
                all_transforms.extend(self.flatten_transforms(sub_transform))
        else:
            # Base case: No further nesting, add the transform directly
            all_transforms.append(transform)
        return all_transforms

    def sequence_transform_values(self, filter: FilterASTNode, transformation: Transforms):
        """
        Returns the values of the transformed grid, takes a sequence of transformations
        """
        self.input_abstracted_graphs_original[self.abstraction] = [
            getattr(input, Image.abstraction_ops[self.abstraction])()
            for input in self.train_input
        ]
        all_transforms = []
        for transform in transformation:
            all_transforms.extend(self.flatten_transforms(transform))
        print("all-transform",
              [transform.code for transform in all_transforms])
        og_graph = self.input_abstracted_graphs_original[self.abstraction]
        # TODO: some issue here for na, mcccg
        for idx, transform in enumerate(all_transforms):
            if "Var" in transform.code:
                transformed_values, spec = self.compute_all_transformed_values(
                    filter, transform, og_graph)
                for values, input_abstracted_graph in zip(transformed_values, og_graph):
                    input_abstracted_graph.var_apply_all(
                        values, filter, transform)
            else:
                for input_abstracted_graph in og_graph:
                    nodes_list = list(input_abstracted_graph.graph.nodes())
                    for node in nodes_list:
                        # bit hacky
                        if len(node) == 3 and "moveNode" in all_transforms[idx-1].code:
                            input_abstracted_graph.remove_node(node)
                for input_abstracted_graph in og_graph:
                    input_abstracted_graph.apply_all(filter, transform)
        return og_graph

    def transform_values(
            self, filter: FilterASTNode, transformation: TransformASTNode
    ):
        """
        Returns the values of the transformed grid, takes a single transformation
        """
        self.spec = None
        self.input_abstracted_graphs_original[self.abstraction] = [
            getattr(input, Image.abstraction_ops[self.abstraction])()
            for input in self.train_input
        ]

        for input_abstracted_graph in self.input_abstracted_graphs_original[self.abstraction]:
            input_abstracted_graph.apply_all(filter, transformation)

        return self.input_abstracted_graphs_original[self.abstraction]

    def filter_values(self, filter: FilterASTNode):
        filtered_nodes, filtered_nodes_dict_list = [], []
        self.input_abstracted_graphs_original[self.abstraction] = [
            getattr(input, Image.abstraction_ops[self.abstraction])()
            for input in self.train_input
        ]

        for input_abstracted_graph in self.input_abstracted_graphs_original[
            self.abstraction
        ]:
            filtered_nodes_i = []
            for node in input_abstracted_graph.graph.nodes(data=True):
                if input_abstracted_graph.apply_filters(node[0], filter):
                    filtered_nodes_i.append(node[0])
            filtered_nodes.append(filtered_nodes_i)
        for i, _ in enumerate(filtered_nodes):
            filtered_nodes_dict = {node: [] for node in filtered_nodes[i]}
            filtered_nodes_dict_list.append(filtered_nodes_dict)
        if self.spec: # todo
            return self.spec
        else:
            return filtered_nodes_dict_list

    def compute_transformation_params(self, input_graph, transformation):
        object_params_dict, object_params = defaultdict(list), []
        if "updateColor" in transformation.code:
            object_params = set([input_graph.get_color(obj)
                                for obj in input_graph.graph.nodes()])
            for node_obj in input_graph.graph.nodes():
                for neighbor in input_graph.graph.neighbors(node_obj):
                    object_params_dict[node_obj].append(
                    (input_graph.get_color(neighbor), neighbor))

        elif "extendNode" in transformation.code or "moveNode" in transformation.code \
                or "moveNodeMax" in transformation.code:
            for node_obj in input_graph.graph.nodes():
                for node_other, _ in input_graph.graph.nodes(data=True):
                    if node_obj != node_other:
                        relative_pos = input_graph.get_relative_pos(
                            node_obj, node_other)
                        object_params.append(relative_pos)
                        if relative_pos is not None:
                            object_params_dict[node_obj].append(
                                (relative_pos, node_other))
        elif "Flip" in transformation.code:
            for node_obj in input_graph.graph.nodes():
                for node_other, _ in input_graph.graph.nodes(data=True):
                    if node_obj != node_other:
                        target_mirror_dir = self.get_mirror_direction(
                            node_obj, node_other)
                        object_params.append(target_mirror_dir)
                        if target_mirror_dir is not None:
                            object_params_dict[node_obj].append(
                                (target_mirror_dir, node_other))
        elif "Insert" in transformation.code:
            print("transformation-insert:", transformation.code)
            object_params = set([input_graph.get_centroid(obj)
                                for obj in input_graph.graph.nodes()])
            for node_obj in input_graph.graph.nodes():
                for node_other in input_graph.graph.nodes():
                    if node_obj != node_other:
                        object_params_dict[node_obj].append((input_graph.get_centroid(node_other), node_other))
        elif "mirror" in transformation.code:
            print("transformation-mirror:", transformation.code)
            for node_obj in input_graph.graph.nodes():
                for neighbor in input_graph.graph.nodes():
                    if node_obj != neighbor:
                        target_axis = input_graph.get_mirror_axis(
                            node_obj, neighbor)
                        object_params.append(target_axis)
                        if target_axis is not None:
                            object_params_dict[node_obj].append(
                                (target_axis, neighbor))
            print("object_params_dict", object_params_dict)
        return object_params_dict, set(object_params)

    def compute_transformed_values(self, diff_nodes, input_graph, output_graph, filter, transformation):
        per_task = {}
        per_task_spec = {}
        output_nodes_set = {tuple(out_props["nodes"]): out_props
                            for _, out_props in output_graph.graph.nodes(data=True)}
        coordinate_color_map = {coordinate: value['color']
                        for key, value in output_nodes_set.items()
                        for coordinate in key}
        object_params_dict, object_params = self.compute_transformation_params(
            input_graph, transformation)
        for node_object in diff_nodes:
            print("==========================Applying the transform==========================")
            print("node-object:", node_object, input_graph.graph.nodes[node_object]['nodes'])            
            for param in set(object_params_dict[node_object]):
                if param is not None:
                    input_graph_copy = copy.deepcopy(input_graph)
                    input_graph_copy.var_apply_all(
                        dict({node_object: param[0]}), filter, transformation
                    )
                    new_object_data = input_graph_copy.graph.nodes[node_object]
                    new_object_nodes = tuple(
                        set(input_graph_copy.graph.nodes[node_object]["nodes"]))
                    print("==========================Computing the output nodes==========================")
                    print("param being considered:", param)
                    print("new_object_data:", new_object_data)
                    print("new_object_nodes:", new_object_nodes, bool(new_object_nodes))
                    print("======================================================================================================")
                    if bool(new_object_nodes) and all(new_object_data["color"] == coordinate_color_map.get(node, None)
                        for node in new_object_nodes): # checking if the variable parameter transforms that node/object correctly
                        # todo: this does not work for sequences of variable transforms!
                        per_task[node_object] = param[0]
                        if node_object not in per_task_spec: # adding all that are satisfied for filter synthesis later
                            per_task_spec[node_object] = [param[1]]
                        else:
                            per_task_spec[node_object].append(param[1])
                        print("per_task_spec:", per_task_spec)
        return per_task, per_task_spec

    def compute_all_transformed_values(self, filter, transformation, og_graph):
        all_transformed_values = []
        all_specs = []

        for input_graph, output_graph in zip(og_graph,
                                             self.output_abstracted_graphs_original[self.abstraction]):
            matching_nodes = [
                in_node for in_node, in_props in input_graph.graph.nodes(data=True)
                if any(in_props['color'] == out_props['color'] and
                       in_props['nodes'] == out_props['nodes'] and
                       in_props['size'] == out_props['size']
                       for _, out_props in output_graph.graph.nodes(data=True))
            ]

            diff_nodes = set(input_graph.graph.nodes) - set(matching_nodes)
            per_task, per_task_spec = self.compute_transformed_values(
                diff_nodes, input_graph, output_graph, filter, transformation)
            print("per-task-spec:", per_task_spec)
            all_transformed_values.append(per_task)
            if per_task_spec is not None:
                all_specs.append(per_task_spec)
        return all_transformed_values, all_specs

    def var_transform_values(
        self, filter: FilterASTNode, transformation: TransformASTNode
    ):
        """
        Returns the values of the transformed grid with different possibilities for variable transformation
        """
        if self.abstraction == "na":  # todo: does it sense to do variable for the na abstraction?
            return []

        self.input_abstracted_graphs_original[self.abstraction] = [
            getattr(input, Image.abstraction_ops[self.abstraction])()
            for input in self.train_input
        ]
        self.output_abstracted_graphs_original[self.abstraction] = [
            getattr(input, Image.abstraction_ops[self.abstraction])()
            for input in self.train_output
        ]
        transformed_values, spec = self.compute_all_transformed_values(
            filter, transformation, self.input_abstracted_graphs_original[self.abstraction])

        self.input_abstracted_graphs_original[self.abstraction] = [
            getattr(input, Image.abstraction_ops[self.abstraction])()
            for input in self.train_input
        ]

        for values, input_abstracted_graph in zip(transformed_values, self.input_abstracted_graphs_original[self.abstraction]):
            input_abstracted_graph.var_apply_all(
                values, filter, transformation)
        #self.spec.update({transformation.code: spec})
        self.spec = spec
        return self.input_abstracted_graphs_original[self.abstraction]