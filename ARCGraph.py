import copy
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
from utils import *
from filters import *
from transform import *
from task import *
import itertools
from typing import *
from OEValuesManager import *
from VocabMaker import *


class ARCGraph:
    def __init__(self, graph, name, image, abstraction=None):
        self.graph = graph
        self.image = image
        self.grid_size = self.image.image_size[0] * self.image.image_size[1]
        self.abstraction = abstraction
        if abstraction is None:
            self.name = name
        elif abstraction in name.split("_"):
            self.name = name
        else:
            self.name = name + "_" + abstraction
        if self.abstraction in image.multicolor_abstractions:
            self.is_multicolor = True
            self.most_common_color = 0
            self.least_common_color = 0
        else:
            self.is_multicolor = False
            self.most_common_color = image.most_common_color
            self.least_common_color = image.least_common_color

        self.width = max([node[1] for node in self.image.graph.nodes()]) + 1
        self.height = max([node[0] for node in self.image.graph.nodes()]) + 1
        self.task_id = name.split("_")[0]

    # ------------------------------------------ transformations ------------------------------------------
    def NoOp(self, node):
        return self

    def UpdateColor(self, node, color: Color):
        """
        update node color to given color
        """
        if color == "most":
            color = self.most_common_color
        elif color == "least":
            color = self.least_common_color
        color_map = {
            "O": 0,
            "B": 1,
            "R": 2,
            "G": 3,
            "Y": 4,
            "X": 5,
            "F": 6,
            "A": 7,
            "C": 8,
            "W": 9,
            "most": self.most_common_color,
            "least": self.least_common_color
        }
        if self.is_multicolor:
            if not isinstance(color, int):
                self.graph.nodes[node]["color"] = [color_map[color]] * sum(
                    len(data["nodes"]) for node, data in self.graph.nodes(data=True)
                )
            else:
                self.graph.nodes[node]["color"] = [color] * sum(
                    len(data["nodes"]) for node, data in self.graph.nodes(data=True)
                )
        else:
            if not isinstance(color, int):
                self.graph.nodes[node]["color"] = color_map[color]
            else:
                self.graph.nodes[node]["color"] = color
        return self

    def MoveNode(self, node, direction: Dir, n: Height = 1):
        """
        move node by 1 pixel in a given direction
        """
        assert direction is not None
        ogpixels = set()  # list of pixels which should go back to their original color!
        updated_sub_nodes = set()
        delta_x = 0
        delta_y = 0
        if direction == "U" or direction == Dir.UP:
            delta_y = -n
        elif direction == "D" or direction == Dir.DOWN:
            delta_y = n
        elif direction == "L" or direction == Dir.LEFT:
            delta_x = -n
        elif direction == "R" or direction == Dir.RIGHT:
            delta_x = n
        elif direction == Dir.DOWN_RIGHT or direction == "DR":
            delta_x = n
            delta_y = n
        elif direction == Dir.DOWN_LEFT or direction == "DL":
            delta_y = n
            delta_x = -n
        elif direction == Dir.UP_LEFT or direction == "UL":
            delta_x = -n
            delta_y = -n
        elif direction == Dir.UP_RIGHT or direction == "UR":
            delta_x = n
            delta_y = -n
        for sub_node in self.graph.nodes[node]["nodes"]:
            if sub_node[0] + delta_y == self.width or sub_node[1] + delta_x == self.height:
                updated_sub_nodes.add((sub_node[0], sub_node[1]))
            elif self.check_inbound((sub_node[0] + delta_y, sub_node[1] + delta_x)):
                updated_sub_nodes.add(
                    (sub_node[0] + delta_y, sub_node[1] + delta_x))

        for sub_node in self.graph.nodes[node]["nodes"]:
            if (sub_node[0] + delta_y == sub_node[0] or sub_node[0] + delta_y == self.width) \
                    and (sub_node[1] + delta_x == sub_node[1] or sub_node[1] + delta_x == self.height) or sub_node in updated_sub_nodes:
                continue
            else:
                ogpixels.add(sub_node)

        if ogpixels:
            new_node_id = self.generate_node_id(self.image.background_color)
            if self.is_multicolor:
                self.graph.add_node(
                    (node[0], node[1], 0),
                    nodes=list(ogpixels),
                    color=[self.image.background_color for _ in ogpixels],
                    size=len(ogpixels),
                )
            else:
                self.graph.add_node(
                    (node[0], node[1], 0),
                    nodes=list(ogpixels),
                    color=self.image.background_color,
                    size=len(ogpixels),
                )
        self.graph.nodes[node]["nodes"] = updated_sub_nodes
        self.graph.nodes[node]["size"] = len(updated_sub_nodes)

        return self

    def ExtendNode(self, node, direction: Dir, overlap: Overlap = False, n: int = 1):
        """
        extend node in a given direction,
        if overlap is true, extend node even if it overlaps with another node
        if overlap is false, stop extending before it overlaps with another node
        """
        assert direction is not None
        updated_sub_nodes = []
        delta_x = 0
        delta_y = 0
        if direction == "U" or direction == Dir.UP:
            delta_y = -n
        elif direction == "D" or direction == Dir.DOWN:
            delta_y = n
        elif direction == "L" or direction == Dir.LEFT:
            delta_x = -n
        elif direction == "R" or direction == Dir.RIGHT:
            delta_x = n
        elif direction == Dir.DOWN_RIGHT or direction == "DR":
            delta_x = n
            delta_y = n
        elif direction == Dir.DOWN_LEFT or direction == "DL":
            delta_y = n
            delta_x = -n
        elif direction == Dir.UP_LEFT or direction == "UL":
            delta_x = -n
            delta_y = -n
        elif direction == Dir.UP_RIGHT or direction == "UR":
            delta_x = n
            delta_y = -n
        for sub_node in self.graph.nodes[node]["nodes"]:
            sub_node_y = sub_node[0]
            sub_node_x = sub_node[1]
            max_allowed = 1000
            for foo in range(max_allowed):
                updated_sub_nodes.append((sub_node_y, sub_node_x))
                sub_node_y += delta_y
                sub_node_x += delta_x
                if overlap and not self.check_inbound((sub_node_y, sub_node_x)):
                    # if overlap allowed, stop extending node until hitting edge of image
                    break
                elif not overlap and (
                    self.check_collision(node, [(sub_node_y, sub_node_x)])
                    or not self.check_inbound((sub_node_y, sub_node_x))
                ):
                    # if overlap not allowed, stop extending node until hitting edge of image or another node
                    break
        self.graph.nodes[node]["nodes"] = list(set(updated_sub_nodes))
        self.graph.nodes[node]["size"] = len(updated_sub_nodes)
        return self

    def MoveNodeMax(self, node, direction: Dir, n: int = 1):
        """
        move node in a given direction until it hits another node or the edge of the image
        """
        ogpixels = set()  # list of pixels which should go back to their original color!
        assert direction is not None
        delta_x = 0
        delta_y = 0
        if direction == "U" or direction == Dir.UP:
            delta_y = -n
        elif direction == "D" or direction == Dir.DOWN:
            delta_y = n
        elif direction == "L" or direction == Dir.LEFT:
            delta_x = -n
        elif direction == "R" or direction == Dir.RIGHT:
            delta_x = n
        elif direction == Dir.DOWN_RIGHT or direction == "DR":
            delta_x = n
            delta_y = n
        elif direction == Dir.DOWN_LEFT or direction == "DL":
            delta_y = n
            delta_x = -n
        elif direction == Dir.UP_LEFT or direction == "UL":
            delta_x = -n
            delta_y = -n
        elif direction == Dir.UP_RIGHT or direction == "UR":
            delta_x = n
            delta_y = -n
        max_allowed = 1000
        from copy import deepcopy
        og_graph = deepcopy(self.graph.nodes[node]["nodes"])
        for foo in range(max_allowed):
            updated_nodes = set()
            for sub_node in self.graph.nodes[node]["nodes"]:
                updated_nodes.add(
                    (sub_node[0] + delta_y, sub_node[1] + delta_x))

            if self.check_collision(node, updated_nodes) or not self.check_inbound(
                list(updated_nodes)
            ):
                break
            self.graph.nodes[node]["nodes"] = list(updated_nodes)
        for sub_node in og_graph:
            if sub_node not in self.graph.nodes[node]["nodes"]:
                ogpixels.add(sub_node)

        if ogpixels:
            if self.is_multicolor:
                self.graph.add_node(
                    (node[0], node[1], 0),
                    nodes=list(ogpixels),
                    color=[self.image.background_color for _ in ogpixels],
                    size=len(ogpixels),
                )
            else:
                self.graph.add_node(
                    (node[0], node[1], 0),
                    nodes=list(ogpixels),
                    color=self.image.background_color,
                    size=len(ogpixels),
                )
        return self

    def RotateNode(self, node, rotation_dir: Rotation_Angle):
        """
        rotates node around its center point in a given rotational direction
        """
        mul = 0
        rotate_times = 1
        if rotation_dir == "270":
            mul = -1
        elif rotation_dir == "90":
            mul = 1
        elif rotation_dir == "180":
            rotate_times = 2
            mul = -1

        # Check if the node size is zero or the nodes list is empty to prevent division by zero
        if self.graph.nodes[node]["size"] == 0 or not self.graph.nodes[node]["nodes"]:
            print("Cannot rotate node due to zero size or no subnodes.")
            return self  # Exit the function early
        for t in range(rotate_times):
            center_point = (
                sum([n[0] for n in self.graph.nodes[node]["nodes"]])
                // self.graph.nodes[node]["size"],
                sum([n[1] for n in self.graph.nodes[node]["nodes"]])
                // self.graph.nodes[node]["size"],
            )
            new_nodes = []
            for sub_node in self.graph.nodes[node]["nodes"]:
                new_sub_node = (
                    sub_node[0] - center_point[0],
                    sub_node[1] - center_point[1],
                )
                new_sub_node = (-new_sub_node[1] * mul, new_sub_node[0] * mul)

                new_sub_node = (
                    new_sub_node[0] + center_point[0],
                    new_sub_node[1] + center_point[1],
                )
                if self.check_inbound(new_sub_node):
                    new_nodes.append(new_sub_node)
            self.graph.nodes[node]["nodes"] = new_nodes
        return self

    def AddBorder(self, node, border_color: Color):
        """
        add a border with thickness 1 and border_color around the given node
        """
        delta = [-1, 0, 1]
        border_pixels = []
        for sub_node in self.graph.nodes[node]["nodes"]:
            for x in delta:
                for y in delta:
                    border_pixel = (sub_node[0] + y, sub_node[1] + x)
                    if border_pixel not in border_pixels and not self.check_pixel_occupied(border_pixel) and \
                            self.height > sub_node[0] + y >= 0 and self.width > sub_node[1] + x >= 0:
                        border_pixels.append(border_pixel)
        color_map = {
            "O": 0,
            "B": 1,
            "R": 2,
            "G": 3,
            "Y": 4,
            "X": 5,
            "F": 6,
            "A": 7,
            "C": 8,
            "W": 9,
            "most": self.most_common_color,
            "least": self.least_common_color
        }
        border_color = color_map[border_color]
        new_node_id = self.generate_node_id(border_color)
        if self.is_multicolor:
            self.graph.add_node(
                (node[0], node[1], 0),
                nodes=list(border_pixels),
                color=[border_color for j in border_pixels],
                size=len(border_pixels),
            )
        else:
            self.graph.add_node(
                (node[0], node[1], 0),
                nodes=list(border_pixels),
                color=border_color,
                size=len(border_pixels),
            )
        return self

    def FillRectangle(self, node, color: Color, overlap: Overlap):
        """
        fill the rectangle containing the given node with the given color.
        if overlap is True, fill the rectangle even if it overlaps with other nodes.
        """

        if color == "same":
            color = self.graph.nodes[node]["color"]
        color_map = {
            "O": 0,
            "B": 1,
            "R": 2,
            "G": 3,
            "Y": 4,
            "X": 5,
            "F": 6,
            "A": 7,
            "C": 8,
            "W": 9,
            "most": self.most_common_color,
            "least": self.least_common_color
        }
        color = color_map[color]
        all_x = [sub_node[1] for sub_node in self.graph.nodes[node]["nodes"]]
        all_y = [sub_node[0] for sub_node in self.graph.nodes[node]["nodes"]]
        self.graph.add_node(
            (node[0], node[1], 0),
            nodes=list(self.graph.nodes[node]["nodes"]),
            color=self.graph.nodes[node]["color"],
            size=len(list(self.graph.nodes[node]["nodes"]))
        )
        if not all_x or not all_y:  # Check if either list is empty
            print("Cannot fill rectangle as node has no subnodes.")
            return  # Exit the function early if there are no subnodes
        min_x, min_y, max_x, max_y = min(all_x), min(
            all_y), max(all_x), max(all_y)
        unfilled_pixels = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                pixel = (y, x)
                if pixel not in self.graph.nodes[node]["nodes"]:
                    if overlap:
                        unfilled_pixels.append(pixel)
                    elif not self.check_pixel_occupied(pixel):
                        unfilled_pixels.append(pixel)
        if len(unfilled_pixels) > 0:
            new_node_id = self.generate_node_id(color)
            if self.is_multicolor:
                self.graph.add_node(
                    (node[0], node[1], 0),
                    nodes=list(unfilled_pixels),
                    color=[color for j in unfilled_pixels],
                    size=len(unfilled_pixels),
                )
            else:
                self.graph.add_node(
                    (node[0], node[1], 0),
                    nodes=list(unfilled_pixels),
                    color=color,
                    size=len(unfilled_pixels),
                )
        return self

    def HollowRectangle(self, node, color: Color):
        """
        hollowing the rectangle containing the given node with the given color.
        """
        if not self.graph.nodes[node]["nodes"]:  # Checks if the list of subnodes is empty
            print("Node has no subnodes")
            return
        all_y = [n[0] for n in self.graph.nodes[node]["nodes"]]
        all_x = [n[1] for n in self.graph.nodes[node]["nodes"]]
        border_y = [min(all_y), max(all_y)]
        border_x = [min(all_x), max(all_x)]
        non_border_pixels = []
        new_subnodes = []
        color_map = {
            "O": 0,
            "B": 1,
            "R": 2,
            "G": 3,
            "Y": 4,
            "X": 5,
            "F": 6,
            "A": 7,
            "C": 8,
            "W": 9,
            "most": self.most_common_color,
            "least": self.least_common_color
        }
        if self.is_multicolor:
            if not isinstance(color, int):
                color = [color_map[color]] * sum(
                    len(data["nodes"]) for node, data in self.graph.nodes(data=True)
                )
            else:
                color = [color] * sum(
                    len(data["nodes"]) for node, data in self.graph.nodes(data=True)
                )
        else:
            if not isinstance(color, int):
                color = color_map[color]

        for subnode in self.graph.nodes[node]["nodes"]:
            if subnode[0] in border_y or subnode[1] in border_x:
                new_subnodes.append(subnode)
            else:
                non_border_pixels.append(subnode)
        self.graph.nodes[node]["nodes"] = new_subnodes
        # Updated the size parameter here
        self.graph.nodes[node]["size"] = len(new_subnodes)
        if color != self.image.background_color:
            new_node_id = self.generate_node_id(color)
            self.graph.add_node(
                (node[0], node[1], 0),
                nodes=list(non_border_pixels),
                color=color,
                size=len(non_border_pixels),
            )
        return self

    def Mirror(self, node, mirror_axis):
        """
        mirroring a node with respect to the given axis.
        mirror_axis takes the form of (y, x) where one of y, x equals None to
        indicate the other being the axis of mirroring
        """
        if mirror_axis[1] is None and mirror_axis[0] is not None:
            axis = mirror_axis[0]
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = axis - (subnode[0] - axis)
                new_x = subnode[1]
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_axis[0] is None and mirror_axis[1] is not None:
            axis = mirror_axis[1]
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = subnode[0]
                new_x = axis - (subnode[1] - axis)
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        return self

    def Flip(self, node, mirror_direction: Symmetry_Axis):
        """
        flips the given node given direction horizontal, vertical, diagonal left/right
        """
        if not self.graph.nodes[node]["nodes"]:  # Checks if the list is empty
            print("No subnodes to flip")
            return self  # Exit the function as there's nothing to flip

        if mirror_direction == "VERTICAL" or mirror_direction == Symmetry_Axis.VERTICAL:  # todo: check sanity
            max_y = max([subnode[0]
                        for subnode in self.graph.nodes[node]["nodes"]])
            min_y = min([subnode[0]
                        for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = max_y - (subnode[0] - min_y)
                new_x = subnode[1]
                if self.check_inbound((new_x, new_y)):
                    new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_direction == "HORIZONTAL" or mirror_direction == Symmetry_Axis.HORIZONTAL:
            max_x = max([subnode[1]
                        for subnode in self.graph.nodes[node]["nodes"]])
            min_x = min([subnode[1]
                        for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = subnode[0]
                new_x = max_x - (subnode[1] - min_x)
                if self.check_inbound((new_x, new_y)):
                    new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_direction == "DIAGONAL_LEFT" or mirror_direction == Symmetry_Axis.DIAGONAL_LEFT:  # \
            min_x = min([subnode[1]
                        for subnode in self.graph.nodes[node]["nodes"]])
            min_y = min([subnode[0]
                        for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_subnode = (subnode[0] - min_y, subnode[1] - min_x)
                new_subnode = (new_subnode[1], new_subnode[0])
                new_subnode = (new_subnode[0] + min_y, new_subnode[1] + min_x)
                if self.check_inbound(new_subnode):
                    new_subnodes.append(new_subnode)
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_direction == "DIAGONAL_RIGHT" or mirror_direction == Symmetry_Axis.DIAGONAL_RIGHT:  # /
            max_x = max([subnode[1]
                        for subnode in self.graph.nodes[node]["nodes"]])
            min_y = min([subnode[0]
                        for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_subnode = (subnode[0] - min_y, subnode[1] - max_x)
                new_subnode = (-new_subnode[1], -new_subnode[0])
                new_subnode = (new_subnode[0] + min_y, new_subnode[1] + max_x)
                if self.check_inbound(new_subnode):
                    new_subnodes.append(new_subnode)
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        return self

    def Insert(
        self, node, object_id, point: ImagePoints, relative_pos: RelativePosition
    ):
        """
        insert some pattern identified by object_id at some location,
        the location is defined as, the relative position between the given node and point.
        for example, point=top, relative_pos=middle will insert the pattern between the given node
        and the top of the image.
        if object_id is -1, use the pattern given by node
        """
        node_centroid = self.get_centroid(node)
        if not isinstance(point, tuple):
            if point == "TOP" or point == ImagePoints.TOP:
                point = (0, node_centroid[1])
            elif point == "BOTTOM" or point == ImagePoints.BOTTOM:
                point = (self.image.height - 1, node_centroid[1])
            elif point == "LEFT" or point == ImagePoints.LEFT:
                point = (node_centroid[0], 0)
            elif point == "RIGHT" or point == ImagePoints.RIGHT:
                point = (node_centroid[0], self.image.width - 1)
            elif point == "TOP_LEFT" or point == ImagePoints.TOP_LEFT:
                point = (0, 0)
            elif point == "TOP_RIGHT" or point == ImagePoints.TOP_RIGHT:
                point = (0, self.image.width - 1)
            elif point == "BOTTOM_LEFT" or point == ImagePoints.BOTTOM_LEFT:
                point = (self.image.height - 1, 0)
            elif point == "BOTTOM_RIGHT" or point == ImagePoints.BOTTOM_RIGHT:
                point = (self.image.height - 1, self.image.width - 1)
        if object_id == -1:
            # special id for dynamic objects, which uses the given nodes as objects
            object = self.graph.nodes[node]
        else:
            object = self.image.task.static_objects_for_insertion[self.abstraction][
                object_id
            ]
        target_point = self.get_point_from_relative_pos(
            node_centroid, point, relative_pos
        )
        object_centroid = self.get_centroid_from_pixels(object["nodes"])
        subnodes_coords = []
        for subnode in object["nodes"]:
            delta_y = subnode[0] - object_centroid[0]
            delta_x = subnode[1] - object_centroid[1]
            if self.check_inbound((target_point[0] + delta_y, target_point[1] + delta_x)):
                subnodes_coords.append(
                    (target_point[0] + delta_y, target_point[1] + delta_x)
                )
        new_node_id = self.generate_node_id(object["color"])
        self.graph.add_node(  # todo: side-effects
            new_node_id,
            nodes=list(subnodes_coords),
            color=object["color"],
            size=len(list(subnodes_coords)),
        )
        return self

    def remove_node(self, node):
        """
        remove a node from the graph
        """
        self.graph.remove_node(node)

    # ------------------------------------- filters ------------------------------------------
    #  filters take the form of filter(node, params), return true if node satisfies filter
    def FilterByColor(self, node, color: Color):
        """
        return true if node has given color.
        if exclude, return true if node does not have given color.
        """
        color_map = {
            "most": self.most_common_color,
            "least": self.least_common_color,
            "O": 0,
            "B": 1,
            "R": 2,
            "G": 3,
            "Y": 4,
            "X": 5,
            "F": 6,
            "A": 7,
            "C": 8,
            "W": 9,
        }
        if self.is_multicolor:
            return color in self.graph.nodes[node]["color"]
        else:
            return self.graph.nodes[node]["color"] == color_map[color]

    def FilterByHeight(self, node, height: Height):
        if height == "MAX":
            height = self.get_attribute_max("height")
        elif height == "MIN":
            height = self.get_attribute_min("height")
        elif height == "ODD":
            return self.graph.nodes[node]["height"] % 2 != 0
        return self.graph.nodes[node]["height"] == height

    def FilterByWidth(self, node, width: Width):
        if width == "MAX":
            width = self.get_attribute_max("width")
        elif width == "MIN":
            width = self.get_attribute_min("width")
        elif width == "ODD":
            return self.graph.nodes[node]["width"] % 2 != 0
        return self.graph.nodes[node]["width"] == width

    def FilterBySize(self, node, size: Size):
        """
        return true if node has size equal to given size.
        if exclude, return true if node does not have size equal to given size.
        """
        if size == "MAX":
            size = self.get_attribute_max("size")
        elif size == "MIN":
            size = self.get_attribute_min("size")
        elif size == "ODD":
            return self.graph.nodes[node]["size"] % 2 != 0
        return self.graph.nodes[node]["size"] == size

    def FilterByDegree(self, node, degree: Degree):
        """
        return true if node has degree equal to given degree.
        if exclude, return true if node does not have degree equal to given degree.
        """
        return self.graph.degree[node] == degree

    def FilterByNeighborSize(self, node, size: Size):
        """
        return true if node has a neighbor of a given size.
        if exclude, return true if node does not have a neighbor of a given size.
        """
        if size == "MAX":
            size = self.get_attribute_max("size")
        elif size == "MIN":
            size = self.get_attribute_min("size")

        for neighbor in self.graph.neighbors(node):
            if size == "ODD":
                if self.graph.nodes[neighbor]["size"] % 2 != 0:
                    return True
            else:
                if self.graph.nodes[neighbor]["size"] == size:
                    return True
        return False

    def FilterByNeighborColor(self, node, color: Color):
        """
        return true if node has a neighbor of a given color.
        if exclude, return true if node does not have a neighbor of a given color.
        """
        color_map = {
            "most": self.most_common_color,
            "least": self.least_common_color,
            "O": 0,
            "B": 1,
            "R": 2,
            "G": 3,
            "Y": 4,
            "X": 5,
            "F": 6,
            "A": 7,
            "C": 8,
            "W": 9,
        }
        for neighbor in self.graph.neighbors(node):
            if self.graph.nodes[neighbor]["color"] == color:
                return True
        return False

    def FilterByNeighborDegree(self, node, degree: Degree):
        """
        return true if node has a neighbor of a given degree.
        if exclude, return true if node does not have a neighbor of a given degree.
        """
        for neighbor in self.graph.neighbors(node):
            if self.graph.degree[neighbor] == degree:
                return True
        return False

    def FilterByShape(self, node, shape: Shape):
        """
        return true if node contains a square-shaped hole
        """
        if shape == Shape.square or shape == "square":
            nodes = self.graph.nodes(data=True)[node]['nodes']
            print("nodes:", nodes)
            if not nodes:
                return False
            min_x = min(nodes, key=lambda x: x[0])[0]
            max_x = max(nodes, key=lambda x: x[0])[0]
            min_y = min(nodes, key=lambda x: x[1])[1]
            max_y = max(nodes, key=lambda x: x[1])[1]
            def is_square_formed_by_points(points):
                if not points:  # Check if the points list is empty
                    return False
                if len(points) == 1:
                    return True
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
            # Check if the dimensions form a square and if the number of points matches the area of the square
                if width == height and len(points) == width * height:
                    return True
                return False
            all_points = {(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)}
            missing_points = set(all_points) - set(nodes)
            edge_points = {(x, y) for x in [min_x, max_x] for y in range(min_y, max_y + 1)} | {(x, y) for y in [min_y, max_y] for x in range(min_x, max_x + 1)}
            if any(point in edge_points for point in missing_points):
                return False  # Missing points are at the edge
            return is_square_formed_by_points(list(missing_points))
        return False

    def FilterByColumns(self, node):
        """

        """
        pass
    # ------------------------------------- utils ------------------------------------------
    def get_attribute_max(self, attribute_name):
        """
        get the maximum value of the given attribute
        """
        if len(list(self.graph.nodes)) == 0:
            return None
        return max([data[attribute_name] for node, data in self.graph.nodes(data=True)])

    def get_attribute_min(self, attribute_name):
        """
        get the minimum value of the given attribute
        """
        if len(list(self.graph.nodes)) == 0:
            return None
        return min([data[attribute_name] for node, data in self.graph.nodes(data=True)])

    def get_color(self, node):
        """
        return the color of the node
        """
        if isinstance(node, list):
            return [self.graph.nodes[node_i]["color"] for node_i in node]
        else:
            return self.graph.nodes[node]["color"]

    def check_inbound(self, pixels):
        """
        check if given pixels are all within the image boundary
        """
        if not isinstance(pixels, list):
            pixels = [pixels]
        for pixel in pixels:
            y, x = pixel
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return False
        return True

    def check_collision(self, node_id, pixels_list=None):
        """
        check if given pixels_list collide with other nodes in the graph
        node_id is used to retrieve pixels_list if not given.
        node_id is also used so that only collision with other nodes are detected.
        """
        if pixels_list is None:
            pixels_set = set(self.graph.nodes[node_id]["nodes"])
        else:
            pixels_set = set(pixels_list)
        for node, data in self.graph.nodes(data=True):
            if len(set(data["nodes"]) & pixels_set) != 0 and node != node_id:
                return True
        return False

    def check_pixel_occupied(self, pixel):
        """
        check if a pixel is occupied by any node in the graph
        """
        for node, data in self.graph.nodes(data=True):
            if pixel in data["nodes"]:
                return True
        return False

    def get_shape(self, node):
        """
        given a node, get the shape of the node.
        the shape of the node is defined using its pixels shifted so that the top left is 0,0
        """
        sub_nodes = self.graph.nodes[node]["nodes"]
        if len(sub_nodes) == 0:
            return set()
        min_x = min([sub_node[1] for sub_node in sub_nodes])
        min_y = min([sub_node[0] for sub_node in sub_nodes])
        return set([(y - min_y, x - min_x) for y, x in sub_nodes])

    def get_centroid(self, node):
        """
        get the centroid of a node
        """
        center_y = (
            sum([n[0] for n in self.graph.nodes[node]["nodes"]])
            + self.graph.nodes[node]["size"] // 2
        ) // self.graph.nodes[node]["size"]
        center_x = (
            sum([n[1] for n in self.graph.nodes[node]["nodes"]])
            + self.graph.nodes[node]["size"] // 2
        ) // self.graph.nodes[node]["size"]
        return (center_y, center_x)

    def get_centroid_from_pixels(self, pixels):
        """
        get the centroid of a list of pixels
        """
        size = len(pixels)
        center_y = (sum([n[0] for n in pixels]) + size // 2) // size
        center_x = (sum([n[1] for n in pixels]) + size // 2) // size
        return (center_y, center_x)

    def get_relative_pos(self, node1, node2):
        # when both objects are single pixels
        if len(self.graph.nodes[node1]["nodes"]) == 1 and len(self.graph.nodes[node2]["nodes"]) == 1:
            for sub_node_1 in self.graph.nodes[node1]["nodes"]:
                for sub_node_2 in self.graph.nodes[node2]["nodes"]:
                    if sub_node_1[0] == sub_node_2[0]:
                        if sub_node_1[1] < sub_node_2[1]:
                            print("single-pixel: Dir.RIGHT")
                            return Dir.RIGHT
                        elif sub_node_1[1] > sub_node_2[1]:
                            print("single-pixel: Dir.LEFT")
                            return Dir.LEFT
                    elif sub_node_1[1] == sub_node_2[1]:
                        if sub_node_1[0] < sub_node_2[0]:
                            print("single-pixel: Dir.DOWN")
                            return Dir.DOWN
                        elif sub_node_1[0] > sub_node_2[0]:
                            print("single-pixel: Dir.UP")
                            return Dir.UP
                    elif sub_node_1[0] < sub_node_2[0]:  # DOWN
                        if sub_node_1[1] < sub_node_2[1]:
                            print("single-pixel: Dir.DOWN_RIGHT")
                            return Dir.DOWN_RIGHT
                        elif sub_node_1[1] > sub_node_2[1]:
                            print("single-pixel: Dir.DOWN_LEFT")
                            return Dir.DOWN_LEFT
                    elif sub_node_1[0] > sub_node_2[0]:  # UP
                        if sub_node_1[1] < sub_node_2[1]:
                            print("single-pixel: Dir.UP_RIGHT")
                            return Dir.UP_RIGHT
                        elif sub_node_1[1] > sub_node_2[1]:
                            print("single-pixel: Dir.UP_LEFT")
                            return Dir.UP_LEFT
        elif len(self.graph.nodes[node1]["nodes"]) == 1 or len(self.graph.nodes[node2]["nodes"]) == 1:
            # at least one of the objects is single-pixeled
            min_y_node1, max_y_node1 = min(node[0] for node in self.graph.nodes[node1]["nodes"]), max(
                node[0] for node in self.graph.nodes[node1]["nodes"])
            min_x_node1, max_x_node1 = min(node[1] for node in self.graph.nodes[node1]["nodes"]), max(
                node[1] for node in self.graph.nodes[node1]["nodes"])

            min_y_node2, max_y_node2 = min(node[0] for node in self.graph.nodes[node2]["nodes"]), max(
                node[0] for node in self.graph.nodes[node2]["nodes"])
            min_x_node2, max_x_node2 = min(node[1] for node in self.graph.nodes[node2]["nodes"]), max(
                node[1] for node in self.graph.nodes[node2]["nodes"])

            if min_x_node2 <= min_x_node1 and max_x_node1 <= max_x_node2:  # node-1 is within range
                if max_y_node1 < max_x_node2:
                    return Dir.DOWN
                elif max_y_node1 > max_x_node2:
                    return Dir.UP
            elif min_y_node2 <= min_y_node1 and max_y_node1 <= max_y_node2:  # node-1 is within range
                if max_x_node1 > max_x_node2:
                    return Dir.LEFT
                elif max_x_node1 < max_x_node2:
                    return Dir.RIGHT
            elif max_x_node2 < min_x_node1 and max_y_node1 < min_y_node2 and \
                    abs(max_x_node2 - min_x_node1) == abs(min_y_node2 - max_y_node1):
                return Dir.DOWN_LEFT
            elif max_x_node1 < min_x_node2 and max_y_node1 < min_y_node2 and \
                    abs(max_x_node1 - min_x_node2) == abs(max_y_node1 - min_y_node2):
                return Dir.DOWN_RIGHT
            elif max_x_node1 < min_x_node2 and min_y_node1 > max_y_node2 and \
                    abs(max_x_node1 - min_x_node2) == abs(min_y_node1 - max_y_node2):
                return Dir.UP_RIGHT
            elif max_x_node1 > min_x_node2 and max_y_node1 > min_y_node2 and \
                    abs(min_x_node2 - max_x_node1) == abs(min_y_node2 - max_y_node1):
                return Dir.UP_LEFT
        else:
            # node: the case where both objects are not single-pixeled -- diagonal not supported
            node1_x, node2_x = [node[0] for node in self.graph.nodes[node1]["nodes"]], [
                node[0] for node in self.graph.nodes[node2]["nodes"]]
            node1_y, node2_y = [node[1] for node in self.graph.nodes[node1]["nodes"]], [
                node[1] for node in self.graph.nodes[node2]["nodes"]]
            common_y_coords = set(node1_y).intersection(set(node2_y))
            common_x_coords = set(node1_x).intersection(set(node2_x))
            combined_nodes = list(
                self.graph.nodes[node1]["nodes"]) + list(self.graph.nodes[node2]["nodes"])
            common_y = [tup for tup in combined_nodes if tup[1]
                        in common_y_coords]  # UP or DOWN
            common_x = [tup for tup in combined_nodes if tup[0]
                        in common_x_coords]  # LEFT or RIGHT
            if common_y:
                max_y = max(common_y)
                if max_y in self.graph.nodes[node1]["nodes"]:
                    return Dir.UP
                else:
                    return Dir.DOWN
            if common_x:
                max_x = max(common_x)
                if max_x in self.graph.nodes[node1]["nodes"]:
                    return Dir.LEFT
                else:
                    return Dir.RIGHT
        return None

    def get_mirror_axis(self, node1, node2):
        """
        get the axis to mirror node1 with given node2
        """
        node2_centroid = self.get_centroid(node2)

        if (node1, node2) in self.graph.edges and self.graph.edges[node1, node2]["direction"] == "vertical":
            return (node2_centroid[0], None)
        else:
            return (None, node2_centroid[1])

    def get_point_from_relative_pos(
        self, filtered_point, relative_point, relative_pos: RelativePosition
    ):
        """
        get the point to insert new node given
        filtered_point: the centroid of the filtered node
        relative_point: the centroid of the target node, or static point such as (0,0)
        relative_pos: the relative position of the filtered_point to the relative_point
        """
        if relative_pos == "SOURCE" or relative_pos == RelativePosition.SOURCE:
            return filtered_point
        elif relative_pos == RelativePosition.TARGET or relative_pos == "TARGET":
            return relative_point
        elif relative_pos == RelativePosition.MIDDLE or relative_pos == "MIDDLE":
            y = (filtered_point[0] + relative_point[0]) // 2
            x = (filtered_point[1] + relative_point[1]) // 2
            return (y, x)

    # ------------------------------------------ apply functions -----------------------------------
    def apply_all(self, filter: FilterASTNode, transformation: TransformASTNode):
        """
        perform a full operation on the abstracted graph
        1. apply filters to get a list of nodes to transform
        2. apply param binding to the filtered nodes to retrieve parameters for the transformation
        3. apply transformation to the nodes
        """
        transformed_nodes = {}
        for node, info in self.graph.nodes(data=True):
            if self.apply_filters(node, filter):
                transformed_nodes[node] = [
                    child.value for child in transformation.children
                ]
        for node, params in transformed_nodes.items():
            self.apply_transform_inner(node, transformation, params)
        # update the edges in the abstracted graph to reflect the changes
        self.update_abstracted_graph(list(transformed_nodes.keys()))

    def var_apply_all(
        self, parameters: dict, filter: FilterASTNode, transformation: TransformASTNode
    ):
        transformed_nodes = {}
        for node in self.graph.nodes():
            if self.apply_filters(node, filter) and node in list(parameters.keys()):
                transformed_nodes[node] = [parameters[node]]
        for node, params in transformed_nodes.items():
            if params:
                self.apply_transform_inner(node, transformation, params)
        # update the edges in the abstracted graph to reflect the changes
        self.update_abstracted_graph(list(transformed_nodes.keys()))

    def apply_transform_inner(
        self, node, transformation: TransformASTNode, args: List[TransformASTNode]
    ):
        """
        apply transformation to a node
        """
        function_name = transformation.__class__.__name__
        try:
            getattr(self, function_name)(node, *args)  # apply transformation
        except AttributeError:
            function_name_var = function_name.replace("Var", "")
            getattr(self, function_name_var)(
                node, *args)  # apply var transformation

    def apply_filters(self, node, filter: FilterASTNode):
        """
        given filters and a node, return True if node satisfies all filters
        """
        filter_name = filter.__class__.__name__
        if filter is None:
            return self
        if filter_name == "Not":
            return not self.apply_filters(node, filter.children[0])
        elif filter_name == "Or":
            return any(self.apply_filters(node, child) for child in filter.children)
        elif filter_name == "And":
            return all(self.apply_filters(node, child) for child in filter.children)
        else:
            args = [child.value for child in filter.children]
            filter_method = getattr(self, filter_name, None)
            if filter_method:
                return getattr(self, filter_name)(node, *args)
            else:
                raise AttributeError(
                    f"Method for filter '{filter_name}' not found in ARCGraph'"
                )

    def update_abstracted_graph(self, affected_nodes):
        """
        update the abstracted graphs so that they remain consistent after transformation
        """
        pixel_assignments = {}
        for node, data in self.graph.nodes(data=True):
            for subnode in data["nodes"]:
                if subnode in pixel_assignments:
                    pixel_assignments[subnode].append(node)
                else:
                    pixel_assignments[subnode] = [node]
        for pixel, nodes in pixel_assignments.items():
            if len(nodes) > 1:
                for node_1, node_2 in combinations(nodes, 2):
                    if not self.graph.has_edge(node_1, node_2):
                        self.graph.add_edge(
                            node_1, node_2, direction="overlapping")

        for node1, node2 in combinations(self.graph.nodes, 2):
            if node1 == node2 or (
                self.graph.has_edge(node1, node2)
                and self.graph.edges[node1, node2]["direction"] == "overlapping"
            ):
                continue
            else:
                nodes_1 = self.graph.nodes[node1]["nodes"]
                nodes_2 = self.graph.nodes[node2]["nodes"]
                for n1 in nodes_1:
                    for n2 in nodes_2:
                        if n1[0] == n2[0]:  # two nodes on the same row
                            for column_index in range(
                                min(n1[1], n2[1]) + 1, max(n1[1], n2[1])
                            ):
                                # try:
                                pixel_assignment = pixel_assignments.get(
                                    (n1[0], column_index), []
                                )
                                if len(pixel_assignment) == 0 or (
                                    len(pixel_assignment) == 1
                                    and (
                                        pixel_assignment[0] == node1
                                        or pixel_assignment[0] == node2
                                    )
                                ):
                                    continue
                                break
                            else:
                                if self.graph.has_edge(node1, node2):
                                    self.graph.edges[node1, node2][
                                        "direction"
                                    ] = "horizontal"
                                else:
                                    self.graph.add_edge(
                                        node1, node2, direction="horizontal"
                                    )
                                break
                        elif n1[1] == n2[1]:  # two nodes on the same column:
                            for row_index in range(
                                min(n1[0], n2[0]) + 1, max(n1[0], n2[0])
                            ):
                                pixel_assignment = pixel_assignments.get(
                                    (row_index, n1[1]), []
                                )
                                if len(pixel_assignment) == 0 or (
                                    len(pixel_assignment) == 1
                                    and (
                                        pixel_assignment[0] == node1
                                        or pixel_assignment[0] == node2
                                    )
                                ):
                                    continue
                                break
                            else:
                                if self.graph.has_edge(node1, node2):
                                    self.graph.edges[node1, node2][
                                        "direction"
                                    ] = "vertical"
                                else:
                                    self.graph.add_edge(
                                        node1, node2, direction="vertical"
                                    )
                                break
                    else:
                        continue
                    break

    def generate_node_id(self, color):
        """
        find the next available id for a given color,
        ex: if color=1 and there are already (1,0) and (1,1), return (1,2)
        """
        if isinstance(color, list):  # multi-color cases
            color = color[0]
        max_id = 0
        for node in self.graph.nodes():
            if node[0] == color:
                max_id = max(max_id, node[1])
        return (color, max_id + 1)

    import itertools

    def undo_abstraction(self):
        """
        undo the abstraction to get the corresponding 2D grid
        return it as an ARCGraph object
        """

        width, height = self.image.image_size
        reconstructed_graph = nx.grid_2d_graph(height, width)
        nx.set_node_attributes(
            reconstructed_graph, self.image.background_color, "color"
        )
        if self.abstraction in self.image.multicolor_abstractions:
            for component, data in self.graph.nodes(data=True):
                for i, node in enumerate(data["nodes"]):
                    try:
                        reconstructed_graph.nodes[node]["color"] = data["color"][i]
                    except KeyError:  # ignore pixels outside of frame
                        pass
        else:
            for component, data in self.graph.nodes(data=True):
                for node in data["nodes"]:
                    try:
                        reconstructed_graph.nodes[node]["color"] = data["color"]
                    except KeyError:  # ignore pixels outside of frame
                        pass

        return ARCGraph(
            reconstructed_graph, self.name + "_reconstructed", self.image, None
        )

    def compute_grid(self) -> np.ndarray:
        if self.abstraction == None:
            reconstructed = self
        else:
            reconstructed = self.undo_abstraction()
        grid = np.zeros(
            (reconstructed.height, reconstructed.width), dtype=np.int32)
        for node, data in reconstructed.graph.nodes(data=True):
            x, y = node
            grid[x, y] = data["color"]

        return grid

    def plot(self, ax=None, save_fig=False, file_name=None):
        """
        visualize the graph
        """
        if ax is None:
            if self.abstraction is None:
                fig = plt.figure(figsize=(6, 6))
            else:
                fig = plt.figure(figsize=(4, 4))
        else:
            fig = ax.get_figure()

        if self.abstraction is None:
            pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}
            color = [
                self.colors[self.graph.nodes[x, y]["color"]]
                for x, y in self.graph.nodes()
            ]

            nx.draw(self.graph, ax=ax, pos=pos,
                    node_color=color, node_size=600)
            nx.draw_networkx_labels(
                self.graph, ax=ax, font_color="#676767", pos=pos, font_size=8
            )

        else:
            pos = {}
            for node in self.graph.nodes:
                centroid = self.get_centroid(node)
                pos[node] = (centroid[1], -centroid[0])

            if self.abstraction == "mcccg":
                color = [self.colors[0]
                         for node, data in self.graph.nodes(data=True)]
            else:
                color = [
                    self.colors[data["color"]]
                    for node, data in self.graph.nodes(data=True)
                ]
            size = [300 * data["size"]
                    for node, data in self.graph.nodes(data=True)]

            nx.draw(self.graph, pos=pos, node_color=color, node_size=size)
            nx.draw_networkx_labels(
                self.graph, font_color="#676767", pos=pos, font_size=8
            )

            edge_labels = nx.get_edge_attributes(self.graph, "direction")
            nx.draw_networkx_edge_labels(
                self.graph, pos=pos, edge_labels=edge_labels)

        if save_fig:
            if file_name is not None:
                fig.savefig(self.save_dir + "/" + file_name)
            else:
                fig.savefig(self.save_dir + "/" + self.name)
        plt.close()
