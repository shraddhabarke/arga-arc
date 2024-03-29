from networkx.algorithms.components import connected_components
from ARCGraph import *

class Image:
    abstractions = ["na", "nbccg", "ccgbr", "ccgbr2", "ccg", "mcccg", "lrg", "nbvcg", "nbccgm", "sp"]
    abstraction_ops = {
        "nbccg": "get_non_black_components_graph",
        "ccgbr": "get_connected_components_graph_background_removed",
        "ccgbr2": "get_connected_components_graph_background_removed_2",
        "ccg": "get_connected_components_graph",
        "mcccg": "get_multicolor_connected_components_graph",
        "na": "get_no_abstraction_graph",
        "nbvcg": "get_non_background_vertical_connected_components_graph",
        "nbhcg": "get_non_background_horizontal_connected_components_graph",
        "lrg": "get_largest_rectangle_graph",
        "nbccgm": "get_non_black_components_graph_moore",
        "sp": "get_single_pixels_graph"
    }
    multicolor_abstractions = ["mcccg", "na"]

    def __init__(self, task, grid=None, width=None, height=None, graph=None, name="image"):
        """
        an image represents a 2D grid of pixels.
        the coordinate system follows the convention of 0,0 being the top left pixel of the image
        :param grid: a grid that represent the image
        :param width: if a grid is not given, determines the width of the graph
        :param height: if a grid is not given, determines the height of the graph
        :param graph: if a networkx graph is given, use it directly as the graph
        """
        self.task = task

        self.name = name
        self.colors_included = set()
        self.background_color = 0
        self.grid = grid
        self.most_common_color = 0
        self.least_common_color = 0

        if not grid and not graph:
            # create a graph with default color
            self.width = width
            self.height = height
            self.image_size = (width, height)
            self.graph = nx.grid_2d_graph(height, width)
            nx.set_node_attributes(self.graph, 0, "color")
            self.arc_graph = ARCGraph(self.graph, self.name, self)
            self.colors_included.add(0)
        elif graph:
            self.width = max([node[1] for node in graph.nodes()]) + 1
            self.height = max([node[0] for node in graph.nodes()]) + 1
            self.image_size = (width, height)
            self.graph = graph
            self.arc_graph = ARCGraph(self.graph, self.name, self)
            colors = []
            for node, data in graph.nodes(data=True):
                colors.append(data["color"])
            if 0 not in colors:
                self.background_color = max(set(colors), key=colors.count)  # simple way to retrieve most common item
            self.colors_included = set(colors)
            if len(colors) != 0:
                self.most_common_color = max(set(colors), key=colors.count)
                self.least_common_color = min(set(colors), key=colors.count)
        else:
            # create a graph with the color in given grid
            self.width = len(grid[0])
            self.height = len(grid)
            self.image_size = (self.width, self.height)
            self.graph = nx.grid_2d_graph(self.height, self.width)
            colors = []
            for r, row in enumerate(grid):
                for c, color in enumerate(row):
                    self.graph.nodes[r, c]["color"] = color
                    colors.append(color)
            self.arc_graph = ARCGraph(self.graph, self.name, self)
            if 0 not in colors:
                self.background_color = max(set(colors), key=colors.count)  # simple way to retrieve most common item
            self.colors_included = set(colors)
            if len(colors) != 0:
                self.most_common_color = max(set(colors), key=colors.count)
                self.least_common_color = min(set(colors), key=colors.count)
        self.corners = {(0, 0), (0, self.width - 1), (self.height - 1, 0), (self.height - 1, self.width - 1)}

    def copy(self):
        """
        return a copy of the image
        """
        return Image(self.task, grid=self.grid, name=self.name)

    #  --------------------------------------abstractions-----------------------------------
    def get_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as: 
        a group of adjacent pixels of the same color in the original graph
        """
        if not graph:
            graph = self.graph

        color_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        for color in range(10):
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            for i, component in enumerate(color_connected_components):
                height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                color_connected_components_graph.add_node((color, i), nodes=list(component), color=color,
                                                          size=len(list(component)), width=width+1, height=height+1)

        for node_1, node_2 in combinations(color_connected_components_graph.nodes, 2):
            nodes_1 = color_connected_components_graph.nodes[node_1]["nodes"]
            nodes_2 = color_connected_components_graph.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            color_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            color_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        return ARCGraph(color_connected_components_graph, self.name, self, "ccg")

    def get_connected_components_graph_background_removed(self, graph=None):
        """
        return an abstracted graph where a node is defined as: 
        a group of adjacent pixels of the same color in the original graph.
        remove nodes identified as background.
        background is defined as a node that includes a corner and has the most common color
        """
        if not graph:
            graph = self.graph
        ccgbr = nx.Graph()

        for color in range(10):
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            if color != self.background_color:
                for i, component in enumerate(color_connected_components):
                    height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                    width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                    ccgbr.add_node((color, i), nodes=list(component), color=color, size=len(list(component)), width=width+1, height=height+1)
            else:
                for i, component in enumerate(color_connected_components):
                    height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                    width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                    if len(set(component) & self.corners) == 0:  # background color + contains a corner
                        ccgbr.add_node((color, i), nodes=list(component), color=color, size=len(list(component)), width=width+1, height=height+1)

        for node_1, node_2 in combinations(ccgbr.nodes, 2):
            nodes_1 = ccgbr.nodes[node_1]["nodes"]
            nodes_2 = ccgbr.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            ccgbr.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            ccgbr.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        return ARCGraph(ccgbr, self.name, self, "ccgbr")

    def get_connected_components_graph_background_removed_2(self, graph=None):
        """
        return an abstracted graph where a node is defined as: 
        a group of adjacent pixels of the same color in the original graph.
        remove nodes identified as background.
        background is defined as a node that includes a corner or an edge node and has the most common color
        """
        if not graph:
            graph = self.graph

        ccgbr2 = nx.Graph()

        for color in range(10):
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)

            for i, component in enumerate(color_connected_components):
                if color != self.background_color:
                    height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                    width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                    ccgbr2.add_node((color, i), nodes=list(component), color=color, size=len(list(component)), width=width+1, height=height+1)
                else:
                    component = list(component)
                    height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                    width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                    for node in component:
                        # if the node touches any edge of image it is not included
                        if node[0] == 0 or node[0] == self.height - 1 or node[1] == 0 or node[1] == self.width - 1:
                            break
                    else:
                        ccgbr2.add_node((color, i), nodes=component, color=color, size=len(component), width=width+1, height=height+1)

        for node_1, node_2 in combinations(ccgbr2.nodes, 2):
            nodes_1 = ccgbr2.nodes[node_1]["nodes"]
            nodes_2 = ccgbr2.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            ccgbr2.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            ccgbr2.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        return ARCGraph(ccgbr2, self.name, self, "ccgbr2")

    def get_non_background_vertical_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as: 
        a group of vertically adjacent pixels of the same color in the original graph, excluding background color.
        """
        if not graph:
            graph = self.graph

        non_background_vertical_connected_components_graph = nx.Graph()

        for color in range(10):
            color_connected_components = []
            if color == self.background_color:
                continue
            for column in range(self.width):
                color_nodes = (node for node, data in graph.nodes(data=True) if
                               node[1] == column and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                non_background_vertical_connected_components_graph.add_node((color, i), nodes=list(component),
                                                                            color=color, size=len(list(component)), width=width+1, height=height+1)

        for node_1, node_2 in combinations(non_background_vertical_connected_components_graph.nodes, 2):
            nodes_1 = non_background_vertical_connected_components_graph.nodes[node_1]["nodes"]
            nodes_2 = non_background_vertical_connected_components_graph.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2,
                                                                                        direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2,
                                                                                        direction="vertical")
                            break
                else:
                    continue
                break

        return ARCGraph(non_background_vertical_connected_components_graph, self.name, self, "nbvcg")

    def get_non_background_horizontal_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as: 
        a group of horizontally adjacent pixels of the same color in the original graph, excluding background color.
        """
        if not graph:
            graph = self.graph

        non_background_horizontal_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        for color in range(10):
            color_connected_components = []
            if color == self.background_color:
                continue
            for row in range(self.height):
                color_nodes = (node for node, data in graph.nodes(data=True) if
                               node[0] == row and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                non_background_horizontal_connected_components_graph.add_node((color, i), nodes=list(component),
                                                                              color=color, size=len(list(component)), width=width+1, height=height+1)

        for node_1, node_2 in combinations(non_background_horizontal_connected_components_graph.nodes, 2):
            nodes_1 = non_background_horizontal_connected_components_graph.nodes[node_1]["nodes"]
            nodes_2 = non_background_horizontal_connected_components_graph.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2,
                                                                                          direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2,
                                                                                          direction="vertical")
                            break
                else:
                    continue
                break
        return ARCGraph(non_background_horizontal_connected_components_graph, self.name, self, "nbhcg")

    def get_non_black_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as: 
        a group of adjacent pixels of the same color in the original graph, excluding background color.
        """
        if not graph:
            graph = self.graph

        non_black_components_graph = nx.Graph()

        # for color in self.colors_included:
        for color in range(10):
            if color == 0:
                continue
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            for i, component in enumerate(color_connected_components):
                height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                non_black_components_graph.add_node((color, i), nodes=list(component), color=color,
                                                    size=len(list(component)), height=height+1, width=width+1)

        for node_1, node_2 in combinations(non_black_components_graph.nodes, 2):
            nodes_1 = non_black_components_graph.nodes[node_1]["nodes"]
            nodes_2 = non_black_components_graph.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != 0:
                                break
                        else:
                            non_black_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != 0:
                                break
                        else:
                            non_black_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        return ARCGraph(non_black_components_graph, self.name, self, "nbccg")

    def get_largest_rectangle_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        a group of adjacent pixels of the same color in the original graph that makes up a rectangle, excluding black.
        rectangles are identified from largest to smallest.
        """
        if not graph:
            graph = self.graph

        # https://www.drdobbs.com/database/the-maximal-rectangle-problem/184410529?pgno=1
        def area(llx, lly, urx, ury):
            if llx > urx or lly > ury or [llx, lly, urx, ury] == [0, 0, 0, 0]:
                return 0
            else:
                return (urx - llx + 1) * (ury - lly + 1)

        def all_nb(llx, lly, urx, ury, g):
            for x in range(llx, urx + 1):
                for y in range(lly, ury + 1):
                    if (y, x) not in g:
                        return False
            return True

        lrg = nx.Graph()
        for color in range(10):
            if color == 0:
                continue
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            subgraph_nodes = set(color_subgraph.nodes())
            i = 0
            while len(subgraph_nodes) != 0:
                best = [0, 0, 0, 0]
                for llx in range(self.width):
                    for lly in range(self.height):
                        for urx in range(self.width):
                            for ury in range(self.height):
                                cords = [llx, lly, urx, ury]
                                if area(*cords) > area(*best) and all_nb(*cords, subgraph_nodes):
                                    best = cords
                component = []
                for x in range(best[0], best[2] + 1):
                    for y in range(best[1], best[3] + 1):
                        component.append((y, x))
                        subgraph_nodes.remove((y, x))
                height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
                width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
                lrg.add_node((color, i), nodes=component, color=color, size=len(component), width=width+1, height=height+1)
                i += 1

        for node_1, node_2 in combinations(lrg.nodes, 2):
            nodes_1 = lrg.nodes[node_1]["nodes"]
            nodes_2 = lrg.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != 0:
                                break
                        else:
                            lrg.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != 0:
                                break
                        else:
                            lrg.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        return ARCGraph(lrg, self.name, self, "lrg")

    def get_multicolor_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        a group of adjacent pixels of any non-background color in the original graph.
        """
        if not graph:
            graph = self.graph
        multicolor_connected_components_graph = nx.Graph()

        non_background_nodes = [node for node, data in graph.nodes(data=True) if data["color"] != self.background_color]
        color_subgraph = graph.subgraph(non_background_nodes)
        multicolor_connected_components = connected_components(color_subgraph)

        for i, component in enumerate(multicolor_connected_components):
            sub_nodes = []
            sub_nodes_color = []
            for node in component:
                sub_nodes.append(node)
                sub_nodes_color.append(graph.nodes[node]["color"])
            height = max([pixel[0] for pixel in component]) - min([pixel[0] for pixel in component])
            width = max([pixel[1] for pixel in component]) - min([pixel[1] for pixel in component])
            multicolor_connected_components_graph.add_node((len(sub_nodes), i), nodes=sub_nodes, color=sub_nodes_color,
                                                           size=len(sub_nodes), width=width+1, height=height+1)
        # add edges between the abstracted nodes
        for node_1, node_2 in combinations(multicolor_connected_components_graph.nodes, 2):
            nodes_1 = multicolor_connected_components_graph.nodes[node_1]["nodes"]
            nodes_2 = multicolor_connected_components_graph.nodes[node_2]["nodes"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            multicolor_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            multicolor_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        return ARCGraph(multicolor_connected_components_graph, self.name, self, "mcccg")

    def get_no_abstraction_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        the entire graph as one multi-color node.
        """
        if not graph:
            graph = self.graph

        no_abs_graph = nx.Graph()
        sub_nodes = []
        sub_nodes_color = []
        for node, data in graph.nodes(data=True):
            sub_nodes.append(node)
            sub_nodes_color.append(graph.nodes[node]["color"])
        height = max([pixel[0] for pixel in sub_nodes]) - min([pixel[0] for pixel in sub_nodes])
        width = max([pixel[1] for pixel in sub_nodes]) - min([pixel[1] for pixel in sub_nodes])
        no_abs_graph.add_node((0, 0), nodes=sub_nodes, color=sub_nodes_color, size=len(sub_nodes), width=width+1, height=height+1)

        return ARCGraph(no_abs_graph, self.name, self, "na")

    def make_moore_neighborhood_graph(self, graph=None):
        """
        turns a graph in the nx.grid_2d_graph format into a moore neighborhood graph
        by adding diagonal edges between nodes
        """
        if not graph:
            graph = self.graph

        moore_neighborhood_graph = graph.copy()
        for node in graph.nodes:
            r, c = node
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                if (r + dr, c + dc) in graph.nodes:
                    moore_neighborhood_graph.add_edge(node, (r + dr, c + dc))
        return moore_neighborhood_graph

    def get_non_black_components_graph_moore(self, graph=None):
        if not graph:
            graph = self.graph
        
        graph = self.make_moore_neighborhood_graph(graph)

        cc_graph = self.get_non_black_components_graph(graph).graph

        return ARCGraph(cc_graph, self.name, self, "nbccgm")
    
    # def get_all_same_color_graph(self, graph=None):
    #     """
    #     return an abstracted graph where a node is defined as all the pixels of the same color
    #     """
    #     if not graph:
    #         graph = self.graph
    #     all_same_color_graph = nx.Graph()

    #     # in this case we do not remove the background color
    #     for color in range(10):
    #         color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
    #         # all the nodes of the same color become a single node
    #         all_same_color_graph.add_node(color, nodes=list(color_nodes), color=color, size=len(list(color_nodes)))
        

    def get_single_pixels_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as a single pixel of non-background color
        """
        if not graph:
            graph = self.graph
        single_pixels_graph = nx.Graph()

        for node, data in graph.nodes(data=True):
            if data["color"] != self.background_color:
                single_pixels_graph.add_node(
                    node, 
                    nodes=[node], 
                    color=data["color"], 
                    size=1, 
                    height=1, 
                    width=1
                )
        
        single_pixels_graph = self.make_visibility_graph(graph, single_pixels_graph)

        return ARCGraph(single_pixels_graph, self.name, self, "sp")

    def make_visibility_graph(self, original_grid, abstracted_graph):
        """
        given a graph of objects in the grid, adds edges between nodes that are visible to each other
        the original nodes are removed
        """
        # create a new graph with the same nodes, but no edges
        visibility_graph = nx.Graph()
        for node, data in abstracted_graph.nodes(data=True):
            visibility_graph.add_node(node, **data)

        def visible_pixels(pixel_1, pixel_2):
            # check for horizontal visibility
            if pixel_1[0] == pixel_2[0]:
                for column_index in range(min(pixel_1[1], pixel_2[1]) + 1, max(pixel_1[1], pixel_2[1])):
                    if original_grid.nodes[pixel_1[0], column_index]["color"] != self.background_color:
                        break
                else:
                    return "horizontal"
            # check for vertical visibility
            elif pixel_1[1] == pixel_2[1]:
                for row_index in range(min(pixel_1[0], pixel_2[0]) + 1, max(pixel_1[0], pixel_2[0])):
                    if original_grid.nodes[row_index, pixel_1[1]]["color"] != self.background_color:
                        break
                else:
                    return "vertical"
            return None

        def visible_nodes(node_1, node_2):
            # we need to iterate over all pairs of pixels in both of the nodes, and check if there is a line of sight
            # between any of them
            for pixel_1 in abstracted_graph.nodes[node_1]["nodes"]:
                for pixel_2 in abstracted_graph.nodes[node_2]["nodes"]:
                    visibility_direction = visible_pixels(pixel_1, pixel_2)
                    if visibility_direction:
                        return visibility_direction

        # add edges between nodes that are visible to each other
        for node_1, node_2 in combinations(visibility_graph.nodes, 2):
            visibility_direction = visible_nodes(node_1, node_2)
            if visibility_direction:
                visibility_graph.add_edge(node_1, node_2, direction=visibility_direction)

        return visibility_graph

                    

    # undo abstraction
    def undo_abstraction(self, arc_graph):
        return arc_graph.undo_abstraction()