from enum import Enum
from typing import Union, List, Dict
import typing
from task import *
import unittest
from filter import *

class TestFilterGrammarRepresentation(unittest.TestCase):

    def test_color_enum(self):
        color_instance = Color.C0
        self.assertEqual(color_instance.nodeType, FilterTypes.COLOR)
        self.assertEqual(color_instance.code, "0")
        self.assertEqual(color_instance.size, 1)
        self.assertEqual(color_instance.children, [])

    def test_size_enum(self):
        size_instance = Size.MIN
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "min")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])
    
    def test_size_enum_dyn(self):
        size_instance = Size.S15
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "15")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])

    def test_degree_enum(self):
        degree_instance = Degree.D1
        self.assertEqual(degree_instance.nodeType, FilterTypes.DEGREE)
        self.assertEqual(degree_instance.code, "1")
        self.assertEqual(degree_instance.size, 1)
        self.assertEqual(degree_instance.children, [])

    def test_exclude_enum(self):
        exclude_instance = Exclude.TRUE
        self.assertEqual(exclude_instance.nodeType, FilterTypes.EXCLUDE)
        self.assertEqual(exclude_instance.code, "True")
        self.assertEqual(exclude_instance.size, 1)
        self.assertEqual(exclude_instance.children, [])

    def test_filter_by_color(self):
        filter_instance = FilterByColor(Color.C1, Exclude.TRUE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "FilterByColor(1, True)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_size(self):
        filter_instance = FilterBySize(Size.MIN, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "FilterBySize(min, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_size_enum(self):
        filter_instance = FilterBySize(Size.S25, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "FilterBySize(25, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_degree(self):
        filter_instance = FilterByDegree(Degree.MAX, Exclude.TRUE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "FilterByDegree(max, True)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_and_operator(self):
        filter1 = FilterByColor(Color.C1, Exclude.TRUE)
        filter2 = FilterByDegree(Degree.MAX, Exclude.TRUE)
        and_instance = And(Filters(filter1), Filters(filter2))
        self.assertEqual(and_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(and_instance.code, "And(FilterByColor(1, True), FilterByDegree(max, True))")
        self.assertEqual(and_instance.size, 7)
        self.assertEqual(len(and_instance.children), 2)

    def test_or_operator(self):
        filter1 = FilterByColor(Color.C1, Exclude.TRUE)
        filter2 = FilterBySize(Size.MIN, Exclude.FALSE)
        or_instance = Or(Filters(filter1), Filters(filter2))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(FilterByColor(1, True), FilterBySize(min, False))")
        self.assertEqual(or_instance.size, 7)
        self.assertEqual(len(or_instance.children), 2)

    def test_filter_by_neighbor_color(self):
        filter_instance = FilterByNeighborColor(Color.C3, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "FilterByNeighborColor(3, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_neighbor_size(self):
        filter_instance = FilterByNeighborSize(Size.ODD, Exclude.TRUE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "FilterByNeighborSize(odd, True)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filter_by_neighbor_degree(self):
        filter_instance = FilterByNeighborDegree(Degree.MIN, Exclude.FALSE)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTER_OPS)
        self.assertEqual(filter_instance.code, "FilterByNeighborDegree(min, False)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)

    def test_filters_representation(self):
        filter_instance = FilterByColor(Color.C5, Exclude.TRUE)
        filters_instance = Filters(filter_instance)
        self.assertEqual(filters_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(filters_instance.code, "FilterByColor(5, True)")
        self.assertEqual(filters_instance.size, 3)
        self.assertEqual(len(filters_instance.children), 1)

    def test_complex_and_operator(self):
        filter1 = FilterByNeighborSize(Size.ODD, Exclude.TRUE)
        filter2 = FilterByNeighborDegree(Degree.MIN, Exclude.FALSE)
        filter3 = FilterByColor(Color.C1, Exclude.TRUE)
        and_instance = And(Filters(filter1), And(Filters(filter2), Filters(filter3)))
        self.assertEqual(and_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(and_instance.code, "And(FilterByNeighborSize(odd, True), And(FilterByNeighborDegree(min, False), FilterByColor(1, True)))")
        self.assertEqual(and_instance.size, 11)
        self.assertEqual(len(and_instance.children), 2)

    def test_complex_or_operator(self):
        filter1 = FilterBySize(Size.MIN, Exclude.FALSE)
        filter2 = FilterByDegree(Degree.MAX, Exclude.TRUE)
        filter3 = FilterByNeighborColor(Color.C7, Exclude.FALSE)
        or_instance = Or(Filters(filter1), Or(Filters(filter2), Filters(filter3)))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(FilterBySize(min, False), Or(FilterByDegree(max, True), FilterByNeighborColor(7, False)))")
        self.assertEqual(or_instance.size, 11)
        self.assertEqual(len(or_instance.children), 2)

if __name__ == "__main__":
    taskNumber = "bb43febb"
    task = Task("dataset/" + taskNumber + ".json")
    task.abstraction = "nbccg"
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                               input in task.train_input]
    task.get_static_object_attributes(task.abstraction)
    setup_size_and_degree_based_on_task(task)
    unittest.main()