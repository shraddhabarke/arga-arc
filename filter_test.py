from enum import Enum
from typing import Union, List, Dict
import typing
from task import *
import unittest
from filters import *

class TestFilterGrammarRepresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taskNumber = "bb43febb"
        cls.task = Task("dataset/" + cls.taskNumber + ".json")
        cls.task.abstraction = "nbccg"
        cls.task.input_abstracted_graphs_original[cls.task.abstraction] = [
            getattr(input, Image.abstraction_ops[cls.task.abstraction])() for
            input in cls.task.train_input
        ]
        cls.task.get_static_object_attributes(cls.task.abstraction)
        setup_size_and_degree_based_on_task(cls.task)

    def test_color_enum(self):
        color_instance = Color.C0
        self.assertEqual(color_instance.nodeType, FilterTypes.COLOR)
        self.assertEqual(color_instance.code, "Color.C0")
        self.assertEqual(color_instance.size, 1)
        self.assertEqual(color_instance.children, [])
        color_instance.apply(self.task)
        self.assertEqual(color_instance.values, [])

    def test_size_enum(self):
        size_instance = Size.MIN
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "SIZE.MIN")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])
        size_instance.apply(self.task)
        self.assertEqual(size_instance.values, [])
    
    def test_size_enum_dyn(self):
        size_instance = Size.S15
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "SIZE.S15")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])
        size_instance.apply(self.task)
        self.assertEqual(size_instance.values, [])

    def test_degree_enum(self):
        degree_instance = Degree.D1
        self.assertEqual(degree_instance.nodeType, FilterTypes.DEGREE)
        self.assertEqual(degree_instance.code, "DEGREE.D1")
        self.assertEqual(degree_instance.size, 1)
        self.assertEqual(degree_instance.children, [])
        degree_instance.apply(self.task)
        self.assertEqual(degree_instance.values, [])

    def test_exclude_enum(self):
        exclude_instance = Exclude.TRUE
        self.assertEqual(exclude_instance.nodeType, FilterTypes.EXCLUDE)
        self.assertEqual(exclude_instance.code, "Exclude.TRUE")
        self.assertEqual(exclude_instance.size, 1)
        self.assertEqual(exclude_instance.children, [])
        exclude_instance.apply(self.task)
        self.assertEqual(exclude_instance.values, [])

    def test_filter_by_color(self):
        filter_instance = FilterByColor(Color.C0, Exclude.TRUE)
        new_instance = filter_instance.apply(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByColor(Color.C0, Exclude.TRUE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(new_instance.values, [[(5, 0), (5, 1)], [(5, 0), (5, 1)]])
        self.assertEqual(len(new_instance.children), 2)

    def test_filter_by_size(self):
        filter_instance = FilterBySize(Size.MIN, Exclude.FALSE)
        new_instance = filter_instance.apply(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterBySize(SIZE.MIN, Exclude.FALSE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_filter_by_size_enum(self):
        filter_instance = FilterBySize(Size.S25, Exclude.FALSE)
        new_instance = filter_instance.apply(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterBySize(SIZE.S25, Exclude.FALSE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_filter_by_degree(self):
        filter_instance = FilterByDegree(Degree.MAX, Exclude.TRUE)
        new_instance = filter_instance.apply(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByDegree(DEGREE.MAX, Exclude.TRUE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(len(new_instance.children), 2)

    def test_filter_by_neighbor_color(self):
        filter_instance = FilterByNeighborColor(Color.C3, Exclude.FALSE)
        new_instance = filter_instance.apply(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByNeighborColor(Color.C3, Exclude.FALSE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_filter_by_neighbor_size(self):
        filter_instance = FilterByNeighborSize(Size.ODD, Exclude.TRUE)
        new_instance = filter_instance.apply(self.task, filter_instance.children)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(filter_instance.code, "FilterByNeighborSize(SIZE.ODD, Exclude.TRUE)")
        self.assertEqual(filter_instance.size, 3)
        self.assertEqual(len(filter_instance.children), 2)
        self.assertEqual(new_instance.values, [[(5, 0), (5, 1)], [(5, 0), (5, 1)]])

    def test_filter_by_neighbor_degree(self):
        filter_instance = FilterByNeighborDegree(Degree.MIN, Exclude.FALSE)
        new_instance = filter_instance.apply(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByNeighborDegree(DEGREE.MIN, Exclude.FALSE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_filters_representation(self):
        filters_instance = FilterByColor(Color.C5, Exclude.TRUE)
        new_instance = filters_instance.apply(self.task, filters_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByColor(Color.C5, Exclude.TRUE)")
        self.assertEqual(new_instance.size, 3)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_and_operator(self):
        filter1 = FilterByColor(Color.C1, Exclude.TRUE)
        filter2 = FilterByColor(Color.C1, Exclude.FALSE)
        newf1 = filter1.apply(self.task, filter1.children)
        newf2 = filter2.apply(self.task, filter2.children)
        and_instance = And(newf1, newf2)
        new_instance = and_instance.apply(self.task, and_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "And(FilterByColor(Color.C1, Exclude.TRUE), FilterByColor(Color.C1, Exclude.FALSE))")
        self.assertEqual(new_instance.size, 7)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_and_operator(self):
        filter1 = FilterByColor(Color.C1, Exclude.TRUE)
        filter2 = FilterByColor(Color.C1, Exclude.FALSE)
        newf1 = filter1.apply(self.task, filter1.children)
        newf2 = filter2.apply(self.task, filter2.children)
        and_instance = Or(newf1, newf2)
        new_instance = and_instance.apply(self.task, and_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "Or(FilterByColor(Color.C1, Exclude.TRUE), FilterByColor(Color.C1, Exclude.FALSE))")
        self.assertEqual(new_instance.size, 7)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[(5, 0), (5, 1)], [(5, 0), (5, 1)]])

    def test_or_operator(self):
        filter1 = FilterByColor(Color.C1, Exclude.TRUE)
        filter2 = FilterBySize(Size.MIN, Exclude.FALSE)
        or_instance = Or(Filters(filter1), Filters(filter2))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(FilterByColor(Color.C1, Exclude.TRUE), FilterBySize(SIZE.MIN, Exclude.FALSE))")
        self.assertEqual(or_instance.size, 7)
        self.assertEqual(len(or_instance.children), 2)

    def test_complex_and_operator(self):
        filter1 = FilterByNeighborSize(Size.ODD, Exclude.TRUE)
        filter2 = FilterByNeighborDegree(Degree.MIN, Exclude.FALSE)
        filter3 = FilterByColor(Color.C1, Exclude.TRUE)
        and_instance = And(filter1, And(filter2, filter3))
        self.assertEqual(and_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(and_instance.code, "And(FilterByNeighborSize(SIZE.ODD, Exclude.TRUE), And(FilterByNeighborDegree(DEGREE.MIN, Exclude.FALSE), FilterByColor(Color.C1, Exclude.TRUE)))")
        self.assertEqual(and_instance.size, 11)
        self.assertEqual(len(and_instance.children), 2)

    def test_complex_or_operator(self):
        filter1 = FilterBySize(Size.MIN, Exclude.FALSE)
        filter2 = FilterByDegree(Degree.MAX, Exclude.TRUE)
        filter3 = FilterByNeighborColor(Color.C7, Exclude.FALSE)
        or_instance = Or(filter1, Or(filter2, filter3))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(FilterBySize(SIZE.MIN, Exclude.FALSE), Or(FilterByDegree(DEGREE.MAX, Exclude.TRUE), FilterByNeighborColor(Color.C7, Exclude.FALSE)))")
        self.assertEqual(or_instance.size, 11)
        self.assertEqual(len(or_instance.children), 2)

if __name__ == "__main__":
    unittest.main()