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
        color_instance = FColor.C0
        self.assertEqual(color_instance.nodeType, FilterTypes.COLOR)
        self.assertEqual(color_instance.code, "FColor.C0")
        self.assertEqual(color_instance.size, 1)
        self.assertEqual(color_instance.children, [])
        color_instance.execute(self.task, color_instance.children)
        self.assertEqual(color_instance.values, [])

    def test_size_enum(self):
        size_instance = Size.MIN
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "SIZE.MIN")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])
        size_instance.execute(self.task)
        self.assertEqual(size_instance.values, [])
    
    def test_size_enum_dyn(self):
        size_instance = Size.S15
        self.assertEqual(size_instance.nodeType, FilterTypes.SIZE)
        self.assertEqual(size_instance.code, "SIZE.S15")
        self.assertEqual(size_instance.size, 1)
        self.assertEqual(size_instance.children, [])
        size_instance.execute(self.task)
        self.assertEqual(size_instance.values, [])

    def test_degree_enum(self):
        degree_instance = Degree.D1
        self.assertEqual(degree_instance.nodeType, FilterTypes.DEGREE)
        self.assertEqual(degree_instance.code, "DEGREE.D1")
        self.assertEqual(degree_instance.size, 1)
        self.assertEqual(degree_instance.children, [])
        degree_instance.execute(self.task)
        self.assertEqual(degree_instance.values, [])

    def test_filter_by_color(self):
        filter_instance = FilterByColor(FColor.C0)
        new_instance = filter_instance.execute(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByColor(FColor.C0)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(new_instance.values, [[], []])
        self.assertEqual(len(new_instance.children), 1)

    def test_filter_by_size(self):
        filter_instance = FilterBySize(Size.MIN)
        new_instance = filter_instance.execute(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterBySize(SIZE.MIN)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(len(new_instance.children), 1)
        self.assertEqual(new_instance.values, [[], []])

    def test_filter_by_size_enum(self):
        filter_instance = FilterBySize(Size.S25)
        new_instance = filter_instance.execute(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterBySize(SIZE.S25)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(len(new_instance.children), 1)
        self.assertEqual(new_instance.values, [[(5, 0)], []])

    def test_filter_by_degree(self):
        filter_instance = FilterByDegree(Degree.MAX)
        new_instance = filter_instance.execute(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByDegree(DEGREE.MAX)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(len(new_instance.children), 1)

    def test_filter_by_neighbor_color(self):
        filter_instance = FilterByNeighborColor(FColor.C3)
        new_instance = filter_instance.execute(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByNeighborColor(FColor.C3)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(len(filter_instance.children), 1)
        self.assertEqual(new_instance.values, [[], []])

    def test_filter_by_neighbor_size(self):
        filter_instance = FilterByNeighborSize(Size.ODD)
        new_instance = filter_instance.execute(self.task, filter_instance.children)
        self.assertEqual(filter_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(filter_instance.code, "FilterByNeighborSize(SIZE.ODD)")
        self.assertEqual(filter_instance.size, 2)
        self.assertEqual(len(filter_instance.children), 1)
        self.assertEqual(new_instance.values, [[], []])

    def test_filter_by_neighbor_degree(self):
        filter_instance = FilterByNeighborDegree(Degree.MIN)
        new_instance = filter_instance.execute(self.task, filter_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByNeighborDegree(DEGREE.MIN)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(len(new_instance.children), 1)
        self.assertEqual(new_instance.values, [[], []])

    def test_filters_representation(self):
        filters_instance = FilterByColor(FColor.C5)
        new_instance = filters_instance.execute(self.task, filters_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "FilterByColor(FColor.C5)")
        self.assertEqual(new_instance.size, 2)
        self.assertEqual(len(new_instance.children), 1)
        self.assertEqual(new_instance.values, [[(5, 0), (5, 1)], [(5, 0), (5, 1)]])

    def test_and_operator(self):
        filter1 = FilterByColor(FColor.C1)
        filter2 = FilterByColor(FColor.C1)
        newf1 = filter1.execute(self.task, filter1.children)
        newf2 = filter2.execute(self.task, filter2.children)
        and_instance = And(newf1, newf2)
        new_instance = and_instance.execute(self.task, and_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "And(FilterByColor(FColor.C1), FilterByColor(FColor.C1))")
        self.assertEqual(new_instance.size, 7)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_not_operator(self):
        filter = FilterByColor(FColor.C1)
        newf = filter.execute(self.task, filter.children)
        not_instance = Not(newf)
        new_instance = not_instance.execute(self.task, not_instance.children)

        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "Not(FilterByColor(FColor.C1))")
        self.assertEqual(new_instance.size, 3)

    def test_and_operator(self):
        filter1 = FilterByColor(FColor.C1)
        filter2 = FilterByColor(FColor.C1)
        newf1 = filter1.execute(self.task, filter1.children)
        newf2 = filter2.execute(self.task, filter2.children)
        and_instance = Or(newf1, newf2)
        new_instance = and_instance.execute(self.task, and_instance.children)
        self.assertEqual(new_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(new_instance.code, "Or(FilterByColor(FColor.C1), FilterByColor(FColor.C1))")
        self.assertEqual(new_instance.size, 5)
        self.assertEqual(len(new_instance.children), 2)
        self.assertEqual(new_instance.values, [[], []])

    def test_or_operator(self):
        filter1 = FilterByColor(FColor.C1)
        filter2 = FilterBySize(Size.MIN)
        or_instance = Or(Filters(filter1), Filters(filter2))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(FilterByColor(FColor.C1), FilterBySize(SIZE.MIN))")
        self.assertEqual(or_instance.size, 5)
        self.assertEqual(len(or_instance.children), 2)

    def test_complex_and_operator(self):
        filter1 = FilterByNeighborSize(Size.ODD)
        filter2 = FilterByNeighborDegree(Degree.MIN)
        filter3 = FilterByColor(FColor.C1)
        and_instance = And(filter1, And(filter2, filter3))
        self.assertEqual(and_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(and_instance.code, "And(FilterByNeighborSize(SIZE.ODD), And(FilterByNeighborDegree(DEGREE.MIN), FilterByColor(FColor.C1)))")
        self.assertEqual(and_instance.size, 8)
        self.assertEqual(len(and_instance.children), 2)

    def test_complex_or_operator(self):
        filter1 = FilterBySize(Size.MIN)
        filter2 = FilterByDegree(Degree.MAX)
        filter3 = FilterByNeighborColor(FColor.C7)
        or_instance = Or(filter1, Or(filter2, filter3))
        self.assertEqual(or_instance.nodeType, FilterTypes.FILTERS)
        self.assertEqual(or_instance.code, "Or(FilterBySize(SIZE.MIN), Or(FilterByDegree(DEGREE.MAX), FilterByNeighborColor(FColor.C7)))")
        self.assertEqual(or_instance.size, 8)
        self.assertEqual(len(or_instance.children), 2)

if __name__ == "__main__":
    unittest.main()