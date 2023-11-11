from task import *
import unittest
from typing import Union, List, Dict, Iterator
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator
from childrenIterator import ChildrenIterator
from filters import *
from OEValuesManager import *
from filter_synthesis import *
import unittest
from transform_synthesis import *

class TestFSizeEnumerator(unittest.TestCase):
    def setUp(self):
        taskNumber = "bb43febb"
        task = Task("dataset/" + taskNumber + ".json")
        task.abstraction = "nbccg"
        task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                              input in task.train_input]
        task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                               output in task.train_output]
        task.get_static_inserted_objects()
        task.get_static_object_attributes(task.abstraction)
        setup_size_and_degree_based_on_task(task)
        vocabMakers = [Degree, FColor, FilterByColor, And]
        vocab = VocabFactory.create(vocabMakers)
        self.enumerator = FSizeEnumerator(task, vocab, ValuesManager())

    def test_next_program(self):
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("DEGREE.MIN", self.enumerator.next().code)
        self.assertEqual("DEGREE.MAX", self.enumerator.next().code)
        self.assertEqual("DEGREE.ODD", self.enumerator.next().code)
        self.assertEqual("DEGREE.D1", self.enumerator.next().code)
        self.assertEqual("FColor.C0", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("FColor.C1", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("FColor.C2", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual(1, self.enumerator.next().size)
        self.assertEqual("FColor.C4", self.enumerator.next().code)
        self.assertEqual("FColor.C5", self.enumerator.next().code)
        self.assertEqual("FColor.C6", self.enumerator.next().code)
        self.assertEqual("FColor.C7", self.enumerator.next().code)
        self.assertEqual("FColor.C8", self.enumerator.next().code)
        self.assertEqual("FColor.C9", self.enumerator.next().code)
        self.assertEqual("FColor.LEAST", self.enumerator.next().code)
        self.assertEqual("FColor.MOST", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("FilterByColor(FColor.C5)", self.enumerator.next().code)
        #self.assertEqual("FilterByColor(FColor.C0, Exclude.FALSE)", self.enumerator.next().code)
        #self.assertEqual("FilterByColor(FColor.C1, Exclude.TRUE)", self.enumerator.next().code)
        #self.assertEqual("FilterByColor(FColor.C1, Exclude.FALSE)", self.enumerator.next().code)
        #self.assertEqual("FilterByColor(FColor.C2, Exclude.TRUE)", self.enumerator.next().code)
        #self.assertEqual("FilterByColor(FColor.C2, Exclude.FALSE)", self.enumerator.next().code)
        #self.assertEqual("FilterByColor(FColor.C3, Exclude.TRUE)", self.enumerator.next().code)
        #for i in range(18):
            #self.enumerator.next()
        #self.assertEqual("And(FilterByColor(FColor.C0, Exclude.TRUE), FilterByColor(FColor.C0, Exclude.FALSE))", self.enumerator.next().code)
        #self.assertEqual("And(FilterByColor(FColor.C0, Exclude.TRUE), FilterByColor(FColor.C1, Exclude.TRUE))", self.enumerator.next().code)

import unittest
class TestTSizeEnumerator(unittest.TestCase):
    def setUp(self):
        taskNumber = "bb43febb"
        task = Task("dataset/" + taskNumber + ".json")
        task.abstraction = "nbccg"
        task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                              input in task.train_input]
        task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                               output in task.train_output]
        task.get_static_inserted_objects()
        task.get_static_object_attributes(task.abstraction)
        vocabMakers = [Color, UpdateColor, Transforms, NoOp]
        vocab = VocabFactory.create(vocabMakers)
        self.filter = Not(FilterByColor(FColor.C0))
        self.enumerator = TSizeEnumerator(task, vocab, self.filter, ValuesManager())

    def test_next_program(self):
        print(list(self.enumerator.vocab.leaves()))
        self.assertEqual(list(self.enumerator.vocab.leaves()), [Color.C0, Color.C1, Color.C2, Color.C3, Color.C4, Color.C5, Color.C6, Color.C7, Color.C8, Color.C9, Color.LEAST, Color.MOST, NoOp()])
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("Color.C0", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("Color.C1", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("Color.C2", self.enumerator.next().code)     
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual(1, self.enumerator.next().size)
        self.assertEqual("Color.C4", self.enumerator.next().code)
        self.assertEqual("Color.C5", self.enumerator.next().code)
        self.assertEqual("Color.C6", self.enumerator.next().code)
        self.assertEqual("Color.C7", self.enumerator.next().code)
        self.assertEqual("Color.C8", self.enumerator.next().code)
        self.assertEqual("Color.C9", self.enumerator.next().code)
        self.assertEqual("Color.LEAST", self.enumerator.next().code)
        self.assertEqual("Color.MOST", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("NoOp", self.enumerator.next().code)
        self.assertEqual("updateColor(Color.C0)", self.enumerator.next().code)
        self.assertEqual("updateColor(Color.C1)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C2)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C3)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C4)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C5)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C6)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C7)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C8)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.C9)", self.enumerator.next().code)
        self.assertFalse(self.enumerator.hasNext())  # OE gets rid of all other programs
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C0), updateColor(Color.C0)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C1), updateColor(Color.C0)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C1), updateColor(Color.C1)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C1), updateColor(Color.C2)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C0), updateColor(Color.C4)]")

if __name__ == "__main__":
    unittest.main()