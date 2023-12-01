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
        vocabMakers = [Degree, Size, FColor, FilterByColor, And]
        vocab = VocabFactory.create(vocabMakers)
        self.enumerator = FSizeEnumerator(task, vocab, ValuesManager())

    def test_next_program(self):
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("DEGREE.MIN", self.enumerator.next().code)
        self.assertEqual("DEGREE.MAX", self.enumerator.next().code)
        self.assertEqual("DEGREE.ODD", self.enumerator.next().code)
        self.assertEqual("DEGREE.1", self.enumerator.next().code)
        self.assertEqual("SIZE.MIN", self.enumerator.next().code)
        self.assertEqual("SIZE.MAX", self.enumerator.next().code)
        self.assertEqual("SIZE.ODD", self.enumerator.next().code)
        self.assertEqual("SIZE.25", self.enumerator.next().code)
        self.assertEqual("SIZE.12", self.enumerator.next().code)
        self.assertEqual("SIZE.30", self.enumerator.next().code)
        self.assertEqual("SIZE.15", self.enumerator.next().code)
        self.assertEqual("FColor.black", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("FColor.blue", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("FColor.red", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual(1, self.enumerator.next().size)
        self.assertEqual("FColor.yellow", self.enumerator.next().code)
        self.assertEqual("FColor.grey", self.enumerator.next().code)
        self.assertEqual("FColor.fuchsia", self.enumerator.next().code)
        self.assertEqual("FColor.orange", self.enumerator.next().code)
        self.assertEqual("FColor.cyan", self.enumerator.next().code)
        self.assertEqual("FColor.brown", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("FilterByColor(FColor.grey)", self.enumerator.next().code)
        self.assertFalse(self.enumerator.hasNext())
        #self.assertEqual("FilterByColor(FColor.C0)", self.enumerator.next().code)
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
        tleaf_makers = [Color.black, Color.blue, Color.red, Color.green, Color.yellow, Color.grey, Color.fuchsia, Color.orange, Color.cyan, Color.brown, NoOp()]
        t_vocabMakers = [UpdateColor, Transforms]
        transform_vocab = VocabFactory(tleaf_makers, t_vocabMakers)

        #UpdateColor, MoveNode, ExtendNode, MoveNodeMax, RotateNode, AddBorder, FillRectangle, HollowRectangle, Flip
        self.filter = Not(FilterByColor(FColor.black))
        self.enumerator = TSizeEnumerator(task, transform_vocab, ValuesManager())

    def test_next_program(self):
        #self.assertEqual(list(self.enumerator.vocab.leaves()), [Color.black, Color.blue, Color.red, Color.green, Color.yellow, Color.grey, Color.fuchsia, Color.orange, Color.cyan, Color.brown])
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("Color.black", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("Color.blue", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("Color.red", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual(1, self.enumerator.next().size)
        self.assertEqual("Color.yellow", self.enumerator.next().code)
        self.assertEqual("Color.grey", self.enumerator.next().code)
        self.assertEqual("Color.fuchsia", self.enumerator.next().code)
        self.assertEqual("Color.orange", self.enumerator.next().code)
        self.assertEqual("Color.cyan", self.enumerator.next().code)
        self.assertEqual("Color.brown", self.enumerator.next().code)
        self.assertEqual("NoOp", self.enumerator.next().code)
        self.assertEqual("updateColor(Color.black)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.blue)", self.enumerator.next().code)
        self.assertEqual("updateColor(Color.red)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.green)", self.enumerator.next().code)
        self.assertEqual("updateColor(Color.yellow)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.grey)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.fuchsia)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.orange)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.cyan)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.brown)", self.enumerator.next().code)
        self.assertFalse(self.enumerator.hasNext())
        self.assertFalse(self.enumerator.hasNext())  # OE gets rid of all other programs
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C0), updateColor(Color.C0)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C1), updateColor(Color.C0)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C1), updateColor(Color.C1)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C1), updateColor(Color.C2)]")
        #self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C0), updateColor(Color.C4)]")

if __name__ == "__main__":
    unittest.main()