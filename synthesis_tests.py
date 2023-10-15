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

class TestSizeEnumerator(unittest.TestCase):
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
        vocabMakers = [Degree, FColor, Exclude, FilterByColor, And]
        vocab = VocabFactory.create(vocabMakers)
        self.enumerator = FSizeEnumerator(task, vocab, FilterValuesManager())

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
        self.assertEqual("Exclude.TRUE", self.enumerator.next().code)
        self.assertEqual("Exclude.FALSE", self.enumerator.next().code)
        self.assertEqual("FilterByColor(FColor.C0, Exclude.TRUE)", self.enumerator.next().code)
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

if __name__ == "__main__":
    unittest.main()