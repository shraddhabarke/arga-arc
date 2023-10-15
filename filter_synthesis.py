# taskName: str, vocab: VocabFactory, oeManager: OEValuesManager, contexts: List[Map[String, Any]]):
from task import *
import unittest
from typing import Union, List, Dict, Iterator
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator
from childrenIterator import ChildrenIterator
from filters import *
from OEValuesManager import *

class FSizeEnumerator:
    def __init__(self, task: Task, vocab: VocabFactory, oeManager: FilterValuesManager):
        self.vocab = vocab
        self.oeManager = oeManager
        self.nextProgram = None
        self.bank: Dict[int, List[FilterASTNode]] = {}
        self.costLevel = 1
        self.currLevelProgs: List[FilterASTNode] = []
        self.currIter = LookaheadIterator(iter(vocab.leaves()))
        self.rootMaker = self.currIter.next()
        self.childrenIterator = LookaheadIterator(iter([None]))
        self.task = task

    def hasNext(self) -> bool:
        if self.nextProgram:
            return True
        else:
            self.nextProgram = self.getNextProgram()
            return self.nextProgram is not None

    def next(self) -> FilterASTNode:
        if not self.nextProgram:
            self.nextProgram = self.getNextProgram()
        res = self.nextProgram
        self.nextProgram = None
        return res

    def advanceRoot(self) -> bool:
        if not self.currIter.hasNext():
            return False
        self.rootMaker = self.currIter.next()
        if self.rootMaker.arity == 0 and self.rootMaker.size == self.costLevel:
            self.childrenIterator = LookaheadIterator(iter([None]))
        elif 1 < self.costLevel: # TODO: update this later for the cost-based enumeration
            childrenCost = self.costLevel - 1
            self.childrenIterator = ChildrenIterator(self.rootMaker.childTypes, childrenCost, self.bank)
        else:
            self.childrenIterator = LookaheadIterator(iter([]))
        return True

    def changeLevel(self) -> bool:
        self.costLevel += 1
        self.currIter = LookaheadIterator(iter(self.vocab.nonLeaves()))
        for p in self.currLevelProgs:
            self.updateBank(p)
        self.currLevelProgs.clear()
        return self.advanceRoot()

    def getNextProgram(self):
        res = None
        while not res:
            if self.childrenIterator.hasNext():
                children = self.childrenIterator.next()
                if (children is None and self.rootMaker.arity == 0) or (self.rootMaker.arity == len(children) and
                all(child.nodeType == child_type for child, child_type in zip(children, self.rootMaker.childTypes))):
                    prog = self.rootMaker.execute(self.task, children)
                    if self.oeManager.is_representative(prog) or children is None:
                        res = prog
            elif self.currIter.hasNext():
                if (not self.advanceRoot()):
                    return None
            else:
                if (not self.changeLevel()):
                    self.changeLevel()
        self.currLevelProgs.append(res)
        return res

    def updateBank(self, program):
        if program.size not in self.bank:
            self.bank[program.size] = [program]
        else:
            self.bank[program.size].append(program)

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