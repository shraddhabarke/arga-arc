# taskName: str, vocab: VocabFactory, oeManager: OEValuesManager, contexts: List[Map[String, Any]]):
from transform import *
from task import *
import unittest
from typing import Union, List, Dict, Iterator
from enum import Enum
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator
from childrenIterator import ChildrenIterator
from filters import *

class FSizeEnumerator:
    def __init__(self, taskName: str, vocab: VocabFactory, oeManager, contexts):
        self.taskName = taskName
        self.vocab = vocab
        self.oeManager = oeManager
        self.contexts = contexts
        self.nextProgram = None
        self.bank: Dict[int, List[FilterASTNode]] = {}
        self.costLevel = 1
        self.currLevelProgs: List[FilterASTNode] = []
        self.currIter = LookaheadIterator(iter(vocab.leaves()))
        self.rootMaker = self.currIter.next()
        self.childrenIterator = LookaheadIterator(iter([None]))

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
            if self.costLevel > 10:
                break
            if self.childrenIterator.hasNext():
                children = self.childrenIterator.next()
                if (children is None and self.rootMaker.arity == 0) or (self.rootMaker.arity == len(children) and
                all(child.nodeType == child_type for child, child_type in zip(children, self.rootMaker.childTypes))):                    
                    prog = self.rootMaker.apply(children)
                    # evaluate here!
                    print("Res:", prog.code)
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
        # Create a mock oeManager that just returns true for any program.
        class MockOeManager:
            def isRepresentative(self, prog):
                return True
        vocabMakers = [Color, Exclude, FilterByColor, And]
        vocab = VocabFactory.create(vocabMakers)
        self.enumerator = FSizeEnumerator("TestTask", vocab, MockOeManager(), [])

    def test_next_program(self):
        print(list(self.enumerator.vocab.leaves()))
        self.assertEqual(list(self.enumerator.vocab.leaves()), [Color.C0, Color.C1, Color.C2, Color.C3, Color.C4, Color.C5, Color.C6, Color.C7, Color.C8, Color.C9, Color.LEAST, Color.MOST, Exclude.TRUE, Exclude.FALSE])
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
        self.assertEqual("Exclude.TRUE", self.enumerator.next().code)
        self.assertEqual("Exclude.FALSE", self.enumerator.next().code)
        self.assertEqual("FilterByColor(Color.C0, Exclude.TRUE)", self.enumerator.next().code)
        self.assertEqual("FilterByColor(Color.C0, Exclude.FALSE)", self.enumerator.next().code)
        self.assertEqual("FilterByColor(Color.C1, Exclude.TRUE)", self.enumerator.next().code)
        self.assertEqual("FilterByColor(Color.C1, Exclude.FALSE)", self.enumerator.next().code)
        self.assertEqual("FilterByColor(Color.C2, Exclude.TRUE)", self.enumerator.next().code)
        self.assertEqual("FilterByColor(Color.C2, Exclude.FALSE)", self.enumerator.next().code)
        self.assertEqual("FilterByColor(Color.C3, Exclude.TRUE)", self.enumerator.next().code)
        for i in range(18):
            self.enumerator.next()
        self.assertEqual("And(FilterByColor(Color.C0, Exclude.TRUE), FilterByColor(Color.C0, Exclude.FALSE))", self.enumerator.next().code)
        self.assertEqual("And(FilterByColor(Color.C0, Exclude.TRUE), FilterByColor(Color.C1, Exclude.TRUE))", self.enumerator.next().code)

if __name__ == "__main__":
    unittest.main()