# taskName: str, vocab: VocabFactory, oeManager: OEValuesManager, contexts: List[Map[String, Any]]):
from transform import *
from task import *
import unittest
from typing import Union, List, Dict, Iterator
from enum import Enum
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator
from childrenIterator import ChildrenIterator

class SizeEnumerator:
    def __init__(self, taskName: str, vocab: VocabFactory, oeManager, contexts):
        self.taskName = taskName
        self.vocab = vocab
        self.oeManager = oeManager
        self.contexts = contexts
        self.nextProgram = None
        self.bank: Dict[int, List[TransformASTNode]] = {}
        self.costLevel = 1
        self.currLevelProgs: List[TransformASTNode] = []
        self.currIter = LookaheadIterator(iter(vocab.leaves()))
        self.rootMaker = self.currIter.next()
        self.childrenIterator = LookaheadIterator(iter([None]))

    def hasNext(self) -> bool:
        if self.nextProgram:
            return True
        else:
            self.nextProgram = self.getNextProgram()
            return self.nextProgram is not None

    def next(self) -> TransformASTNode:
        if not self.nextProgram:
            self.nextProgram = self.getNextProgram()
        res = self.nextProgram
        self.nextProgram = None
        return res

    def advanceRoot(self) -> bool:
        if not self.currIter.hasNext():
            return False
        self.rootMaker = self.currIter.next()
        if self.rootMaker.nodeType == Types.TRANSFORMS:
            if (self.costLevel - 1) % 2 == 0: 
                self.rootMaker.arity = (self.costLevel - 1) / 2
                self.rootMaker.childTypes = [Types.TRANSFORM_OPS] * int(self.rootMaker.arity)
            else:
                return False
        if self.rootMaker.arity == 1 and self.rootMaker.nodeType == Types.TRANSFORMS:
            return False
        if self.rootMaker.arity == 0 and self.rootMaker.size == self.costLevel:
            self.childrenIterator = LookaheadIterator(iter([None]))
        elif 1 < self.costLevel: # TODO: update this later for the cost-based enumeration
            print(self.rootMaker.childTypes)
            childrenCost = self.costLevel - 1
            self.childrenIterator = ChildrenIterator(self.rootMaker.childTypes, childrenCost, self.bank)
        else:
            self.childrenIterator = LookaheadIterator(iter([]))
        return True

    def changeLevel(self) -> bool:
        self.costLevel += 1
        if self.costLevel > 2:
            self.currIter = LookaheadIterator(iter([Transforms]))
        else:
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
                    prog = self.rootMaker.apply(children)
                    print("Res:", prog)
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
        vocabMakers = [Color, UpdateColor]
        vocab = VocabFactory.create(vocabMakers)
        self.enumerator = SizeEnumerator("TestTask", vocab, MockOeManager(), [])

    def test_next_program(self):
        self.assertEqual(list(self.enumerator.vocab.leaves()), [Color.C0, Color.C1, Color.C2, Color.C3, Color.C4, Color.C5, Color.C6, Color.C7, Color.C8, Color.C9, Color.LEAST, Color.MOST])
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
        self.assertEqual("updateColor(Color.C0)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
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
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.LEAST)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual("updateColor(Color.MOST)", self.enumerator.next().code)
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C0), updateColor(Color.C0)]")
        self.assertTrue(self.enumerator.hasNext())
        self.assertEqual(self.enumerator.next().code, "[updateColor(Color.C0), updateColor(Color.C1)]")

if __name__ == "__main__":
    unittest.main()