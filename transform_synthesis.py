# taskName: str, vocab: VocabFactory, oeManager: OEValuesManager, contexts: List[Map[String, Any]]):
from transform import *
from task import *
import unittest
from typing import Union, List, Dict, Iterator
from enum import Enum
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator

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
        self.rootMaker = next(self.currIter)
        self.childrenIterator = LookaheadIterator(iter([None])) #TODO: should depend on current rootMaker later

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
        if not self.currIter.has_next():
            return False
        self.rootMaker = next(self.currIter)
        if self.rootMaker.arity == 0 and self.rootMaker.size == self.costLevel:
            self.childrenIterator = LookaheadIterator(iter([None])) 
        elif self.rootMaker.size < self.costLevel: 
            print("UpdateColor's children???")
            print(self.rootMaker.childTypes)
            self.childrenIterator = ChildrenIterator(self.currLevelProgs, self.rootMaker.childTypes, self.height)
        else:
            self.childrenIterator = LookaheadIterator(iter([])) 
        return True

    def changeLevel(self) -> bool:
        self.currIter = LookaheadIterator(iter(self.vocab.nonLeaves()))  # TODO: only non-terminals? what about Transforms?
        self.costLevel += 1
        for p in self.currLevelProgs:
            self.updateBank(p)
        self.currLevelProgs.clear() #TODO: does clear work?
        return self.advanceRoot()

    def getNextProgram(self):
        res = None
        while not res:
            if self.childrenIterator.has_next():
                children = next(self.childrenIterator)
                if (children is None and self.rootMaker.arity == 0) or (self.rootMaker.arity == len(children)): # TODO: and check types!
                    prog = self.rootMaker.apply(children)
                    # if (oeManager.isRepresentative(prog))
                    print("Res:", prog)
                    res = prog
            elif self.currIter.has_next():
                if (not self.advanceRoot()):
                    return None
            else:
                if (not self.changeLevel()):
                    return None
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
        self.assertEqual("UpdateColor(Color.C1)", self.enumerator.next().code)
        #self.assertTrue(self.enumerator.hasNext())

if __name__ == "__main__":
    unittest.main()