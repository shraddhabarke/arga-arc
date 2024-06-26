# taskName: str, vocab: VocabFactory, oeManager: OEValuesManager, contexts: List[Map[String, Any]]):
import unittest
from typing import Union, List, Dict, Iterator
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator
from childrenIterator import ChildrenIterator
from filters import *
from OEValuesManager import *
from task import Task
from itertools import chain

class FSizeEnumerator:
    def __init__(self, task: Task, vocab: VocabFactory, oeManager: ValuesManager):
        self.vocab = vocab
        self.oeManager = oeManager
        self.nextProgram = None
        self.bank: Dict[int, List[FilterASTNode]] = {}
        self.costLevel = 1
        self.currLevelProgs: List[FilterASTNode] = []
        self.currIter = LookaheadIterator(chain(iter(vocab.leaves()), iter(self.vocab.nonLeaves())))
        self.rootMaker = self.currIter.next()
        self.childrenIterator = LookaheadIterator(iter([None]))
        self.childrenIterators = []
        self.currentChildIteratorIndex = 0
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
        elif self.rootMaker.arity > 0: # TODO: Cost-based enumeration
            childrenCost = self.costLevel - self.rootMaker.default_size
            self.childrenIterators = [ChildrenIterator(
                childType, childrenCost, self.bank) for childType in self.rootMaker.childTypes]
            self.currentChildIteratorIndex = 0  # Keep track of which iterator is current
            self.childrenIterator = self.childrenIterators[self.currentChildIteratorIndex]
            self.rootMaker.arity = len(self.childrenIterator.childTypes)
        else:
            self.childrenIterator = LookaheadIterator(iter([]))
        return True

    def changeLevel(self) -> bool:
        self.costLevel += 1
        self.currIter = LookaheadIterator(chain(iter(self.vocab.leaves()), iter(self.vocab.nonLeaves())))
        for p in self.currLevelProgs:
            self.updateBank(p)
        self.currLevelProgs.clear()
        return self.advanceRoot()

    def getNextProgram(self):
        res = None
        while not res:
            if self.costLevel > 14: # TODO: parallelize filter and transform synthesis # todo: 12
                break
            if self.childrenIterator.hasNext():
                children = self.childrenIterator.next()
                if (children is None and self.rootMaker.arity == 0) or (self.rootMaker.arity == len(children) and
                all(child.nodeType == child_type for child, child_type in zip(children, self.rootMaker.childTypes[self.currentChildIteratorIndex]))):
                    prog = self.rootMaker.execute(self.task, children)
                    #print("OE:", prog.code, prog.size, prog.values)
                    if self.oeManager.is_frepresentative(prog) or children is None:
                        res = prog
            elif self.currentChildIteratorIndex + 1 < len(self.childrenIterators):
                self.currentChildIteratorIndex += 1
                self.childrenIterator = self.childrenIterators[self.currentChildIteratorIndex]
                self.rootMaker.arity = len(self.childrenIterator.childTypes)
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