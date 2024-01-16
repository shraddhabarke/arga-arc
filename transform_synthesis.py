# taskName: str, vocab: VocabFactory, oeManager: OEValuesManager, contexts: List[Map[String, Any]]):
from transform import *
from task import *
import unittest
from typing import Union, List, Dict, Iterator
from enum import Enum
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator
from childrenIterator import ChildrenIterator


class TSizeEnumerator:
    def __init__(self, task: Task, vocab: VocabFactory, oeManager, filterast=None, contexts=[]):
        self.task = task
        self.vocab = vocab
        self.oeManager = oeManager
        self.contexts = contexts
        self.filter = filterast
        self.nextProgram = None
        self.bank: Dict[int, List[TransformASTNode]] = {}
        self.costLevel = 1
        self.currLevelProgs: List[TransformASTNode] = []
        self.currIter = LookaheadIterator(iter(vocab.leaves()))
        self.rootMaker = self.currIter.next()
        self.childrenIterator = LookaheadIterator(iter([None]))
        self.maxterminals = max(
            [nonleaf.default_size + nonleaf.arity for nonleaf in vocab.nonLeaves()])

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
        if self.rootMaker.arity == 0 and self.rootMaker.size == self.costLevel:
            self.childrenIterator = LookaheadIterator(iter([None]))
        elif self.rootMaker.arity == 0 and self.rootMaker.nodeType == Types.TRANSFORMS:
            self.childrenIterator = LookaheadIterator(iter([None]))
        elif self.rootMaker.childTypes == [Types.TRANSFORMS, Types.TRANSFORMS] and self.rootMaker.arity == 2:
            childrenCost = self.costLevel - 1
            self.childrenIterator = ChildrenIterator(
                self.rootMaker.childTypes, childrenCost, self.bank)
        elif self.rootMaker.arity > 0:  # TODO: Cost-based enumeration
            childrenCost = self.costLevel - self.rootMaker.default_size
            self.childrenIterator = ChildrenIterator(
                self.rootMaker.childTypes, childrenCost, self.bank)
        else:
            self.childrenIterator = LookaheadIterator(iter([]))
        return True

    def changeLevel(self) -> bool:
        self.costLevel += 1
        if self.costLevel > self.maxterminals + 2:
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
            if self.costLevel > 25:  # TODO: test transform sequence
                break
            if self.childrenIterator.hasNext():
                children = self.childrenIterator.next()
                if (children is None and self.rootMaker.arity == 0) or (self.rootMaker.arity == len(children) and
                                                                        all(child.nodeType == child_type for child, child_type in zip(children, self.rootMaker.childTypes))):
                    prog = self.rootMaker.apply(
                        self.task, children, self.filter)
                    if children is None:
                        res = prog
                    elif "Var" in prog.code: # always add variable programs
                        res = prog
                    elif self.oeManager.is_representative(prog.values):
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
