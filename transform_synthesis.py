# taskName: str, vocab: VocabFactory, oeManager: OEValuesManager, contexts: List[Map[String, Any]]):
from transform import *
from task import *
import unittest
from typing import Union, List, Dict, Iterator
from enum import Enum
from VocabMaker import VocabFactory
from lookaheadIterator import LookaheadIterator
from childrenIterator import ChildrenIterator
from itertools import chain

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
        self.childrenIterators = []
        self.currentChildIteratorIndex = 0
        self.maxterminals = max(
            [nonleaf.default_size + nonleaf.arity for nonleaf in vocab.nonLeaves()])
        self.programCounter = 0
        self.currentValueSets = None
        self.currentProgram = None

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
        elif self.rootMaker.arity > 0:  # TODO: Cost-based enumeration
            childrenCost = self.costLevel - self.rootMaker.default_size
            self.childrenIterator = ChildrenIterator(
                self.rootMaker.childTypes, childrenCost, self.bank)
            self.childrenIterators = [ChildrenIterator(
                childType, childrenCost, self.bank) for childType in self.rootMaker.childTypes]
            self.currentChildIteratorIndex = 0  # Keep track of which iterator is current
            self.childrenIterator = self.childrenIterators[self.currentChildIteratorIndex]
        elif self.rootMaker.arity == 2 and self.rootMaker.childTypes == [Types.TRANSFORMS, Types.TRANSFORMS]:
            childrenCost = self.costLevel - 1
            self.childrenIterator = ChildrenIterator(
                self.rootMaker.childTypes, childrenCost, self.bank)
        else:
            self.childrenIterator = LookaheadIterator(iter([]))
        return True

    def changeLevel(self) -> bool:
        self.costLevel += 1
        #if self.costLevel > self.maxterminals + 2:
            #self.currIter = LookaheadIterator(iter([Transforms]))
        self.currIter = LookaheadIterator(chain(self.vocab.leaves(), self.vocab.nonLeaves(), [Transforms]))
        for p in self.currLevelProgs:
            self.updateBank(p)
        self.currLevelProgs.clear()
        return self.advanceRoot()

    def getNextProgram(self):
        if self.currentValueSets is not None and self.currentValueSets.hasNext():
            value = self.currentValueSets.get_nextValue()
            return self.createProgram(value)
        while not self.nextProgram:
            if self.costLevel > 25:
                break
            if self.childrenIterator.hasNext():
                children = self.childrenIterator.next()
                if (children is None and self.rootMaker.arity == 0) or (self.rootMaker.arity == len(children)
                    and all(child.nodeType == child_type for child, child_type
                    in zip(children, self.rootMaker.childTypes[self.currentChildIteratorIndex]))):
                    prog = self.rootMaker.apply(self.task, children, self.filter)
                    if isinstance(prog.values, VariableIterator):
                        self.programCounter = 0
                        self.currentValueSets = prog.values # save the iterator values
                        value = self.currentValueSets.get_nextValue() # first value in the iterator
                        self.currentProgram = prog
                        if self.oeManager.is_representative(value): # OE
                            return self.createProgram(value)
                    elif children is None or self.oeManager.is_representative(prog.values):
                        self.nextProgram = prog
                        if children is not None:
                            if any("Var" in child.code for child in children):
                                self.nextProgram.values_apply = self.task.values_to_apply[0]
            elif self.currentChildIteratorIndex + 1 < len(self.childrenIterators):
                self.currentChildIteratorIndex += 1
                self.childrenIterator = self.childrenIterators[self.currentChildIteratorIndex]

            elif self.currIter.hasNext():
                if (not self.advanceRoot()):
                    return None
            else:
                if (not self.changeLevel()):
                    self.changeLevel()

        if self.nextProgram:
            self.currLevelProgs.append(self.nextProgram)
            res = self.nextProgram
            self.nextProgram = None
            return res
        return None

    def createProgram(self, value_set):
        code_parts = self.currentProgram.code.rsplit('_', 1)
        if len(code_parts) > 1 and code_parts[-1].isdigit():
            base_code = code_parts[0]
        else:
            base_code = self.currentProgram.code
        new_code = f"{base_code}_{self.programCounter}"
        new_program = self.currentProgram.custom_copy()
        new_program.values = [value_set]
        new_program.code = new_code
        new_program.values_apply = self.task.values_to_apply[self.programCounter]
        new_program.spec = self.task.all_specs[self.programCounter]
        self.programCounter += 1
        self.currLevelProgs.append(new_program)
        return new_program

    def updateBank(self, program):
        if program.size not in self.bank:
            self.bank[program.size] = [program]
        else:
            self.bank[program.size].append(program)