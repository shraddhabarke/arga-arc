from typing import Union, List, Dict, Iterator
from transform import *
from lookaheadIterator import LookaheadIterator

class ChildrenIterator:
    def __init__(self, childTypes, childrenCost, bank):
        self.childTypes = childTypes
        self.childrenCost = childrenCost
        self.bank = bank
        self.childrenCosts = list(bank.keys())
        self.costs = ProbCosts.getCosts(self.childrenCost, self.childrenCosts, len(self.childTypes))
        self.nextChild = None
        self.costsIterator = LookaheadIterator(self.costs)
        self.childrenLists: List[List[TransformASTNode]] = []
        self.candidates: List[LookaheadIterator[TransformASTNode]] = []
        self.allExceptLast: List[TransformASTNode] = []

    def resetIterators(self, cost: List[int]):
        self.childrenLists = [
            [node for node in self.bank[c] if node.nodeType == t]
            for t, c in zip(self.childTypes, cost)
        ]
        self.candidates = [
            LookaheadIterator(child_list) if child_list else LookaheadIterator([])
            for child_list in self.childrenLists
        ]
        if self.candidates and (self.candidates[0]).has_next():
            self.allExceptLast = [next(candidate) for candidate in self.candidates[:-1]]

    def getChild(self):
        self.next_child = None
        while not self.next_child:
            self.next_child = self.getNextChild()
            if not self.next_child:
                if not self.costsIterator.hasNext():
                    return
                newCost = next(self.costsIterator)
                self.resetIterators(newCost)

    def hasNext(self) -> bool:
        if not self.next_child:
            self.getChild()
        return next_child is not None

    def next(self) -> List[TransformASTNode]:
        if not self.next_child:
            self.getChild()
        res = self.next_child
        self.next_child = None
        return res

from itertools import combinations, permutations
from typing import List

class ProbCosts:
    @staticmethod
    def getCosts(childrenCost, childrenCosts, childrenArity, memo=None):
        if childrenArity == 1:
            return [[cost] for cost in childrenCosts if cost == childrenCost]
        if memo is None:
            memo = {}
        key = (childrenCost, childrenArity)
        if key in memo:
            return memo[key]
        valid_combinations = []
        for i, cost in enumerate(childrenCosts):
            if cost <= childrenCost:
                subcombinations = ProbCosts.getCosts(childrenCost - cost, childrenCosts, childrenArity - 1, memo)
                for subcombo in subcombinations:
                    valid_combinations.append([cost] + subcombo)
        memo[key] = valid_combinations
        return valid_combinations

import unittest

class TestGetCosts(unittest.TestCase):
    def test_single_arity(self):
        costs = ProbCosts.getCosts(5, [5, 10, 15], 1)
        self.assertEqual(costs, [[5]])
        costs = ProbCosts.getCosts(10, [5, 10, 15], 1)
        self.assertEqual(costs, [[10]])
        costs = ProbCosts.getCosts(20, [5, 10, 15], 1)
        self.assertEqual(costs, [])  # No single cost matches 20
    def test_arity_two(self):
        costs = ProbCosts.getCosts(20, [5, 10, 15], 2)
        self.assertCountEqual(costs, [[5, 15], [10, 10], [15, 5]])
        costs = ProbCosts.getCosts(30, [5, 10, 15], 2)
        self.assertCountEqual(costs, [[15, 15]])
    def test_arity_three(self):
        costs = ProbCosts.getCosts(30, [5, 10, 15], 3)
        self.assertCountEqual(costs, [[5, 10, 15], [5, 15, 10], [10, 5, 15], [10, 10, 10], [10, 15, 5], [15, 5, 10], [15, 10, 5]])
    def test_no_valid_combinations(self):
        costs = ProbCosts.getCosts(50, [5, 10, 15], 2)
        self.assertEqual(costs, [])
        costs = ProbCosts.getCosts(100, [5, 10, 15], 3)
        self.assertEqual(costs, [])

class ASTNode:
    def __init__(self, nodeType):
        self.nodeType = nodeType

import unittest
class TestResetIterators(unittest.TestCase):
    def test_reset_iterators_advanced(self):
        childTypes = [Types.COLOR, Types.DIRECTION, Types.ROTATION_DIRECTION]
        bank = {
        1: [Color.C1, Color.C2, Direction.LEFT, Overlap.TRUE],
        2: [Direction.UP, Color.C3, Rotation_Direction.CCW],
        3: [Overlap.FALSE, Rotation_Direction.CW]
        }
        cost_scenarios = [
            ([1, 1, 2], [Color.C1, Direction.LEFT]), # allExceptLast
            ([2, 1, 3], [Color.C3, Direction.LEFT])]
        self.obj = ChildrenIterator(childTypes, 6, bank)
        for cost, expected_all_except_last in cost_scenarios:
            self.obj.resetIterators(cost)
            self.assertEqual(len(self.obj.childrenLists), len(childTypes))
            self.assertEqual(len(self.obj.candidates), len(childTypes))
            self.assertEqual(self.obj.allExceptLast, expected_all_except_last)

if __name__ == "__main__":
    unittest.main()
