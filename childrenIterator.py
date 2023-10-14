from typing import Union, List, Dict, Iterator
from transform import *
from lookaheadIterator import LookaheadIterator
from filters import *

class ChildrenIterator:
    def __init__(self, childTypes, childrenCost, bank):
        self.childTypes = childTypes
        self.childrenCost = childrenCost
        self.bank = bank
        self.childrenCosts = list(bank.keys())
        self.costs = ProbCosts.getCosts(self.childrenCost, self.childrenCosts, len(self.childTypes))
        self.nextChild = None
        self.costsIterator = LookaheadIterator(self.costs)
        self.childrenLists: List[List[Union[TransformASTNode, FilterASTNode]]] = []
        self.candidates: List[LookaheadIterator[Union[TransformASTNode, FilterASTNode]]] = []
        self.allExceptLast: List[Union[TransformASTNode, FilterASTNode]] = []
        self.exhausted = False
        while not self.candidates:
            if not self.costsIterator.hasNext():  # Exit if no valid costs are left
                break
            newCost = self.costsIterator.next()
            self.resetIterators(newCost)

    def resetIterators(self, cost: List[int]):
        childrenListsTemp = [
            [node for node in self.bank[c] if node.nodeType == t]
            for t, c in zip(self.childTypes, cost)
        ]
        if any(not lst for lst in childrenListsTemp):
            #self.exhausted = True
            return
        self.childrenLists = childrenListsTemp
        self.candidates = [
            LookaheadIterator(child_list) if child_list else LookaheadIterator([])
            for child_list in self.childrenLists
        ]
        if self.candidates and (self.candidates[0]).hasNext():
            self.allExceptLast = [candidate.next() for candidate in self.candidates[:-1]]

    def getNextChild(self):
        if self.childrenLists == [] or any(not lst for lst in self.childrenLists):  # If iterator is marked as exhausted, return None
            return None
        elif self.candidates is not None:
            while True:
                if self.candidates[-1].hasNext():
                    children = self.allExceptLast + [self.candidates[-1].next()]
                    return children
                else:
                    next_candidates = [(idx, candidate) for idx, candidate in enumerate(self.candidates) if candidate.hasNext()]
                    if not next_candidates:  # If no candidates are left with elements
                        return None
                    idx, iterator = next_candidates[-1]  # Get the last available candidate
                    self.allExceptLast[idx] = iterator.next()
                    # Reset following iterators
                    for i in range(idx + 1, len(self.candidates) - 1):
                        self.candidates[i] = LookaheadIterator(self.childrenLists[i])
                        self.allExceptLast[i] = self.candidates[i].next()
                
                    # Reset the last candidate
                    self.candidates[-1] = LookaheadIterator(self.childrenLists[-1])
        else:
            return None
            
    def getChild(self):
        self.nextChild = None
        while not self.nextChild:
            self.nextChild = self.getNextChild()
            if not self.nextChild:
                if not self.costsIterator.hasNext():
                    return
                newCost = self.costsIterator.next()
                self.resetIterators(newCost)

    def hasNext(self) -> bool:
        if not self.costs or any(not lst for lst in self.childrenLists): # If iterator is marked as exhausted, return False
            return False
        elif self.costs and self.nextChild is None:
            self.getChild()
        return self.nextChild is not None

    def next(self) -> List[Union[TransformASTNode, FilterASTNode]]:
        if self.nextChild is None:
            self.getChild()
        res = self.nextChild
        self.nextChild = None
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

import unittest

class TestChildrenIterator(unittest.TestCase):

    def setUp(self):
        self.bank = {
            1: [Color.C1, Color.C2, Direction.LEFT],
            2: [Direction.UP, Color.C3],
            3: [Rotation_Direction.CW]
        }

    def test_basic(self):
        childTypes = [Types.COLOR, Types.DIRECTION, Types.ROTATION_DIRECTION]
        costs = [1, 2, 3]
        iterator = ChildrenIterator(childTypes, 6, self.bank)
        iterator.resetIterators(costs)
        self.assertEqual(len(iterator.childrenLists), 3)
        self.assertEqual(len(iterator.candidates), 3)
        self.assertEqual(iterator.allExceptLast, [Color.C1, Direction.UP])

    def test_hasNext_and_next_methods(self):
        childTypes = [Types.COLOR, Types.DIRECTION]
        iterator = ChildrenIterator(childTypes, 3, self.bank)
        self.assertTrue(iterator.hasNext())
        result = iterator.next()
        expected = [Color.C1, Direction.UP]
        self.assertListEqual(result, expected)
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.UP])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C3, Direction.LEFT])
    
    def test_complex_combinations(self):
        childTypes = [Types.COLOR, Types.DIRECTION, Types.ROTATION_DIRECTION, Types.OVERLAP]
        self.bank = {
            1: [Color.C1, Color.C2],
            2: [Direction.UP, Direction.LEFT],
            3: [Rotation_Direction.CW, Rotation_Direction.CCW],
            4: [Overlap.TRUE, Overlap.FALSE]
        }
        iterator = ChildrenIterator(childTypes, 10, self.bank)
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.UP, Rotation_Direction.CW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.UP, Rotation_Direction.CW, Overlap.FALSE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.UP, Rotation_Direction.CCW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.UP, Rotation_Direction.CCW, Overlap.FALSE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.LEFT, Rotation_Direction.CW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.LEFT, Rotation_Direction.CW, Overlap.FALSE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.LEFT, Rotation_Direction.CCW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C1, Direction.LEFT, Rotation_Direction.CCW, Overlap.FALSE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.UP, Rotation_Direction.CW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.UP, Rotation_Direction.CW, Overlap.FALSE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.UP, Rotation_Direction.CCW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.UP, Rotation_Direction.CCW, Overlap.FALSE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.LEFT, Rotation_Direction.CW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.LEFT, Rotation_Direction.CW, Overlap.FALSE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.LEFT, Rotation_Direction.CCW, Overlap.TRUE])
        self.assertTrue(iterator.hasNext())
        self.assertListEqual(iterator.next(), [Color.C2, Direction.LEFT, Rotation_Direction.CCW, Overlap.FALSE])
        self.assertFalse(iterator.hasNext())
        iterator = ChildrenIterator(childTypes, 200, self.bank)
        self.assertFalse(iterator.hasNext())

if __name__ == "__main__":
    unittest.main()