from typing import Union, List, Dict, Iterator
from transform import *

class ChildrenIterator:
    def __init__(self, childTypes, childrenCost, bank):
        self.childTypes = childTypes
        self.childrenCost = childrenCost
        self.bank = bank
        self.childrenCosts = list(bank.keys())
        self.costs = self.get_costs(self.childrenCost, self.childrenCosts, len(self.childTypes))

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

if __name__ == "__main__":
    unittest.main()
