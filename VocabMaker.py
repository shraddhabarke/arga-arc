from transform import *
import transform
from typing import Union, List, Dict, Iterator, Any
from filters import *
from task import *
import unittest

class VocabFactory:
    def __init__(self, leavesMakers: List[Union[TransformASTNode, FilterASTNode]], nodeMakers: List[Union[TransformASTNode, FilterASTNode]]):
        self._leavesMakers = leavesMakers
        self._nodeMakers = nodeMakers

    def leaves(self) -> Iterator[Union[TransformASTNode, FilterASTNode]]:
        return iter(self._leavesMakers)

    def nonLeaves(self) -> Iterator[Union[TransformASTNode, FilterASTNode]]:
        return iter(self._nodeMakers)

    @classmethod
    def create(cls, vocabMakers: List[Union[TransformASTNode, FilterASTNode]]):
        leavesMakers, nodeMakers = [], []
        for maker in vocabMakers:
            if maker.arity == 0:
                leavesMakers.extend(maker.get_all_values())
            else:
                nodeMakers.append(maker)
        return cls(leavesMakers, nodeMakers)

class TestVocabFactory(unittest.TestCase):
    def setUp(self):
        self.leaf_makers = [Color.C0, Color.C1, Color.C2, Color.C3, Color.C4, Color.C5, Color.C6, Color.C7, Color.C8, Color.C9, Color.LEAST, Color.MOST, NoOp()]
        self.node_makers = [UpdateColor, MoveNode, ExtendNode, MoveNodeMax, RotateNode, AddBorder, FillRectangle, HollowRectangle, Insert, Mirror, Flip]
        self.vocab_factory = VocabFactory(self.leaf_makers, self.node_makers)

    def test_create(self):
        all_transform_classes = [Color, UpdateColor, MoveNode, ExtendNode, MoveNodeMax, RotateNode, AddBorder, FillRectangle, HollowRectangle, Insert, Mirror, Flip, NoOp] # Transforms
        vocab_factory_from_create = VocabFactory.create(all_transform_classes)
        self.assertEqual(list(vocab_factory_from_create.leaves()), self.leaf_makers)
        self.assertEqual(list(vocab_factory_from_create.nonLeaves()), self.node_makers)

class TestFilterVocabFactory(unittest.TestCase):
    def setUp(self):
        self.leaf_makers = [FColor.C0, FColor.C1,  FColor.C2, FColor.C3, FColor.C4, FColor.C5, FColor.C6, FColor.C7, FColor.C8, FColor.C9, FColor.LEAST, FColor.MOST]
        self.node_makers = [And, Or, FilterByColor, FilterBySize, FilterByDegree, FilterByNeighborSize, FilterByNeighborColor, FilterByNeighborDegree]
        self.vocab_factory = VocabFactory(self.leaf_makers, self.node_makers)

    def test_create(self):
        all_filter_classes = [Degree, FColor, And, Or, FilterByColor, FilterBySize, FilterByDegree, FilterByNeighborSize, FilterByNeighborColor, FilterByNeighborDegree]
        vocab_factory_from_create = VocabFactory.create(all_filter_classes)
        self.assertEqual(list(vocab_factory_from_create.leaves()), self.leaf_makers)
        self.assertEqual(list(vocab_factory_from_create.nonLeaves()), self.node_makers)

if __name__ == "__main__":
    taskNumber = "bb43febb"
    task = Task("dataset/" + taskNumber + ".json")
    task.abstraction = "nbccg"
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                               input in task.train_input]
    task.get_static_object_attributes(task.abstraction)
    setup_size_and_degree_based_on_task(task)
    unittest.main()