from transform import *
import transform
from typing import Union, List, Dict, Iterator, Any

class VocabFactory:
    def __init__(self, leavesMakers: List[TransformASTNode], nodeMakers: List[TransformASTNode]):
        self._leavesMakers = leavesMakers
        self._nodeMakers = nodeMakers

    def leaves(self) -> Iterator[TransformASTNode]:
        return iter(self._leavesMakers)

    def nonLeaves(self) -> Iterator[TransformASTNode]:
        return iter(self._nodeMakers)

    @classmethod
    def create(cls, vocabMakers: List[TransformASTNode]):
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

if __name__ == "__main__":
    unittest.main()