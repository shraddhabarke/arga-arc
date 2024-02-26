import ast
from collections import defaultdict


class PatternExtractor(ast.NodeVisitor):
    def __init__(self):
        self.pattern_counts = defaultdict(int)

    def visit_Subscript(self, node):
        # Handle indexing and slicing operations
        if isinstance(node.slice, ast.Index):  # Indexing operation
            self.pattern_counts['IndexingOperation'] += 1
        elif isinstance(node.slice, ast.Slice):  # Slicing operation
            if isinstance(node.value, ast.Name):  # Axis 0
                if node.slice.lower and node.slice.upper:
                    self.pattern_counts['SlicingAxis0BothOperation'] += 1
                elif node.slice.lower:
                    self.pattern_counts['SlicingAxis0LeftOperation'] += 1
                elif node.slice.upper:
                    self.pattern_counts['SlicingAxis0RightOperation'] += 1
            elif isinstance(node.value, ast.Subscript) and isinstance(node.value.slice, ast.Index) and isinstance(node.value.slice.value, ast.Slice):  # Axis 1
                if node.slice.lower and node.slice.upper:
                    self.pattern_counts['SlicingAxis1BothOperation'] += 1
                elif node.slice.lower:
                    self.pattern_counts['SlicingAxis1LeftOperation'] += 1
                elif node.slice.upper:
                    self.pattern_counts['SlicingAxis1RightOperation'] += 1

        self.generic_visit(node)

    def visit_Tuple(self, node):
        # Handle tuple creation operations
        num_elements = len(node.elts)
        if num_elements == 1:
            self.pattern_counts['SingletonTupleCreationOperation'] += 1
        elif num_elements == 2:
            self.pattern_counts['PairCreationOperation'] += 1
        elif num_elements == 3:
            self.pattern_counts['TripleCreationOperation'] += 1

        self.generic_visit(node)


def extract_patterns(source_code):
    ast_tree = ast.parse(source_code)
    extractor = PatternExtractor()
    extractor.visit(ast_tree)
    return extractor.pattern_counts


completions = [
    "\ndef convert_to_pairs(tensor):\n    return tf.reshape(tensor, (-1, 2))\n\ntensor = tf.constant([0, 0, 0, 1, 3, 3])\npairs = convert_to_pairs(tensor)\n",
    "\ndef convert_indices(indices):\n    indices = tf.constant(indices)\n    second_elements = tf.where(indices)\n    first_elements = tf.subtract(second_elements, 1)\n    index_pairs = tf.stack([first_elements, second_elements], axis=-1)\n    return index_pairs\n\nindices = [0, 0, 0, 1, 3, 3]\n",
    "\ntensor = tf.constant([0, 0, 0, 1, 3, 3])\n\nsorted_tensor = tf.sort(tensor)\nduplicates = tf.equal(sorted_tensor, tf.roll(sorted_tensor, shift=1))\n\nindices = tf.where(duplicates)\n\npairs = tf.gather(tensor, indices)\n\npairs = tf.squeeze(pairs)\n\n",
    "\ndef index_to_pairs(indices):\n    tensor = tf.constant(indices)\n    unique_indices, counts = tf.unique_with_counts(tensor)\n    pairs = tf.transpose(tf.stack([unique_indices, counts]))\n    return pairs\n",
    "\ndef convert_index_to_pairs(index_tensor):\n    unique, counts = tf.unique_with_counts(index_tensor)\n    indices = tf.where(counts > 1)\n    pairs = tf.gather(unique, indices)\n    return pairs\n",
    "\ndef index_to_pairs(index_tensor):\n    unique_elements, _ = tf.unique_with_counts(index_tensor)\n    return tf.reshape(unique_elements, (-1, 2))\n",
    "\ndef index_to_pairs(index_tensor):\n    unique, counts = tf.unique_with_counts(index_tensor)\n    return tf.transpose(tf.stack([unique, counts]))\n\nindex_tensor = tf.constant([0, 0, 0, 1, 3, 3])\n",
    "\ndef tensor_to_pairs(tensor):\n    return tf.reshape(tensor, [-1, 2])\n\ntensor = tf.constant([0, 0, 0, 1, 3, 3])\n",
    "\ndef convert_indices(input_tensor):\n    condition = tf.equal(input_tensor, input_tensor)\n    output = tf.where(condition)\n    return output\n",
    "\ndef index_to_pairs(indices):\n    indices = tf.constant(indices)\n    unique_indices, counts = tf.unique_with_counts(indices)\n    row_indices = tf.expand_dims(unique_indices, -1)\n    col_indices = tf.expand_dims(tf.range(tf.size(unique_indices)), -1)\n    pairs = tf.stack([row_indices, col_indices], axis=-1)\n    return pairs\n\nindices = [0, 0, 0, 1, 3, 3]\npairs = index_to_pairs(indices)\n"
]
aggregate_pattern_counts = defaultdict(int)

for program in completions:
    pattern_counts = extract_patterns(program)
    for pattern, count in pattern_counts.items():
        aggregate_pattern_counts[pattern] += count

print(aggregate_pattern_counts)