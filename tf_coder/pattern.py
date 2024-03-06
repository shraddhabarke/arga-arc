import ast
from collections import defaultdict


class PatternExtractor(ast.NodeVisitor):
    def __init__(self):
        self.pattern_counts = defaultdict(int)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Tuple):
            elements = node.slice.elts
            if len(elements) == 2:
                first_dim, second_dim = elements
                if isinstance(first_dim, ast.Slice) and (first_dim.lower is None and first_dim.upper is None):
                    if not isinstance(second_dim, ast.Slice):
                        self.pattern_counts['IndexingAxis1Operation'] += 1
            for dim, item in enumerate(node.slice.elts):
                if isinstance(item, ast.Slice):
                    if item.lower is None and item.upper is None:
                        pass
                    elif item.lower is not None and item.upper is None:
                        if dim == 1:
                            self.pattern_counts['SlicingAxis1LeftOperation'] += 1
                    elif item.lower is None and item.upper is not None:
                        if dim == 1:
                            self.pattern_counts['SlicingAxis1RightOperation'] += 1
                    elif item.lower is not None and item.upper is not None:
                        if dim == 1:
                            self.pattern_counts['SlicingAxis1BothOperation'] += 1
        elif isinstance(node.slice, ast.Slice):
            if node.slice.lower and node.slice.upper:
                self.pattern_counts['SlicingAxis0BothOperation'] += 1
            elif node.slice.lower:
                self.pattern_counts['SlicingAxis0LeftOperation'] += 1
            elif node.slice.upper:
                self.pattern_counts['SlicingAxis0RightOperation'] += 1

        else:
            self.pattern_counts['IndexingOperation'] += 1
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


completions = ["arg0[arg1]\narg0[arg1:]\narg0[:arg1]\narg0[arg1:arg2]\n",
               "arg0[:, arg1:]\narg0[:, arg1:arg2]\narg0[:, :arg1]\narg0[:, arg1]"]
aggregate_pattern_counts = defaultdict(int)

for program in completions:
    pattern_counts = extract_patterns(program)
    for pattern, count in pattern_counts.items():
        aggregate_pattern_counts[pattern] += count

print(aggregate_pattern_counts)
