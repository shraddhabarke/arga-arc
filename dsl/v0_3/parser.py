import os
import sys

from lark import Lark

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import defs

class Parser:
    def __init__(self, grammar_file):
        if grammar_file == None:
            # grammar_file = os.path.join(PROJECT_ROOT, "dsl/dsl.lark")
            grammar_file = "dsl/dsl.lark"
        
        with open(grammar_file, "r") as f:
            self.grammar = f.read()
        self.parser = Lark(self.grammar, start="program", ambiguity="explicit")
        self.lib_parser = Lark(self.grammar, start="library", ambiguity="explicit")

    def parse_tree(self, program):
        return self.parser.parse(program)

    def lib_parse_tree(self, program):
        return self.lib_parser.parse(program)

if __name__ == '__main__':
    grammar_file = "dsl/v0_3/dsl.lark"
    parser = Parser(grammar_file)
    ref_dir = "dsl/v0_3/reference"
    for filename in os.listdir(ref_dir):
        if filename.endswith(".dsl"):
            with open(os.path.join(ref_dir, filename), "r") as f:
                lib = "(" + f.read() + ")"
            print(f"Testing {filename}...")
            try:
                t = parser.lib_parse_tree(lib)
                # print(t.pretty())
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
                exit(1)
    print("All tests passed!")