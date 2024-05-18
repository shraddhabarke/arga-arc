import lark
from lark import Lark, Transformer, v_args

class SExprStats(Transformer):
    def __init__(self):
        self.stats = {}
    def number(self, items):
        return int(items[0])
    def symbol(self, items):
        return str