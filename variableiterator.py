from typing import Union, List, Dict, Iterator
from transform import *
from lookaheadIterator import LookaheadIterator
from filters import *
from collections import defaultdict
from itertools import product

class VariableIterator:
    def __init__(self, task, input_graphs, output_graphs, filter, transformation):
        self.input_graphs = input_graphs
        self.output_graphs = output_graphs
        self.transformation = transformation
        self.task = task # todo: task
        self.filter = filter
        self.all_values = self.task.compute_all_transformed_values(filter, transformation, og_graph=task.input_abstracted_graphs_original[task.abstraction])
        self._next_value = None

    def hasNext(self):
        if self._next_value is not None:  # Already fetched the next value
            return True
        try:
            self._next_value = next(self.all_values)  # Try to fetch the next value
            return True
        except StopIteration:
            return False

    def get_nextValue(self):
        if self._next_value is not None:  # Return the cached value if it exists
            next_value = self._next_value
            self._next_value = None  # Reset the cache
        else:
            next_value = next(self.all_values)  # Fetch the next value
        self.task.reset_task()
        for value, input_abstracted_graph in zip(next_value, self.task.input_abstracted_graphs_original[self.task.abstraction]):
            input_abstracted_graph.var_apply_all(value, self.filter, self.transformation)
        return self.task.input_abstracted_graphs_original[self.task.abstraction]