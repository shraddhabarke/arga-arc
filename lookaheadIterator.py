class LookaheadIterator:

    def __init__(self, iterable):
        self.iterator = iter(iterable)
        self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.pop()
        else:
            return next(self.iterator)

    def has_next(self):
        if self.buffer:
            return True

        try:
            self.buffer = [next(self.iterator)]
        except StopIteration:
            return False
        else:
            return True

x  = LookaheadIterator(range(3))