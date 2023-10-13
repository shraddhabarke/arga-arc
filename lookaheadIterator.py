class LookaheadIterator:

    def __init__(self, iterable):
        self.iterator = iter(iterable)
        self.buffer = []

    def __iter__(self):
        return self

    def next(self):
        if self.buffer:
            return self.buffer.pop()
        else:
            return next(self.iterator)

    def hasNext(self):
        if self.buffer:
            return True

        try:
            self.buffer = [next(self.iterator)]
        except StopIteration:
            return False
        else:
            return True

x = LookaheadIterator(range(3))