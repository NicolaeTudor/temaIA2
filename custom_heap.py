"""
    Author: jsbueno
    Source: https://stackoverflow.com/a/8875823
"""

import heapq


class CustomHeap(object):
    """
        A wrapper for heap that allows custom comparator
    """
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.index = 0
        if initial:
            self.data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self.data)
            heapq.heapify(self.data)
        else:
            self.data = []

    def push(self, item):
        heapq.heappush(self.data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        return heapq.heappop(self.data)[2]

    def heapify(self):
        heapq.heapify(self.data)

    def gen_key(self, item):
        return self.key(item)


