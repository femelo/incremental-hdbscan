import sys
from math import nan
def merge_dicts(d1, d2):
    d3 = d1.copy()
    d3.update(d2)
    return d3

class DisjointSet(object):
    def __init__(self, elements=[]):
        self.elements = []
        self.ids = []
        self.parents = []
        self.ranks = []
        self.indexes = {}
        self.next_id = 0
        self.make_set(elements)

    def make_set(self, elements):
        n = len(elements)
        self.elements += elements
        new_ids = list(range(self.next_id, self.next_id + n))
        self.ids += new_ids
        self.parents += new_ids
        self.ranks += n * [0]
        self.next_id += n

    def find(self, x):
        while self.parents[x] != x:
            x, self.parents[x] = self.parents[x], self.parents[self.parents[x]]
        return x

    def union(self, x, y):
        # Replace nodes by roots
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        # If necessary, rename variables to ensure that
        # x has rank at least as large as that of y
        if self.ranks[x] < self.ranks[y]:
            x, y = y, x

        # Make x the new root
        self.parents[y] = x
        # Increment the rank if needed
        if self.ranks[x] == self.ranks[y]:
            self.ranks[x] += 1

