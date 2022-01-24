# cython: boundscheck=False
# cython: nonecheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD

import cython
cimport cython

import numpy as np
cimport numpy as np

cpdef inline int compare_min_fn(list x, list y):
    return int(x < y)

cpdef inline int compare_max_fn(list x, list y):
    return int(x > y)

"""D-ary heap queue algorithm."""
cdef class HeapQueue(object):
    cdef int n
    cdef list heap
    cdef object compare_fn
    cdef list ids
    cdef str type
    cdef dict indexes
    cdef int d
    cdef int next_id
    cdef int key_index

    cdef int size(self)
    cdef void set_key(self, int item_id, float new_key) except *
    cdef void push(self, list item, int item_id=*)
    cdef list pop(self)
    cdef list replace(self, list item, int item_id=*)
    cdef list push_and_pop(self, list item, int item_id=*)
    cdef void heapify(self)
    cdef void sift_down(self, int start_pos, int pos)
    cdef void sift_up(self, int pos)