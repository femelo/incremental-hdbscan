"""D-ary heap queue algorithm."""
class HeapQueue(object):
    def __init__(self, element_list=[], heap_type='min', number_of_children=2):
        self.n = len(element_list)
        self.type = heap_type
        self.heap = element_list
        self.compare_fn = lambda x, y: x < y if heap_type == 'min' else lambda x, y: x > y
        self.ids = list(range(self.n))
        self.indexes = dict(zip(self.ids, self.ids))
        self.d = number_of_children
        self.next_id = self.n
        if self.n > 0:
            self.heapify()

    def size(self):
        return self.n

    def set_key(self, item_id, new_key):
        idx = self.indexes[item_id]
        new_item = self.heap[idx]
        new_item[0] = new_key
        self.sift_down(0, idx)
    
    def push(self, item, item_id=None):
        """Push item onto heap, maintaining the heap invariant."""
        if item_id is None:
            item_id = self.next_id
        self.next_id = item_id + 1
        self.heap.append(item)
        self.ids.append(item_id)
        self.indexes[item_id] = self.n
        self.n += 1
        self.sift_down(0, self.n - 1)

    def pop(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        last_element = self.heap.pop()    # raises appropriate IndexError if heap is empty
        last_id = self.ids.pop()
        del self.indexes[last_id]
        if self.heap:
            return_item = self.heap[0]
            return_item_id = self.ids[0]
            del self.indexes[return_item_id]
            self.heap[0] = last_element
            self.ids[0] = last_id
            self.indexes[last_id] = 0
            self.n = len(self.heap)
            self.sift_up(0)
            return return_item
        self.n = len(self.heap)
        return last_element

    def replace(self, item, item_id=None):
        """Pop and return the current smallest value, and add the new item.

        This is more efficient than heappop() followed by heappush(), and can be
        more appropriate when using a fixed-size heap.  Note that the value
        returned may be larger than item!  That constrains reasonable uses of
        this routine unless written as part of a conditional replacement:

            if item > heap[0]:
                item = replace(item)
        """
        if item_id is None:
            item_id = self.next_id
        self.next_id = item_id + 1
        return_item = self.heap[0]    # raises appropriate IndexError if heap is empty
        return_item_id = self.ids[0]
        del self.indexes[return_item_id]
        self.heap[0] = item
        self.ids[0] = item_id
        self.indexes[item_id] = 0
        self.sift_up(0)
        return return_item

    def push_and_pop(self, item, item_id=None):
        """Fast version of a heappush followed by a heappop."""
        if item_id is None:
            item_id = self.next_id
        self.next_id = item_id + 1
        if self.heap and self.heap[0] < item:
            item, self.heap[0] = self.heap[0], item
            item_id, self.ids[0] = self.ids[0], item_id
            del self.indexes[item_id]
            self.indexes[self.ids[0]] = 0
            self.sift_up(0)
        return item

    def heapify(self):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(self.n // self.d)):
            self.sift_up(i)

    # 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
    # is the index of a leaf with a possibly out-of-order value.  Restore the
    # heap invariant.
    def sift_down(self, start_pos, pos):
        new_item = self.heap[pos]
        new_id = self.ids[pos]
        # Follow the path to the root, moving parents down until finding a place
        # new_item fits.
        while pos > start_pos:
            parent_pos = (pos - 1) // self.d
            parent = self.heap[parent_pos]
            parent_id = self.ids[parent_pos]
            if self.compare_fn(new_item, parent):
                self.heap[pos] = parent
                self.ids[pos] = parent_id
                self.indexes[parent_id] = pos
                pos = parent_pos
                continue
            break
        self.heap[pos] = new_item
        self.ids[pos] = new_id
        self.indexes[new_id] = pos

    def sift_up(self, pos):
        end_pos = self.n
        start_pos = pos
        new_item = self.heap[pos]
        new_id = self.ids[pos]
        # Bubble up the smaller child until hitting a leaf.
        first_child = self.d * pos + 1
        child_pos = first_child    # leftmost child position
        while child_pos < end_pos:
            # Set child_pos to index of smaller child.
            for i in range(1, self.d):
                next_child_pos = first_child + i
                if next_child_pos < end_pos and \
                    not self.compare_fn(self.heap[child_pos], self.heap[next_child_pos]):
                    child_pos = next_child_pos
            # Move the smaller child up.
            self.heap[pos] = self.heap[child_pos]
            self.ids[pos] = self.ids[child_pos]
            self.indexes[self.ids[pos]] = pos
            pos = child_pos
            first_child = self.d * pos + 1
            child_pos = first_child
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = new_item
        self.ids[pos] = new_id
        self.indexes[new_id] = pos
        self.sift_down(start_pos, pos)

    def merge(*iterables, key=None, reverse=False):
        '''Merge multiple sorted inputs into a single sorted output.

        Similar to sorted(itertools.chain(*iterables)) but returns a generator,
        does not pull the data into memory all at once, and assumes that each of
        the input streams is already sorted (smallest to largest).

        >>> list(merge([1,3,5,7], [0,2,4,8], [5,10,15,20], [], [25]))
        [0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]

        If *key* is not None, applies a key function to each element to determine
        its sort order.

        >>> list(merge(['dog', 'horse'], ['cat', 'fish', 'kangaroo'], key=len))
        ['dog', 'cat', 'fish', 'horse', 'kangaroo']

        '''

        h = []
        h_append = h.append

        if reverse:
            heap_type = 'max'
            direction = -1
        else:
            heap_type = 'min'
            direction = +1

        if key is None:
            for order, it in enumerate(map(iter, iterables)):
                try:
                    next = it.__next__
                    h_append([next(), order * direction, next])
                except StopIteration:
                    pass
            heap_queue = HeapQueue(h, heap_type)
            while heap_queue.n > 1:
                try:
                    while True:
                        value, order, next = s = heap_queue.heap[0]
                        yield value
                        s[0] = next()           # raises StopIteration when exhausted
                        heap_queue.replace(s)   # restore heap condition
                except StopIteration:
                    _ = heap_queue.pop()        # remove empty iterator
            if heap_queue.heap:
                # fast case when only a single iterator remains
                value, order, next = heap_queue.heap[0]
                yield value
                yield from next.__self__
            return

        for order, it in enumerate(map(iter, iterables)):
            try:
                next = it.__next__
                value = next()
                h_append([key(value), order * direction, value, next])
            except StopIteration:
                pass
        heap_queue = HeapQueue(h, heap_type)
        while heap_queue.n > 1:
            try:
                while True:
                    key_value, order, value, next = s = heap_queue.heap[0]
                    yield value
                    value = next()
                    s[0] = key(value)
                    s[2] = value
                    heap_queue.replace(h, s)
            except StopIteration:
                _ = heap_queue.pop()
        if heap_queue.heap:
            key_value, order, value, next = heap_queue.heap[0]
            yield value
            yield from next.__self__


if __name__ == "__main__":

    import doctest
    print(doctest.testmod())
