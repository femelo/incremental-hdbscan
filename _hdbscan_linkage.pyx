# cython: boundscheck=False
# cython: nonecheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD

import numpy as np
cimport numpy as np
import itertools

from libc.float cimport DBL_MAX, DBL_MIN

from dist_metrics cimport DistanceMetric
from _d_heap_queue cimport HeapQueue

cpdef np.ndarray[np.double_t, ndim=2] boruvka(
    np.ndarray[np.double_t, ndim=2] distance_matrix):
    cdef int n, n_g, i_g, i_g1, i_g2, idx, i, j, u, v, w
    cdef np.ndarray[np.double_t, ndim=2] local_matrix, sub_matrix, distance_sub_matrix, mst, contracted_mst
    cdef np.ndarray[np.intp_t, ndim=1] cheapest_destination
    cdef list F, e, V_g, V_g1, V_g2
    cdef dict G, vertex_to_subgraph, subgraph, contracted_edges
    # Number of vertices
    n = distance_matrix.shape[0]

    local_matrix = distance_matrix.copy()
    local_matrix[np.eye(n).astype(bool)] = np.inf
    # Indexes of the cheapest destination nodes
    cheapest_destination = np.argmin(local_matrix, axis=1)
    F = []
    for u in range(cheapest_destination.shape[0]):
        v = cheapest_destination[u]
        if u > v:
            u, v = v, u
        if not [u, v] in F and not [v, u] in F: 
            F.append([u, v])

    # Subgraphs
    G = {}
    vertex_to_subgraph = {}

    # Add first edge
    e = F.pop(0)
    u, v = e
    G[0] = {'V': [u, v], 'E': [e]}
    vertex_to_subgraph[u] = 0
    vertex_to_subgraph[v] = 0
    while len(F) > 0:
        u, v = F.pop(0)
        if u in vertex_to_subgraph and not v in vertex_to_subgraph:
            vertex_to_subgraph[v] = vertex_to_subgraph[u]
            G[vertex_to_subgraph[v]]['V'].append(v)
            G[vertex_to_subgraph[v]]['E'].append([u, v])
        elif not u in vertex_to_subgraph and v in vertex_to_subgraph:
            vertex_to_subgraph[u] = vertex_to_subgraph[v]
            G[vertex_to_subgraph[u]]['V'].append(u)
            G[vertex_to_subgraph[u]]['E'].append([v, u])
        elif u in vertex_to_subgraph and v in vertex_to_subgraph:
            # Need merging
            if len(G[vertex_to_subgraph[u]]['V']) < len(G[vertex_to_subgraph[v]]['V']):
                u, v = v, u
            subgraph = G[vertex_to_subgraph[v]]
            G[vertex_to_subgraph[u]]['V'] += subgraph['V']
            G[vertex_to_subgraph[u]]['E'] += [[u, v]] + subgraph['E']
            del G[vertex_to_subgraph[v]]
            for w in subgraph['V']:
                vertex_to_subgraph[w] = vertex_to_subgraph[u]
        else:
            # New graph
            i = max(G.keys()) + 1
            G[i] = {'V': [u, v], 'E': [[u, v]]}
            vertex_to_subgraph[u] = i
            vertex_to_subgraph[v] = i

    # Rename graphs
    subgraphs = G.values()
    G = dict(zip(list(range(len(subgraphs))), subgraphs))
    # Add edges to the mst
    E = []
    for i_g in G:
        for e in G[i_g]['E']:
            u, v = e
            E.append([u, v, distance_matrix[u, v]])
    mst = np.array(E)
    # If just one graph, then it is the MST
    n_g = len(G)
    if n_g == 1:
        return mst

    contracted_edges = {}
    distance_sub_matrix = np.infty * np.ones((n_g, n_g))
    for i_g1, i_g2 in itertools.combinations(list(G.keys()), 2):
        V_g1 = G[i_g1]['V']
        V_g2 = G[i_g2]['V']
        sub_matrix = local_matrix[np.ix_(V_g1, V_g2)]
        idx = np.argmin(sub_matrix.ravel())
        i, j = np.unravel_index(idx, (sub_matrix.shape[0], sub_matrix.shape[1]))
        contracted_edges[(i_g1, i_g2)] = [V_g1[i], V_g2[j]]
        distance_sub_matrix[i_g1, i_g2] = sub_matrix[i, j]
        distance_sub_matrix[i_g2, i_g1] = sub_matrix[i, j]

    # If just two graphs, it just link them and return
    if n_g == 2:
        mst = np.vstack((mst, contracted_edges[(0, 1)] + [distance_sub_matrix[0, 1]]))
        return mst

    # Call recursively the Boruvka algorithm for the contracted graph
    contracted_mst = boruvka(distance_sub_matrix)
    # Recover the original vertice ids
    for i in range(contracted_mst.shape[0]):
        u, v = contracted_mst[i, :2].astype(int)
        if u > v:
            u, v = v, u
        contracted_mst[i, :2] = contracted_edges[(u, v)]
    # Join forests by including the edges of the contracted MST
    mst = np.concatenate((mst, contracted_mst), axis=0)
    return mst

cpdef np.ndarray[np.double_t, ndim=2] prim_with_heap(
    np.ndarray[np.double_t, ndim=2] distance_matrix):
    cdef int n
    cdef int v
    cdef list vertices
    cdef list adjacency_lists
    cdef list adjacency_list
    cdef np.ndarray[np.double_t, ndim=1] keys
    cdef np.ndarray[np.intp_t, ndim=1] in_heap
    cdef HeapQueue min_heap
    cdef list null_edge
    cdef list extracted_vertex
    cdef int vertex_id
    cdef int next_vertex_id
    cdef float next_key
    cdef np.ndarray[np.double_t, ndim=2] mst, mst_arr
    # Number of vertices
    n = distance_matrix.shape[0]
    vertices = list(range(n))
    # Set adjacency list (it is a fully connected graph)
    adjacency_lists = [list(set(range(n))-set([v])) for v in vertices]
    # Initialize min-heap
    min_heap = HeapQueue(
        element_list=[[0.0, 0]] + [[DBL_MAX, v] for v in vertices[1:]], 
        number_of_children=2)
    keys = np.array([0.0] + (n - 1) * [DBL_MAX])
    in_heap = np.array(n * [1])
    # Initialize mst
    mst = np.infty * np.ones((n, 3))

    while min_heap.size() > 0:
        extracted_vertex = min_heap.pop()
        vertex_id = extracted_vertex[1]
        in_heap[vertex_id] = 0
        # Get adjacency list
        adjacency_list = adjacency_lists[vertex_id]
        for next_vertex_id in adjacency_list:
            if in_heap[next_vertex_id] > 0:
                next_key = distance_matrix[vertex_id, next_vertex_id]
                if (keys[next_vertex_id] > next_key):
                    min_heap.set_key(next_vertex_id, next_key)
                    mst[next_vertex_id, 0] = vertex_id
                    mst[next_vertex_id, 1] = next_vertex_id
                    mst[next_vertex_id, 2] = next_key
                    keys[next_vertex_id] = next_key
    mst_arr = mst[np.isfinite(mst[:, 0]), :]
    return mst_arr


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core(
                               np.ndarray[np.double_t,
                                          ndim=2] distance_matrix):

    cdef np.ndarray[np.intp_t, ndim=1] node_labels
    cdef np.ndarray[np.intp_t, ndim=1] current_labels
    cdef np.ndarray[np.double_t, ndim=1] current_distances
    cdef np.ndarray[np.double_t, ndim=1] left
    cdef np.ndarray[np.double_t, ndim=1] right
    cdef np.ndarray[np.double_t, ndim=2] result

    cdef np.ndarray label_filter

    cdef np.intp_t current_node
    cdef np.intp_t new_node_index
    cdef np.intp_t new_node
    cdef np.intp_t i

    result = np.zeros((distance_matrix.shape[0] - 1, 3))
    node_labels = np.arange(distance_matrix.shape[0], dtype=np.intp)
    current_node = 0
    current_distances = np.infty * np.ones(distance_matrix.shape[0])
    current_labels = node_labels
    for i in range(1, node_labels.shape[0]):
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        left = current_distances[label_filter]
        right = distance_matrix[current_node][current_labels]
        current_distances = np.where(left < right, left, right)

        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        result[i - 1, 0] = <double> current_node
        result[i - 1, 1] = <double> new_node
        result[i - 1, 2] = current_distances[new_node_index]
        current_node = new_node

    return result


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core_vector(
        np.ndarray[np.double_t, ndim=2, mode='c'] raw_data,
        np.ndarray[np.double_t, ndim=1, mode='c'] core_distances,
        DistanceMetric dist_metric,
        np.double_t alpha=1.0):

    # Add a comment
    cdef np.ndarray[np.double_t, ndim=1] current_distances_arr
    cdef np.ndarray[np.double_t, ndim=1] current_sources_arr
    cdef np.ndarray[np.int8_t, ndim=1] in_tree_arr
    cdef np.ndarray[np.double_t, ndim=2] result_arr

    cdef np.double_t * current_distances
    cdef np.double_t * current_sources
    cdef np.double_t * current_core_distances
    cdef np.double_t * raw_data_ptr
    cdef np.int8_t * in_tree
    cdef np.double_t[:, ::1] raw_data_view
    cdef np.double_t[:, ::1] result

    cdef np.ndarray label_filter

    cdef np.intp_t current_node
    cdef np.intp_t source_node
    cdef np.intp_t right_node
    cdef np.intp_t left_node
    cdef np.intp_t new_node
    cdef np.intp_t i
    cdef np.intp_t j
    cdef np.intp_t dim
    cdef np.intp_t num_features

    cdef double current_node_core_distance
    cdef double right_value
    cdef double left_value
    cdef double core_value
    cdef double new_distance

    dim = raw_data.shape[0]
    num_features = raw_data.shape[1]

    raw_data_view = (<np.double_t[:raw_data.shape[0], :raw_data.shape[1]:1]> (
        <np.double_t *> raw_data.data))
    raw_data_ptr = (<np.double_t *> &raw_data_view[0, 0])

    result_arr = np.zeros((dim - 1, 3))
    in_tree_arr = np.zeros(dim, dtype=np.int8)
    current_node = 0
    current_distances_arr = np.infty * np.ones(dim)
    current_sources_arr = np.ones(dim)

    result = (<np.double_t[:dim - 1, :3:1]> (<np.double_t *> result_arr.data))
    in_tree = (<np.int8_t *> in_tree_arr.data)
    current_distances = (<np.double_t *> current_distances_arr.data)
    current_sources = (<np.double_t *> current_sources_arr.data)
    current_core_distances = (<np.double_t *> core_distances.data)

    for i in range(1, dim):

        in_tree[current_node] = 1

        current_node_core_distance = current_core_distances[current_node]

        new_distance = DBL_MAX
        source_node = 0
        new_node = 0

        for j in range(dim):
            if in_tree[j]:
                continue

            right_value = current_distances[j]
            right_source = current_sources[j]

            left_value = dist_metric.dist(&raw_data_ptr[num_features *
                                                        current_node],
                                          &raw_data_ptr[num_features * j],
                                          num_features)
            left_source = current_node

            if alpha != 1.0:
                left_value /= alpha

            core_value = core_distances[j]
            if (current_node_core_distance > right_value or
                    core_value > right_value or
                    left_value > right_value):
                if right_value < new_distance:
                    new_distance = right_value
                    source_node = right_source
                    new_node = j
                continue

            if core_value > current_node_core_distance:
                if core_value > left_value:
                    left_value = core_value
            else:
                if current_node_core_distance > left_value:
                    left_value = current_node_core_distance

            if left_value < right_value:
                current_distances[j] = left_value
                current_sources[j] = left_source
                if left_value < new_distance:
                    new_distance = left_value
                    source_node = left_source
                    new_node = j
            else:
                if right_value < new_distance:
                    new_distance = right_value
                    source_node = right_source
                    new_node = j

        result[i - 1, 0] = <double> source_node
        result[i - 1, 1] = <double> new_node
        result[i - 1, 2] = new_distance
        current_node = new_node

    return result_arr


cdef class UnionFind (object):

    cdef np.ndarray parent_arr
    cdef np.ndarray size_arr
    cdef np.intp_t next_label
    cdef np.intp_t *parent
    cdef np.intp_t *size

    def __init__(self, N):
        self.parent_arr = -1 * np.ones(2 * N - 1, dtype=np.intp, order='C')
        self.next_label = N
        self.size_arr = np.hstack((np.ones(N, dtype=np.intp),
                                   np.zeros(N-1, dtype=np.intp)))
        self.parent = (<np.intp_t *> self.parent_arr.data)
        self.size = (<np.intp_t *> self.size_arr.data)

    cdef void union(self, np.intp_t m, np.intp_t n):
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

        return

    cdef np.intp_t fast_find(self, np.intp_t n):
        cdef np.intp_t p
        p = n
        while self.parent_arr[n] != -1:
            n = self.parent_arr[n]
        # label up to the root
        while self.parent_arr[p] != n:
            p, self.parent_arr[p] = self.parent_arr[p], n
        return n


cpdef np.ndarray[np.double_t, ndim=2] label(np.ndarray[np.double_t, ndim=2] L):

    cdef np.ndarray[np.double_t, ndim=2] result_arr
    cdef np.double_t[:, ::1] result

    cdef np.intp_t N, a, aa, b, bb, index
    cdef np.double_t delta

    result_arr = np.zeros((L.shape[0], L.shape[1] + 1))
    result = (<np.double_t[:L.shape[0], :4:1]> (
        <np.double_t *> result_arr.data))
    N = L.shape[0] + 1
    U = UnionFind(N)

    for index in range(L.shape[0]):

        a = <np.intp_t> L[index, 0]
        b = <np.intp_t> L[index, 1]
        delta = L[index, 2]

        aa, bb = U.fast_find(a), U.fast_find(b)

        result[index][0] = aa
        result[index][1] = bb
        result[index][2] = delta
        result[index][3] = U.size[aa] + U.size[bb]

        U.union(aa, bb)

    return result_arr


cpdef np.ndarray[np.double_t, ndim=2] single_linkage(distance_matrix):

    cdef np.ndarray[np.double_t, ndim=2] hierarchy
    cdef np.ndarray[np.double_t, ndim=2] for_labelling

    hierarchy = mst_linkage_core(distance_matrix)
    for_labelling = hierarchy[np.argsort(hierarchy.T[2]), :]

    return label(for_labelling)
