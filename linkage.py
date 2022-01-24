# cython: boundscheck=False
# cython: nonecheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD

import itertools
from re import S
import numpy as np
from disjoint_set import DisjointSet
from d_heap_queue import HeapQueue

DBL_MAX = np.finfo('d').max

def boruvka(distance_matrix):
    # Number of vertices
    n = distance_matrix.shape[0]

    local_matrix = distance_matrix.copy()
    local_matrix[np.eye(n).astype(bool)] = np.inf
    # Indexes of the cheapest destination nodes
    cheapest_destination = np.argmin(local_matrix, axis=1)
    edges = np.vstack((np.arange(cheapest_destination.shape[0]), cheapest_destination)).T

    # # not faster
    # # Subgraphs
    # G1 = {}
    # # Add first edge
    # next_edge = 0
    # subgraph_id = 0
    # while edges.shape[0] > 0:
    #     subgraph_done = False
    #     # Create new subgraph
    #     u, v = edges[next_edge]
    #     if u > v:
    #         u, v = v, u
    #     G1[subgraph_id] = {'V': [u, v], 'E': [[u, v]], 'W': [local_matrix[u, v]]}
    #     vertex_idx = 0
    #     edges = edges[next_edge+1:, :]
    #     while not subgraph_done:
    #         mask = np.logical_or(edges[:, 0] == u, edges[:, 1] == u)
    #         linked_vertices = edges[mask, :]
    #         if linked_vertices.shape[0] > 0:
    #             edges = edges[np.logical_not(mask), :]
    #         for i in range(linked_vertices.shape[0]):
    #             x, y = linked_vertices[i, :]
    #             if x > y:
    #                 x, y = y, x
    #             if not x in G1[subgraph_id]['V']: 
    #                 G1[subgraph_id]['V'].append(x)
    #             if not y in G1[subgraph_id]['V']:
    #                 G1[subgraph_id]['V'].append(y)
    #             if not [x, y] in G1[subgraph_id]['E']:
    #                 G1[subgraph_id]['E'].append([x, y])
    #                 G1[subgraph_id]['W'].append(local_matrix[x, y])
    #         if vertex_idx < len(G1[subgraph_id]['V']) - 1:
    #             vertex_idx += 1
    #             u = G1[subgraph_id]['V'][vertex_idx]
    #         else:
    #             subgraph_done = True
    #             subgraph_id += 1
    # G = G1

    F = []
    for u in range(cheapest_destination.shape[0]):
        v = int(cheapest_destination[u])
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
    if u > v:
        u, v = v, u
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
            G[vertex_to_subgraph[u]]['E'].append([u, v])
        elif u in vertex_to_subgraph and v in vertex_to_subgraph:
            # Need merging
            if len(G[vertex_to_subgraph[u]]['V']) < len(G[vertex_to_subgraph[v]]['V']):
                u, v = v, u
            subgraph = G[vertex_to_subgraph[v]]
            G[vertex_to_subgraph[u]]['V'] += subgraph['V']
            e = [u, v] if u < v else [v, u]
            G[vertex_to_subgraph[u]]['E'] += [e] + subgraph['E']
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
        i, j = np.unravel_index(idx, sub_matrix.shape)
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

def prim_with_heap(distance_matrix):
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
    in_heap = np.array(n * [True])
    # Initialize mst
    mst = n * [None]

    while min_heap.size() > 0:
        extracted_vertex = min_heap.pop()
        vertex_id = int(extracted_vertex[1])
        in_heap[vertex_id] = False
        # Get adjacency list
        adjacency_list = adjacency_lists[vertex_id]
        for next_vertex_id in adjacency_list:
            if in_heap[next_vertex_id]:
                next_key = distance_matrix[vertex_id, next_vertex_id]
                if (keys[next_vertex_id] > next_key):
                    min_heap.set_key(next_vertex_id, next_key)
                    mst[next_vertex_id] = [vertex_id, next_vertex_id, next_key]
                    keys[next_vertex_id] = next_key
    mst.remove(None)
    return np.array(mst)

KRUSGAL_THRESHOLD = 10000

def filter_edges(E, W, V):
    indexes = [i for i, e in enumerate(E) if V.find(e[0]) != V.find(e[1])]
    return E[indexes], W[indexes]

def kruskal_with_filter_on_graph(E, W, V, distance_matrix):
    if len(E) < KRUSGAL_THRESHOLD:
        F = kruskal_on_graph(E, W, V, distance_matrix)
        return F
    pivot = len(E) // 2
    indexes = np.argpartition(W, pivot)
    E1 = E[indexes[:pivot]]
    W1 = W[indexes[:pivot]]
    E2 = E[indexes[pivot:]]
    W2 = W[indexes[pivot:]]
    F1 = kruskal_with_filter_on_graph(E1, W1, V, distance_matrix)
    E2, W2 = filter_edges(E2, W2, V)
    F2 = kruskal_with_filter_on_graph(E2, W2, V, distance_matrix)
    return F1 + F2

def kruskal_on_graph(E, W, V, distance_matrix):
    indexes = np.argsort(W)
    W = W[indexes]
    E = E[indexes]
    F = []
    for e in E:
        u, v = e
        if V.find(u) != V.find(v):
            F.append([u, v, distance_matrix[u, v]])
            V.union(u,v)
    return F

def kruskal_with_filter(distance_matrix):
    # Number of vertices
    n = distance_matrix.shape[0]
    vertices = list(range(n))
    # Make set of vertices
    V = DisjointSet(elements=vertices)
    # Flatten distance matrix into sets of weight and edges
    W = np.array([distance_matrix[i, j] for j in range(n) for i in range(j)])
    E = np.array([[i, j] for j in range(n) for i in range(j)], dtype=np.int)
    F = kruskal_with_filter_on_graph(E, W, V, distance_matrix)
    mst_arr = np.array(F)
    return mst_arr 

def kruskal(distance_matrix):
    # Number of vertices
    n = distance_matrix.shape[0]
    vertices = list(range(n))
    # Make set of vertices
    V = DisjointSet(elements=vertices)
    # Flatten distance matrix into sets of weight and edges
    W = np.array([distance_matrix[i, j] for j in range(n) for i in range(j)])
    E = np.array([[i, j] for j in range(n) for i in range(j)], dtype=np.int)
    F = kruskal_on_graph(E, W, V, distance_matrix)
    mst_arr = np.array(F)
    return mst_arr

def mst_linkage_core(distance_matrix):
    result = np.zeros((distance_matrix.shape[0] - 1, 3))
    node_labels = np.arange(distance_matrix.shape[0], dtype=np.intp)
    current_node = 0
    current_distances = np.infty * np.ones(distance_matrix.shape[0])
    current_labels = node_labels
    for i in range(1, node_labels.shape[0]): # errado
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]

        left = current_distances[label_filter]
        right = distance_matrix[current_node][current_labels]
        current_distances = np.where(left < right, left, right)

        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        result[i - 1, 0] = current_node
        result[i - 1, 1] = new_node
        result[i - 1, 2] = current_distances[new_node_index]
        current_node = new_node

    return result


def mst_linkage_core_vector(raw_data, core_distances, dist_metric, alpha=1.0):

    dim = raw_data.shape[0]
    num_features = raw_data.shape[1]

    raw_data_view = raw_data[:raw_data.shape[0], :raw_data.shape[1]:1]
    raw_data_ptr = raw_data_view

    result_arr = np.zeros((dim - 1, 3))
    in_tree_arr = np.zeros(dim, dtype=np.int8)
    current_node = 0
    current_distances_arr = np.infty * np.ones(dim)
    current_sources_arr = np.ones(dim)

    result = result_arr[:dim - 1, :3:1]
    in_tree = in_tree_arr.astype(np.int8)
    current_distances = current_distances_arr
    current_sources = current_sources_arr
    current_core_distances = core_distances

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

            left_value = dist_metric.pairwise(raw_data_ptr[current_node, :].reshape(1,-1),
                                          raw_data_ptr[j, :].reshape(1,-1))[0, 0]
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

        result[i - 1, 0] = source_node
        result[i - 1, 1] = new_node
        result[i - 1, 2] = new_distance
        current_node = new_node

    return result_arr


class UnionFind (object):
    def __init__(self, N):
        self.parent_arr = -1 * np.ones(2 * N - 1, dtype=np.intp, order='C')
        self.next_label = N
        self.size_arr = np.hstack((np.ones(N, dtype=np.intp),
                                   np.zeros(N-1, dtype=np.intp)))
        self.parent = self.parent_arr
        self.size = self.size_arr

    def union(self, m, n):
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

        return

    def fast_find(self, n):
        p = n
        while self.parent_arr[n] != -1:
            n = self.parent_arr[n]
        # label up to the root
        while self.parent_arr[p] != n:
            p, self.parent_arr[p] = self.parent_arr[p], n
        return n


def label(L):
    result_arr = np.zeros((L.shape[0], L.shape[1] + 1))
    result = result_arr[:L.shape[0], :4:1]
    N = L.shape[0] + 1
    U = UnionFind(N)

    for index in range(L.shape[0]):
        a = int(L[index, 0])
        b = int(L[index, 1])
        delta = L[index, 2]

        aa, bb = U.fast_find(a), U.fast_find(b)

        result[index][0] = aa
        result[index][1] = bb
        result[index][2] = delta
        result[index][3] = U.size[aa] + U.size[bb]

        U.union(aa, bb)

    return result_arr


def single_linkage(distance_matrix):

    hierarchy = mst_linkage_core(distance_matrix)
    for_labelling = hierarchy[np.argsort(hierarchy.T[2]), :]

    return label(for_labelling)
