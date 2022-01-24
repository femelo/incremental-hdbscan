import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, os.getcwd())
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
from dist_metrics import DistanceMetric

#from linkage import mst_linkage_core, label
from linkage import kruskal_with_filter, prim_with_heap
from _hdbscan_linkage import mst_linkage_core, mst_linkage_core_vector, boruvka, prim_with_heap
#from _hdbscan_linkage import mst_linkage_core_vector
# from _hdbscan_linkage import mst_linkage_core_vector

def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    def within_tol(x, y, atol, rtol):
        return np.abs(x-y) <= atol + rtol * np.abs(y)

    x = np.array(a)
    y = np.array(b)

    # Make sure y is an inexact type to avoid bad behavior on abs(MIN_INT).
    # This will cause casting of x later. Also, make sure to allow subclasses
    # (e.g., for numpy.ma).
    y = np.array(y, copy=False)

    xfin = np.isfinite(x)
    yfin = np.isfinite(y)
    if np.all(xfin) and np.all(yfin):
        return within_tol(x, y, atol, rtol)
    else:
        finite = np.logical_and(xfin, yfin)
        cond = np.zeros(finite.shape)
        # Because we're using boolean indexing, x & y must be the same shape.
        # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
        # lib.stride_tricks, though, so we can't import it here.
        x = x * np.ones(cond.shape)
        y = y * np.ones(cond.shape)
        # Avoid subtraction with infinite/nan values...
        cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
        # Check for equality of infinite values...
        cond[~finite] = (x[~finite] == y[~finite])
        if equal_nan:
            # Make NaN == NaN
            both_nan = np.logical_and(np.isnan(x), np.isnan(y))

            # Needed to treat masked arrays correctly. = True would not work.
            cond[both_nan] = both_nan[both_nan]

        return cond[()]  # Flatten 0d arrays to scalars

def mutual_reachability(distance_matrix, min_points=5, alpha=1.0):
    size = distance_matrix.shape[0]
    min_points = min(size - 1, min_points)
    try:
        core_distances = np.partition(distance_matrix,
                                      min_points,
                                      axis=0)[min_points]
    except AttributeError:
        core_distances = np.sort(distance_matrix,
                                 axis=0)[min_points]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result

if __name__ == "__main__":
    from time import time
    n = 100
    np.random.seed(0)
    # Generate data
    X = 100.0 * np.random.randn(n, 2)

    # hdbscan_generic
    delta_t = []
    t_start = time()
    distance_matrix = pairwise_distances(X, metric="euclidean")
    reachability_graph = mutual_reachability(distance_matrix)
    delta_t.append(time() - t_start)
    mst = mst_linkage_core(reachability_graph)

    result_min_span_tree = mst.copy()
    for index, row in enumerate(result_min_span_tree[1:], 1):
        candidates = np.where(isclose(reachability_graph[:, int(row[1])], row[2]))[0]
        candidates = np.intersect1d(
            candidates, mst[:index, :2].astype(int)
        )
        candidates = candidates[candidates != row[1]]
        assert len(candidates) > 0
        row[0] = candidates[0]
    delta_t.append(time() - t_start - delta_t[0])
    mst1 = result_min_span_tree
    #mst1 = np.insert(mst1, 3, reachability_graph[mst1[:, 0].astype(int), mst1[:, 1].astype(int)], axis=1)
    # print('Tree 1: \n{}'.format(mst))
    print('Cost 1: {:f}'.format(np.sum(mst1[:, 2])))
    print('Runtime 1: {:f} + {:f} = {:f} s'.format(delta_t[0], delta_t[1], sum(delta_t)))

    # hdbscan_prims_kd_tree
    # Get distance to kth nearest neighbour
    delta_t = []
    t_start = time()
    tree = KDTree(X, metric="euclidean", leaf_size=40)
    core_distances = tree.query(X, k=5 + 1, dualtree=True, breadth_first=True)[0][
        :, -1
    ].copy(order="C")
    delta_t.append(time() - t_start)

    # Mutual reachability distance is implicit in mst_linkage_core_vector
    dist_metric = DistanceMetric.get_metric("euclidean")
    mst2 = mst_linkage_core_vector(X, core_distances, dist_metric, alpha=1.0)
    delta_t.append(time() - t_start - delta_t[0])
    #mst2 = np.insert(mst2, 3, reachability_graph[mst2[:, 0].astype(int), mst2[:, 1].astype(int)], axis=1)
    # print('Tree 2: \n{}'.format(mst2))
    print('Cost 2: {:f}'.format(np.sum(mst2[:, 2])))
    print('Runtime 2: {:f} + {:f} = {:f} s'.format(delta_t[0], delta_t[1], sum(delta_t)))

    delta_t = []
    t_start = time()
    distance_matrix = pairwise_distances(X, metric="euclidean")
    reachability_graph = mutual_reachability(distance_matrix)
    delta_t.append(time() - t_start)
    mst3 = prim_with_heap(reachability_graph)
    delta_t.append(time() - t_start - delta_t[0])
    #mst3 = np.insert(mst3, 3, reachability_graph[mst3[:, 0].astype(int), mst3[:, 1].astype(int)], axis=1)
    print('Tree 3: \n{}'.format(mst3))
    print('Cost 3: {:f}'.format(np.sum(mst3[:, 2])))
    print('Runtime 3: {:f} + {:f} = {:f} s'.format(delta_t[0], delta_t[1], sum(delta_t)))

    delta_t = []
    t_start = time()
    distance_matrix = pairwise_distances(X, metric="euclidean")
    reachability_graph = mutual_reachability(distance_matrix)
    delta_t.append(time() - t_start)
    mst4 = kruskal_with_filter(reachability_graph)
    delta_t.append(time() - t_start - delta_t[0])
    #mst4 = np.insert(mst4, 3, reachability_graph[mst4[:, 0].astype(int), mst4[:, 1].astype(int)], axis=1)
    # print('Tree 4: \n{}'.format(mst4))
    print('Cost 4: {:f}'.format(np.sum(mst4[:, 2])))
    print('Runtime 4: {:f} + {:f} = {:f} s'.format(delta_t[0], delta_t[1], sum(delta_t)))

    delta_t = []
    t_start = time()
    distance_matrix = pairwise_distances(X, metric="euclidean")
    reachability_graph = mutual_reachability(distance_matrix)
    delta_t.append(time() - t_start)
    mst5 = boruvka(reachability_graph)
    delta_t.append(time() - t_start - delta_t[0])
    #mst4 = np.insert(mst5, 3, reachability_graph[mst5[:, 0].astype(int), mst5[:, 1].astype(int)], axis=1)
    print('Tree 5: \n{}'.format(mst5))
    print('Cost 5: {:f}'.format(np.sum(mst5[:, 2])))
    print('Runtime 5: {:f} + {:f} = {:f} s'.format(delta_t[0], delta_t[1], sum(delta_t)))