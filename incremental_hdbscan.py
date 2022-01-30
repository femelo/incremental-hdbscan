import sys
import numpy as np
from sklearn.metrics import pairwise_distances
from linkage import boruvka
from sklearn.neighbors import KDTree
from munkres import Munkres
import collections
import itertools
from _hdbscan_linkage import label, boruvka
from _hdbscan_tree import (
    condense_tree,
    compute_stability,
    get_clusters,
)
from linkage import boruvka

DBL_MAX = sys.float_info.max

def fit(X, params):
    first_iteration = 'raw_data' not in params or len(params['raw_data']) == 0 or params['raw_data'] is None

    # 1. Remove old elements
    remove_data(params, first_iteration)

    # 2. Update data
    update_data(X, params, first_iteration)

    # 3. Update distance matrix
    update_distance_matrix(X, params, first_iteration)

    # 4. Update mutual reachability
    update_mutual_reachability(params)

    # 5. Update edges
    update_forest(params, first_iteration)

    # 6. Boruvka step
    incremental_boruvka_preprocessing(params, first_iteration)

    # 7. Apply boruvka
    compute_mst(params)

    # 8. Compute single linkage tree
    compute_slt(params)

    # 9. Generate clusters
    generate_clusters(params)

    # 10. Return results
    return (
        params['clusters'],
        params['labels'], 
        params['probabilities'], 
        params['stabilities'], 
        params['condensed_tree'], 
        params['single_linkage_tree'])

def find_edges_with_vertex(u, remaining_edges):
    edges = []
    for e in remaining_edges:
        if u in e:
            edges.append(e)
    return edges

def traverse_tree(u, remaining_edges):
    edges = find_edges_with_vertex(u, remaining_edges)
    if len(edges) == 0:
        return []
    visited_nodes = []
    for e in edges:
        other_edges = remaining_edges.copy()
        other_edges.remove(e)
        v = e[1] if u == e[0] else e[0]
        visited_nodes.append(v)
        visited_nodes += traverse_tree(v, other_edges)
    return visited_nodes

def is_vertex_disconnected(tree, w):
    visited_nodes = traverse_tree(w, tree['E'])
    if set(tree['V']) - set([w]) - set(visited_nodes):
        return True
    return False

def is_graph_disconnected(tree):
    for w in tree['V']:
        if is_vertex_disconnected(tree, w):
            return True
    return False

def split_tree(forest, tree_id, connection_check=False):
    tree = forest[tree_id]
    subtree = {'V': [tree['E'][-1][0], tree['E'][-1][1]], 'E': [], 'W': []}
    while True:
        found_match = False
        for w in subtree['V']:
            for idx, e in enumerate(tree['E']):
                if w in e:
                    found_match = True
                    break
            if found_match:
                break
        if found_match:
            x, y = e
            subtree['E'].append(tree['E'].pop(idx))
            subtree['W'].append(tree['W'].pop(idx))
            if not x in subtree['V']:
                subtree['V'].append(x)
            if not y in subtree['V']:
                subtree['V'].append(y)
            if x in tree['V']:
                tree['V'].remove(x)
            if y in tree['V']:
                tree['V'].remove(y)
        else:
            # Add new graph
            new_tree_id = max(forest.keys()) + 1
            forest[new_tree_id] = subtree
            if not connection_check or not is_graph_disconnected(tree):
                break
            else:
                subtree = {'V': [tree['E'][-1][0], tree['E'][-1][1]], 'E': [], 'W': []}
    return

def remove_data(params, first_iteration=False):
    if first_iteration:
        params['time_step'] = 1
        params['window_start'] = 1
        return
    k = params['time_step']
    k_0 = params['window_start']
    w = params['time_window']
    if k - k_0 + 1 == w:
        # Remove old data
        raw_data = params['raw_data']
        data_id = params['data_id']
        data_time = params['data_time']
        distance_matrix = params['distance_matrix']
        mutual_reachability_matrix = params['mutual_reachability_matrix']
        cheapest_edges = params['cheapest_edges']
        forest = params['forest']
        labels = params['labels']

        indexes = np.where(data_time > k_0)[0]
        params['raw_data'] = raw_data[indexes, :]
        params['data_id'] = data_id[indexes]
        params['data_time'] = data_time[indexes]
        params['distance_matrix'] = distance_matrix[np.ix_(indexes, indexes)]
        params['mutual_reachability_matrix'] = mutual_reachability_matrix[np.ix_(indexes, indexes)]
        # Mark destination vertices as -1 instead of removing because the source vertices are defined by position in the array
        # If some vertices were removed from the array, the source vertices positions woudn't be contiguous anymore
        params['cheapest_edges'] = np.array([v if v in data_id[indexes] else -1 for v in cheapest_edges[indexes]], dtype=int)
        tree_ids = list(forest.keys())
        for tree_id in tree_ids:
            new_edges = []
            new_weights = []
            for i, e in enumerate(forest[tree_id]['E']):
                u, v = e
                if u in data_id[indexes] and v in data_id[indexes]:
                    new_edges.append([u, v])
                    new_weights.append(forest[tree_id]['W'][i])
            if len(new_edges) == 0:
                del forest[tree_id]
                continue
            removed_edges = len(new_edges) < len(forest[tree_id]['E'])
            forest[tree_id]['E'] = new_edges
            forest[tree_id]['W'] = new_weights
            l_vertices, r_vertices = zip(*forest[tree_id]['E'])
            forest[tree_id]['V'] = sorted(list(set(l_vertices + r_vertices)))
            if removed_edges and is_graph_disconnected(forest[tree_id]):
                split_tree(forest, tree_id, connection_check=True)

        params['labels'] = labels[indexes]
        params['window_start'] = k_0 + 1
    # Iterate
    params['time_step'] = k + 1

def update_data(X, params, first_iteration=False):
    if first_iteration:
        params['raw_data'] = X
        params['data_id'] = np.arange(X.shape[0])
        params['data_time'] = params['time_step'] * np.ones(X.shape[0], dtype=int)
        return
    params['raw_data'] = np.concatenate((params['raw_data'], X), axis=0)
    if params['data_id'].shape[0] > 0:
        start_id = max(params['data_id']) + 1
    else:
        start_id = 0
    new_data_id = np.arange(start_id, start_id + X.shape[0])
    params['data_id'] = np.concatenate((params['data_id'], new_data_id))
    new_time_step = params['time_step'] * np.ones(X.shape[0], dtype=int)
    params['data_time'] = np.concatenate((params['data_time'], new_time_step))

def update_distance_matrix(X, params, first_iteration=False):
    innov_distance_matrix = pairwise_distances(X, metric=params['metric'])
    if first_iteration:
        params['distance_matrix'] = innov_distance_matrix
        return
    n = params['raw_data'].shape[0] - X.shape[0]
    if n > 0:
        cross_distance_matrix = pairwise_distances(params['raw_data'][:n, :], X, metric=params['metric'])
        augmented_distance_matrix = np.concatenate(
            (np.concatenate((params['distance_matrix'], cross_distance_matrix), axis=1),
            np.concatenate((cross_distance_matrix.T, innov_distance_matrix), axis=1)), axis=0)
    else:
        augmented_distance_matrix = innov_distance_matrix
    params['distance_matrix'] = augmented_distance_matrix

def update_mutual_reachability(params):
    tree = KDTree(params['raw_data'], metric=params['metric'], leaf_size=40)
    core_distances = tree.query(
        params['raw_data'], k=5 + 1, dualtree=True, breadth_first=True)[0][:,-1].copy(order="C")

    stage1 = np.where(core_distances > params['distance_matrix'],
                      core_distances, params['distance_matrix'])
    params['mutual_reachability_matrix'] = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T

def update_forest(params, first_iteration=False):
    # Indexes of the cheapest destination nodes
    n = params['mutual_reachability_matrix'].shape[0]
    local_matrix = params['mutual_reachability_matrix'].copy()
    for i in range(n):
        local_matrix[i, i] = DBL_MAX
    cheapest_destination = np.argmin(local_matrix, axis=1)

    # Create id maps
    indexes = np.arange(params['data_id'].shape[0])
    id_map = dict(zip(indexes, params['data_id']))
    id_map_inv = dict(zip(params['data_id'], indexes))
    params['id_map'] = id_map
    params['id_map_inverse'] = id_map_inv

    if first_iteration:
        params['cheapest_edges'] = cheapest_destination
        return

    # Update local ids for cheapest destination
    for i in range(n):
        # Now position 0 in cheapest_destination means params['data_id'][0], therefore
        # in order to check cheapest_destination against the old cheapest_destination array
        # one must map all vertices in the new cheapest_destination from as i -> index_map[i]
        cheapest_destination[i] = id_map[cheapest_destination[i]]
    
    # Find edges that changed
    cheapest_destination_prv = params['cheapest_edges']
    mask = cheapest_destination[:cheapest_destination_prv.shape[0]] != cheapest_destination_prv
    heavy_edges = []
    for i in range(cheapest_destination_prv.shape[0]):
        if not mask[i]:
            continue
        u = id_map[i]
        v = cheapest_destination_prv[i]
        if v == -1:
            # Vertex was removed
            continue
        if u > v:
            u, v = v, u
        if not [u, v] in heavy_edges and not [v, u] in heavy_edges: 
            heavy_edges.append([u, v])
    
    # Update local ids and remove heavy edges for the forests
    # By removing those edges the trees may become disconnected, which shall be corrected by the Boruvka step
    forest = params['forest']
    for u, v in heavy_edges:
        for tree_id in forest.keys():
            idx = 0
            if [u, v] in forest[tree_id]['E']:
                idx = forest[tree_id]['E'].index([u, v])
            elif [v, u] in forest[tree_id]['E']:
                idx = forest[tree_id]['E'].index([v, u])
            else:
                continue
            del forest[tree_id]['E'][idx]
            del forest[tree_id]['W'][idx]
            if len(forest[tree_id]['E']) == 0:
                del forest[tree_id]
                break
            l_vertices, r_vertices = zip(*forest[tree_id]['E'])
            all_vertices = l_vertices + r_vertices
            if u in all_vertices and v in all_vertices:
                # Need to split into two trees because the removed edge connected them
                split_tree(forest, tree_id)
            elif u in all_vertices and not v in all_vertices:
                # Leads to a leaf ended in v
                forest[tree_id]['V'].remove(v)
            elif not u in all_vertices and v in all_vertices:
                # Leads to a leaf ended in u
                forest[tree_id]['V'].remove(u)
            else:
                pass
            break

    # Replace all weights in the forest
    if not id_map[0] == 0:
        for tree_id in forest:
            forest[tree_id]['W'] = [local_matrix[id_map_inv[u], id_map_inv[v]] for u, v in forest[tree_id]['E']]

    params['cheapest_edges'] = cheapest_destination
    return

def incremental_boruvka_preprocessing(params, first_iteration=False):
    # Get all edges
    F = []
    for i in range(params['cheapest_edges'].shape[0]):
        u = params['id_map'][i]
        v = params['cheapest_edges'][i]
        if u > v:
            u, v = v, u
        if not [u, v] in F and not [v, u] in F: 
            F.append([u, v])
    
    # Compose map from vertex to subgraph
    vertex_to_subgraph = {}
    if first_iteration:
        forest = {}
    else:
        forest = params['forest']
        for tree_id in forest:
            vertex_to_tree = dict(zip(forest[tree_id]['V'], len(forest[tree_id]['V']) * [tree_id]))
            vertex_to_subgraph.update(vertex_to_tree)
            for u, v in forest[tree_id]['E']:
                if [u, v] in F:
                    F.remove([u, v])
                if [v, u] in F:
                    F.remove([v, u])

    # Add first edge
    cost_matrix = params['mutual_reachability_matrix']
    id_map_inv = params['id_map_inverse']
    G = forest
    if first_iteration or len(G.keys()) == 0:
        e = F.pop(0)
        u, v = e
        if u > v:
            u, v = v, u
        G[0] = {'V': [u, v], 'E': [e], 'W': [cost_matrix[id_map_inv[u], id_map_inv[v]]]}
        vertex_to_subgraph[u] = 0
        vertex_to_subgraph[v] = 0
    while len(F) > 0:
        u, v = F.pop(0)
        if u in vertex_to_subgraph and not v in vertex_to_subgraph:
            vertex_to_subgraph[v] = vertex_to_subgraph[u]
            G[vertex_to_subgraph[v]]['V'].append(v)
            G[vertex_to_subgraph[v]]['E'].append([u, v])
            G[vertex_to_subgraph[v]]['W'].append(cost_matrix[id_map_inv[u], id_map_inv[v]])
        elif not u in vertex_to_subgraph and v in vertex_to_subgraph:
            vertex_to_subgraph[u] = vertex_to_subgraph[v]
            G[vertex_to_subgraph[u]]['V'].append(u)
            G[vertex_to_subgraph[u]]['E'].append([u, v])
            G[vertex_to_subgraph[u]]['W'].append(cost_matrix[id_map_inv[u], id_map_inv[v]])
        elif u in vertex_to_subgraph and v in vertex_to_subgraph:
            # Need merging
            if len(G[vertex_to_subgraph[u]]['V']) < len(G[vertex_to_subgraph[v]]['V']):
                u, v = v, u
            subgraph = G[vertex_to_subgraph[v]]
            G[vertex_to_subgraph[u]]['V'] += subgraph['V']
            e = [u, v] if u < v else [v, u]
            G[vertex_to_subgraph[u]]['E'] += [e] + subgraph['E']
            G[vertex_to_subgraph[u]]['W'] += [cost_matrix[id_map_inv[u], id_map_inv[v]]] + subgraph['W']
            del G[vertex_to_subgraph[v]]
            for w in subgraph['V']:
                vertex_to_subgraph[w] = vertex_to_subgraph[u]
        else:
            # New graph
            i = max(G.keys()) + 1
            G[i] = {'V': [u, v], 'E': [[u, v]], 'W': [cost_matrix[id_map_inv[u], id_map_inv[v]]]}
            vertex_to_subgraph[u] = i
            vertex_to_subgraph[v] = i
    # Rename graphs
    subgraphs = G.values()
    G = dict(zip(list(range(len(subgraphs))), subgraphs))
    params['forest'] = G

def compute_mst(params):
    cost_matrix = params['mutual_reachability_matrix']
    id_map_inv = params['id_map_inverse']
    G = params['forest']
    # Add edges to the mst
    E = []
    for i_g in G:
        for e in G[i_g]['E']:
            u, v = e
            E.append([u, v, cost_matrix[id_map_inv[u], id_map_inv[v]]])
    mst = np.array(E)
    # If just one graph, then it is the MST
    n_g = len(G)
    if n_g == 1:
        params['minimum_spanning_tree'] = mst
        return

    contracted_edges = {}
    distance_sub_matrix = np.infty * np.ones((n_g, n_g))
    for i_g1, i_g2 in itertools.combinations(list(G.keys()), 2):
        V_g1 = G[i_g1]['V']
        V_g2 = G[i_g2]['V']
        V_g1_idx = [id_map_inv[u] for u in V_g1]
        V_g2_idx = [id_map_inv[v] for v in V_g2]
        sub_matrix = cost_matrix[np.ix_(V_g1_idx, V_g2_idx)]
        idx = np.argmin(sub_matrix.ravel())
        i, j = np.unravel_index(idx, sub_matrix.shape)
        contracted_edges[(i_g1, i_g2)] = [V_g1[i], V_g2[j]]
        distance_sub_matrix[i_g1, i_g2] = sub_matrix[i, j]
        distance_sub_matrix[i_g2, i_g1] = sub_matrix[i, j]

    # If just two graphs, it just link them and return
    if n_g == 2:
        mst = np.vstack((mst, contracted_edges[(0, 1)] + [distance_sub_matrix[0, 1]]))
        params['minimum_spanning_tree'] = mst
        return

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
    params['minimum_spanning_tree'] = mst
    return

def compute_slt(params):
    # Sort edges of the min_spanning_tree by weight
    mst = params['minimum_spanning_tree'][np.argsort(params['minimum_spanning_tree'].T[2]), :]

    # Reconvert vertices to indexes (the same as using id_map_inv but faster)
    mst[:, :2] = (mst[:, :2] - params['data_id'][0])

    # Convert edge list into standard hierarchical clustering format
    slt = label(mst)
    params['single_linkage_tree'] = slt

def generate_clusters(params):
    labels, probabilities, stabilities, condensed_tree, single_linkage_tree = \
        tree_to_labels(
            params['single_linkage_tree'],
            params['min_cluster_size'],
            params['cluster_selection_method'],
            params['allow_single_cluster'],
            params['match_reference_implementation'],
            params['cluster_selection_epsilon'],
            params['max_cluster_size']
        )

    clusters = {}
    for l in set(labels):
        clusters[l] = np.where(labels == l)[0]
    params['clusters'] = clusters
    # Trick with labels
    params['labels'] = labels
    params['probabilities'] = probabilities
    params['stabilities'] = stabilities
    params['condensed_tree'] = condensed_tree
    reassign_labels(params)

    return

def reassign_labels(params):
    clusters = params['clusters']
    labels = params['labels']
    # Compute new centroids
    n = sum([1 for c in clusters.keys() if c > -1])
    m = params['raw_data'].shape[1]
    centroids = np.zeros((n, m))
    for c, indexes in clusters.items():
        if c > -1:
            centroids[c, :] = np.mean(params['raw_data'][indexes, :], axis=0)
    if not 'centroids' in params or params['centroids'] is None or params['centroids'].shape[0] == 0:
        params['centroids'] = centroids
    else:
        centroids_prv = params['centroids']
        n_prv = centroids_prv.shape[0]
        current_indexes = list(range(centroids.shape[0]))
        centroid_distance_matrix = pairwise_distances(centroids, centroids_prv)
        assignment_algorithm = Munkres()
        if n > n_prv:
            solution_transposed = assignment_algorithm.compute(centroid_distance_matrix.T)
            cols, rows = zip(*solution_transposed)
            solution = list(zip(rows, cols))
        else:
            solution = assignment_algorithm.compute(centroid_distance_matrix)
            rows, cols = zip(*solution)
        remaining_rows = list(set(current_indexes) - set(rows))
        remaining_cols = list(set(current_indexes) - set(cols))
        for i, l in enumerate(remaining_rows):
            solution.append((l, remaining_cols[i]))
        needs_reassignment = False
        for i, j in solution:
            if i != j:
                needs_reassignment = True
                break
        if needs_reassignment:
            assignment_map = dict(solution)
            labels = np.array([assignment_map[l] if l > -1 else -1 for l in labels], dtype=int)
            clusters = dict([(assignment_map[l], indexes) if l > -1 else (-1, indexes) for l in clusters])
            inverted_map = dict([(value, key) for key, value in assignment_map.items()])
            if n < n_prv:
                centroids_prv[[assignment_map[l] for l in current_indexes], :] = centroids
                centroids = centroids_prv
            else:
                centroids = centroids[[inverted_map[l] for l in current_indexes], :]

        # Save new centroids
        params['centroids'] = centroids

    # Trick with labels
    params['labels'] = labels

def tree_to_labels(
    single_linkage_tree,
    min_cluster_size=10,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    match_reference_implementation=False,
    cluster_selection_epsilon=0.0,
    max_cluster_size=0,
):
    """Converts a pretrained tree and cluster size into a
    set of labels and probabilities.
    """
    condensed_tree = condense_tree(single_linkage_tree, min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    labels, probabilities, stabilities = get_clusters(
        condensed_tree,
        stability_dict,
        cluster_selection_method,
        allow_single_cluster,
        match_reference_implementation,
        cluster_selection_epsilon,
        max_cluster_size,
    )

    return (labels, probabilities, stabilities, condensed_tree, single_linkage_tree)