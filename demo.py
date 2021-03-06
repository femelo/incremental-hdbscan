#!/usr/bin/env python3

import argparse

import numpy as np
import sklearn.datasets
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from termcolor import cprint

import incremental_hdbscan
    
parser = argparse.ArgumentParser(description="""
Example of running incremental HDBSCAN.
This will plot points that are naturally clustered and added incrementally,
and then loop through all the hierarchical clusters recognized by the
algorithm.
.""")
parser.add_argument('--nitems', type=int, default=15000,
                    help="Number of items (default 200).")
parser.add_argument('--niters', type=int, default=100,
                    help="Clusters are shown in NITERS stage while being "
                    "added incrementally (default 4).")
parser.add_argument('--centers', type=int, default=5,
                    help="Number of centers for the clusters generated "
                    "(default 5).")
args = parser.parse_args()

np.random.seed(5)
data, labels = sklearn.datasets.make_blobs(args.nitems,
                                           centers=args.centers)
max_label = max(labels)
x, y = data[:, 0], data[:, 1]
r = np.sqrt(x ** 2 + y ** 2)
    
params = {
    'time_step': 0,
    'window_start': 0,
    'time_window': 5,
    'metric': "euclidean",
    'min_cluster_size': 5,
    'cluster_selection_method': "eom",
    'allow_single_cluster': False,
    'match_reference_implementation': False,
    'cluster_selection_epsilon': 0.0,
    'max_cluster_size': 0 
}

plt.ion()
fig = plt.figure(figsize=(25, 25))
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
ax1 = fig.add_subplot(121)

ax1.set_xlim(-1.1*max(r), 1.1*max(r))
ax1.set_ylim(-1.1*max(r), 1.1*max(r))
ax1.margins(0.05)
ax1.set_aspect('equal')
ax1.set_title('Ground truth', fontsize=18)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
genericPlot1 = ax1.scatter([], [])

ax2 = fig.add_subplot(122)
ax2.set_xlim(-1.1*max(r), 1.1*max(r))
ax2.set_ylim(-1.1*max(r), 1.1*max(r))
ax2.margins(0.05)
ax2.set_aspect('equal')
ax2.set_title('I-HDBSCAN clusters', fontsize=18)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
genericPlot2 = ax2.scatter([], [])
genericPlot3 = ax2.scatter([], [], marker='x', s=20)

plt.subplots_adjust(wspace=0, hspace=0)

palette = sns.color_palette()

k = 0
for points in np.split(data, args.niters):
    k += 1
    cprint('Step {}'.format(k), 'cyan')
    # Rotate points
    delta_angle = 2.0 * (np.pi) * k / args.niters
    rotation_matrix = np.array([
        [np.cos(delta_angle), -np.sin(delta_angle)],
        [np.sin(delta_angle),  np.cos(delta_angle)]])
    points = points.dot(rotation_matrix)
    fig.suptitle('Time step = {:d}'.format(k), fontsize=18)
    clusters, lbls, probs, stabs, ctree, slt = incremental_hdbscan.fit(points, params)
    n = len(list([c for c in clusters.keys() if c != -1]))
    nknown = params['data_id']

    # xknown, yknown, labels_known = x[nknown], y[nknown], labels[nknown]
    xknown, yknown, labels_known = params['raw_data'][:, 0], params['raw_data'][:, 1], labels[nknown]
    color = ['rgbcmyk'[l % 7] for l in labels_known]
    genericPlot1.set_offsets(np.c_[xknown, yknown])
    genericPlot1.set_color(color)

    cluster_colors = [sns.desaturate(palette[col], sat)
        if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
        zip(lbls, probs)]

    genericPlot2.set_offsets(np.c_[xknown, yknown])
    genericPlot2.set_color(cluster_colors)
    genericPlot3.set_offsets(params['centroids'][:n, :])
    genericPlot3.set_color([palette[l] for l in range(n)])
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.savefig("figures/I-HDBSCAN-{:03d}.png".format(k), bbox_inches="tight")
