# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import rrcf

import matplotlib.pyplot as plt


# Specify sample parameters
n = 1000
d = 2
num_outliers = 10

# Seed tree with zero-mean, normally distributed data
X = np.random.randn(n,d)
# Set the last ten points as outliers
X[-num_outliers:, :] += 4

# Construct forest
forest = []

# Specify forest parameters
num_trees = 100
tree_size = 128
sample_size_range = (n // tree_size, tree_size)

while len(forest) < num_trees:
    # Select random subsets of points uniformly from point set
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    # Add sampled trees to forest
    trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)

# Compute average CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index

inlier_point = avg_codisp[:-10].mean()
outlier_point = avg_codisp[-10:].mean()
print(inlier_point, outlier_point)
anomaly_score = np.percentile(avg_codisp, 97)
anomaly_points = X[avg_codisp >= anomaly_score]

plt.scatter(X[:-num_outliers, 0], X[:-num_outliers, 1], c='g', label='inlier')
plt.scatter(X[-num_outliers:, 0], X[-num_outliers:, 1], c='b', label='outlier')
plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='r', marker='o', label='mark')
plt.legend()
plt.show()