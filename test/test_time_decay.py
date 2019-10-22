# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.stats
import rrcf
from matplotlib import pyplot as plt

# Read data
abs_path = os.path.abspath(__file__)
data_paths = abs_path.strip().split("/")[:-1]
data_paths.extend(["..", "resources", "nyc_taxi.csv"])
data_path = os.path.join(*data_paths)
if not data_path.startswith("/"):
    data_path = "/" + data_path

taxi = pd.read_csv(data_path, index_col=0)
taxi.index = pd.to_datetime(taxi.index)
data = taxi['value'].astype(float).values
n = data.shape[0]

# Set tree parameters
num_trees = 200
shingle_size = 48
tree_size = 1000

# Use the "shingle" generator to create rolling window
points = rrcf.shingle(data, size=shingle_size)
points = np.vstack([point for point in points])
num_points = points.shape[0]

# Chunk data into segments
num_chunks = 40
chunk_size = num_points // num_chunks
chunks = (chunk * chunk_size for chunk in range(num_chunks))

forest = []
pdfs = []
for chunk in chunks:
    # Create probability density function for sample
    g = scipy.stats.norm(loc=chunk, scale=400).pdf(np.linspace(0, num_points, num_points))
    g *= (1 / g.sum())
    # Sample data with probability density function defined by g
    ixs = np.random.choice(num_points, size=(num_points // tree_size, tree_size),
                           replace=False, p=g)
    # Create trees from sampled points
    trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)
    # Store pdf for plotting purposes
    pdfs.append(g)

# Compute CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(num_points))
index = np.zeros(num_points)
for tree in forest:
    codisp = pd.Series({leaf: tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index
avg_codisp.index = taxi.iloc[(shingle_size - 1):].index


# Plot results
fig, ax = plt.subplots(3, figsize=(10, 9))
(taxi['value'] / 1000).plot(ax=ax[0], color='0.5', alpha=0.8)
avg_codisp.plot(ax=ax[2], color='#E8685D', alpha=0.8, label='RRCF')
for pdf in pdfs[1:]:
    pd.Series(pdf, index=avg_codisp.index).plot(ax=ax[1], c='0.6')
ax[0].set_title('Taxi passenger timeseries')
ax[1].set_title('Rolling probability density functions')
ax[2].set_title('Anomaly score')
ax[0].set_ylabel('Thousands of passengers')
ax[1].set_ylabel('p(x)')
ax[2].set_ylabel('CoDisp')
ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[2].set_xlabel('')
plt.tight_layout()
plt.savefig('decay_example.png')
