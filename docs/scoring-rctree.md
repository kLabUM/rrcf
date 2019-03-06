# Anomaly score

The likelihood that a point is an outlier is measured by its collusive displacement (CoDisp):
if including a new point significantly changes the model complexity (i.e. bit depth),
then that point is more likely to be an outlier.

## Computing the anomaly score using the `codisp` method

The `codisp` method is used to compute the collusive displacement for a point. The `codisp` method
takes the index label of the point as an argument and returns the collusive displacement.

First, let's construct an `RCTree` with 100 normally-distributed 2-dimensional points.

```python
import numpy as np
import rrcf

# Seed tree with zero-mean, normally distributed data
X = np.random.randn(100,2)
tree = rrcf.RCTree(X)
```

Next, generate an inlier and outlier point, and insert them into the tree with the labels `inlier` and `outlier`:

```python
# Generate an inlier and outlier point
inlier = np.array([0, 0])
outlier = np.array([4, 4])

# Insert into tree
tree.insert_point(inlier, index='inlier')
tree.insert_point(outlier, index='outlier')
```

Finally, compute the anomaly score by calling the `codisp` method on each point's index label.

```python
tree.codisp('inlier')
>>> 1.75
```

```python
tree.codisp('outlier')
>>> 39.0
```

Note that the codisp of the outlier point is significantly larger.

## Computing `codisp` over a random forest

To make the `codisp` metric more robust, we can compute the `codisp` for each point over a forest (distribution) of randomly-constructed `RCTrees`. 

First, let's create a sample of 1000 points where the first 990 points are inliers distributed according to \\( x_i \sim \mathcal{N}([0, 0]^T, I_{2 \times 2})\\) and the last ten points are outliers with \\( x_i \sim \mathcal{N}([4, 4]^T, I_{2 \times 2})\\).


```python
import numpy as np
import pandas as pd
import rrcf

# Specify sample parameters
n = 1000
d = 2
num_outliers = 10

# Seed tree with zero-mean, normally distributed data
X = np.random.randn(n,d)
# Set the last ten points as outliers
X[-num_outliers:, :] += 4
```

To create a random forest, we simply create a list of `RCTrees`, with each `RCTree` constructed from a random sample of the input dataset. Let's create a random forest with 100 trees, each containing 128 points from the original sample.

```python
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
    trees = [rrcf.RCTree(X[ix], index_labels=ix)
             for ix in ixs]
    forest.extend(trees)
```

Finally, to determine outliers we compute the average `codisp` over all trees for each point in the original sample.

```python
# Compute average CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index
```

Now, let's see the average `codisp` for each set of points.

For the inlier points:

```python
avg_codisp[:-10].mean()
>>> 5.267
```

And for the outlier points:

```python
avg_codisp[-10:].mean()
>>> 40.681
```

Note that the outlier points have larger codisp. To classify the original points into inlier and outlier classes, we can perform a simple threshold test on the `codisp` result. For example:

```python
avg_codisp > avg_codisp.quantile(0.99)
```
