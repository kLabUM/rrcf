# Anomaly score

The likelihood that a point is an outlier is measured by its collusive displacement (CoDisp):
if including a new point significantly changes the model complexity (i.e. bit depth),
then that point is more likely to be an outlier.

## Computing the anomaly score using the `codisp` method

The `codisp` method is used to compute the collusive displacement for a point. The `codisp` method
takes the index label of the point as an argument and returns the collusive displacement.

First, let's construct an `RCTree` with 100 normally-distributed 2-dimensional points.

```python
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
