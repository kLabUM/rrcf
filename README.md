# rrcf 🌲🌲🌲
[![Build Status](https://travis-ci.org/kLabUM/rrcf.svg?branch=master)](https://travis-ci.org/kLabUM/rrcf) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Implementation of the *Robust Random Cut Forest Algorithm* for anomaly detection by [Guha et al. (2016)](http://proceedings.mlr.press/v48/guha16.pdf).

```
S. Guha, N. Mishra, G. Roy, & O. Schrijvers. Robust random cut forest based anomaly
detection on streams, in Proceedings of the 33rd International conference on machine
learning, New York, NY, 2016 (pp. 2712-2721).
```

## Robust random cut trees

A robust random cut tree is a binary search tree that can be used to detect outliers in a point set. Points located nearer to the root of the tree are more likely to be outliers.

### Creating the tree

```python
import numpy as np
import rrcf

# A (robust) random cut tree can be instantiated from a point set (n x d)
X = np.random.randn(100, 2)
tree = rrcf.RCTree(X)

# A random cut tree can also be instantiated with no points
tree = rrcf.RCTree()
```

### Inserting points

```python
tree = rrcf.RCTree()

for i in range(6):
    x = np.random.randn(2)
    tree.insert_point(x, index=i)
```

```
─+
 ├───+
 │   ├───+
 │   │   ├──(0)
 │   │   └───+
 │   │       ├──(5)
 │   │       └──(4)
 │   └───+
 │       ├──(2)
 │       └──(3)
 └──(1)
```

### Deleting points

```
tree.forget_point(2)
```

```
─+
 ├───+
 │   ├───+
 │   │   ├──(0)
 │   │   └───+
 │   │       ├──(5)
 │   │       └──(4)
 │   └──(3)
 └──(1)
```

## Anomaly score

The likelihood that a point is an outlier is measured by its collusive displacement (CoDisp): if including a new point significantly changes the model complexity (i.e. bit depth), then that point is more likely to be an outlier.

```python
# Seed tree with zero-mean, normally distributed data
X = np.random.randn(100,2)
tree = rrcf.RCTree(X)

# Generate an inlier and outlier point
inlier = np.array([0, 0])
outlier = np.array([4, 4])

# Insert into tree
tree.insert_point(inlier, index='inlier')
tree.insert_point(outlier, index='outlier')
```

```python
tree.codisp('inlier')
>>> 1.75
```

```python
tree.codisp('outlier')
>>> 39.0
```

## Batch anomaly detection

This example shows how a robust random cut forest can be used to detect outliers in a batch setting. Outliers correspond to large CoDisp.

```python
# Set parameters
np.random.seed(0)
n = 2010
num_trees = 40

# Generate data
X = np.zeros((n, 3))
X[:1000,0] = 5
X[1000:2000,0] = -5
X += 0.01*np.random.randn(*X.shape)

# Create random cut forest
forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree(X)
    forest.append(tree)
    
# Compute CoDisp (anomaly score)
avg_codisp = pd.Series(0.0, index=np.arange(n))
for tree in forest:
    codisp = pd.Series({k:tree.codisp(v) for k,v in tree.leaves.items()})
    avg_codisp += codisp
avg_codisp /= num_trees
```

![Image](https://github.com/kLabUM/rrcf/blob/master/resources/batch.png)

## Installation

To install:

```shell
git clone https://github.com/kLabUM/rrcf.git
```

Then:

```shell
cd rrcf
python setup.py install
```

Or:
```shell
cd rrcf
pip install .
```

Currently, only Python 3 is supported.
