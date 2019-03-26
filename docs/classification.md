# Using RRCF for classification

In this example, we use RRCF to classify data into one of two categories. The approach proceeds as follows:

- Create two forests, each one constructed from data from one of the known classes.
- Try inserting a new (unclassified) point into each forest. Assign the point to the class that produces the smaller CoDisp upon insertion.

The advantage of this approach compared to conventional classification algorithms is that it provides a metric of the likelihood that a point does not belong to either class (i.e. is an outlier from both classes).

## Test 1: Classification using two clusters

In this example, we will classify points into one of two normally-distributed clusters.

#### Generate data and run RRCF classification

```python
import numpy as np
import pandas as pd
import scipy.io
import rrcf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

# Set parameters
np.random.seed(0)
n = 256
d = 3
num_trees = 10

# Generate data
X_0 = np.zeros((n, d))
X_1 = np.zeros((n, d))
X_0[:,0] = -5
X_1[:,0] = 5
X_0 += 0.01*np.random.randn(*X_0.shape)
X_1 += 0.01*np.random.randn(*X_1.shape)

# Create two random cut forests
forest_0 = []
forest_1 = []
for _ in range(num_trees):
    tree_0 = rrcf.RCTree(X_0)
    tree_1 = rrcf.RCTree(X_1)
    forest_0.append(tree_0)
    forest_1.append(tree_1)
    
# Generate new points from two different classes
num_points = 500
offsets = np.random.choice([-5, 5], size=num_points)
labels = pd.Series({-5 : 0, 5: 1})[offsets].values
x = 0.01*np.random.randn(num_points, d)
x[:,0] += offsets

# Compute anomaly score for each new point
avg_codisp = np.zeros((num_points, 2))
for index in range(num_points):
    for tree_0, tree_1 in zip(forest_0, forest_1):
        tree_0.insert_point(x[index], index=n + index)
        tree_1.insert_point(x[index], index=n + index)
        avg_codisp[index, 0] += tree_0.codisp(n + index)
        avg_codisp[index, 1] += tree_1.codisp(n + index)
        tree_0.forget_point(n + index)
        tree_1.forget_point(n + index)

avg_codisp /= num_trees
```

#### Compute test error

```python
predictions = np.argmin(avg_codisp, axis=1)
test_error = 1 - ((predictions == labels).sum()/num_points)
print("Test error: {:.1f}%".format(100*test_error))
```

> ```
Test error: 0.0%
```

#### Plot classification results

```python
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_0[:,0], X_0[:,1], X_0[:,2], c='0.5', alpha=0.3,
           label='Training data')
ax.scatter(X_1[:,0], X_1[:,1], X_1[:,2], c='0.5', alpha=0.3)
ax.scatter(x[predictions == 0][:,0], x[predictions == 0][:,1],
           x[predictions == 0][:,2], c='b', label='Class 0')
ax.scatter(x[predictions == 1][:,0], x[predictions == 1][:,1],
           x[predictions == 1][:,2], c='r', label='Class 1')
plt.title('Classification results', size=14)
plt.legend(frameon=True)
plt.tight_layout()
```

![Classification 1](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/rrcf_classification_0.png)

## Test 2: Classification using nuclear particle data

In this example, we will classify points from a real-world nuclear particle energy dataset. This dataset is available in the `/resources` directory in the `rrcf` repo.

#### Load and plot original data

```python
# Load data
nuc = scipy.io.loadmat('nuclear.mat')
x = nuc['x'].astype(float).T
y = nuc['y'].astype(float).T
y = pd.Series({-1:0, 1:1})[y.ravel()].values

plt.scatter(x[y == 0][:,0], x[y == 0][:,1], c='b', alpha=0.3,
            label='Class 0')
plt.scatter(x[y == 1][:,0], x[y == 1][:,1], c='r', alpha=0.3,
            label='Class 1')
plt.title('Original labeled data', size=14)
plt.xlabel('Total energy')
plt.ylabel('Tail energy')
plt.legend(frameon=True)
plt.tight_layout()
```

![Classification 2](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/rrcf_classification_1.png)

#### Construct dual random forests

```python
# Set parameters
np.random.seed(0)
n = 2500
m = 100
d = 2
num_trees = 60

# Take random sample
X_0 = x[np.random.choice(np.flatnonzero(y.ravel() == 0), size=n)]
X_1 = x[np.random.choice(np.flatnonzero(y.ravel() == 1), size=n)]

# Create random cut forests
forest_0 = []
forest_1 = []
for _ in range(num_trees):
    tree_0 = rrcf.RCTree(X_0)
    tree_1 = rrcf.RCTree(X_1)
    forest_0.append(tree_0)
    forest_1.append(tree_1)
    
ix = np.random.randint(0, x.shape[0], size=m)
avg_codisp = np.zeros((m, d))

for index in range(m):
    for tree_0, tree_1 in zip(forest_0, forest_1):
        tree_0.insert_point(x[ix[index]], index=n + index)
        tree_1.insert_point(x[ix[index]], index=n + index)
        avg_codisp[index,0] += tree_0.codisp(n + index)
        avg_codisp[index,1] += tree_1.codisp(n + index)
        tree_0.forget_point(n + index)
        tree_1.forget_point(n + index)

avg_codisp /= num_trees
```

#### Compute test error

```python
predictions = np.argmin(avg_codisp, axis=1)
test_error = 1 - (predictions == y[ix]).sum() / m
print("Test error: {:.1f}%".format(100*test_error))
```

>```
Test error: 9.0%
```

#### Plot classification results

```python
plt.scatter(X_0[:,0], X_0[:,1], c='0.5', alpha=0.1)
plt.scatter(X_1[:,0], X_1[:,1], c='0.5', alpha=0.1, label='Training data')
plt.scatter(x[ix][predictions == 0][:,0], x[ix][predictions == 0][:,1],
            c='b', alpha=0.4, label='Class 0')
plt.scatter(x[ix][predictions == 1][:,0], x[ix][predictions == 1][:,1],
            c='r', alpha=0.4, label='Class 1')
plt.title('Classified points', size=14)
plt.xlabel('Total energy')
plt.ylabel('Tail energy')
plt.legend(frameon=True)
plt.tight_layout()
```

![Classification 3](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/rrcf_classification_2.png)


#### Classify all points in search space

```python
nn = 10000
num_trees = 10

# Create random cut forests
forest_0 = []
forest_1 = []
for _ in range(num_trees):
    tree_0 = rrcf.RCTree(X_0)
    tree_1 = rrcf.RCTree(X_1)
    forest_0.append(tree_0)
    forest_1.append(tree_1)
    
points = np.vstack(np.dstack(np.meshgrid(np.linspace(0, 8, 100),
                                         np.linspace(0, 1.4, 100))))
avg_codisp = np.zeros((nn, d))

for index in range(nn):
    for tree_0, tree_1 in zip(forest_0, forest_1):
        tree_0.insert_point(points[index], index=n + index)
        tree_1.insert_point(points[index], index=n + index)
        avg_codisp[index,0] += tree_0.codisp(n + index)
        avg_codisp[index,1] += tree_1.codisp(n + index)
        tree_0.forget_point(n + index)
        tree_1.forget_point(n + index)

avg_codisp /= num_trees
```

#### Plot decision boundary

```python
fig, ax = plt.subplots(figsize=(10,6))
plt.imshow(-np.log(avg_codisp[:,1] / avg_codisp[:,0]).reshape(100, 100),
           cmap='seismic', extent=(0, 8, 0, 1.4), origin='lower',
           aspect='auto')
plt.colorbar(label='Log ratio of Class 1 Codisp to Class 0 Codisp')
plt.grid('off')
plt.title('Decision regions', size=16)
plt.xlabel('Total energy')
plt.ylabel('Tail energy')
plt.tight_layout()
```

![Classification 4](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/rrcf_classification_3.png)


#### Plot outlier likelihood

```python
fig, ax = plt.subplots(figsize=(10,6))
plt.imshow(np.log(np.min(avg_codisp, axis=1)).reshape(100, 100),
           extent=(0, 8, 0, 1.4), origin='lower',
           aspect='auto', cmap='cubehelix_r')
plt.colorbar(label='$\log(\min(CoDisp(x^{(0)}), CoDisp(x^{(1)})))$')
plt.grid('off')
plt.title('Likelihood of belonging to neither class', size=14)
plt.xlabel('Total energy')
plt.ylabel('Tail energy')
plt.tight_layout()
```

![Classification 5](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/rrcf_classification_4.png)

