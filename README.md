# rrcf
Implementation of the *Robust Random Cut Forest Algorithm* for anomaly detection by Guha et al. (2016).

## Creating a random cut tree

```python
import numpy as np
import rrcf

# A random cut tree can be instantiated from a point set (n x d)
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


## Batch anomaly detection

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
