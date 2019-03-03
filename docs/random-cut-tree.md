# Robust random cut trees

The (robust) random cut tree is the core data structure of the `rrcf` library and is represented by the `rrcf.RCTree` class.
A (robust) random cut tree is a binary search tree that can be used to detect outliers in a point set.
Points located nearer to the root of the tree are more likely to be outliers.

## Creating an `RCTree`

### Instantiating an `RCTree` from existing data

A (robust) random cut tree can be instantiated from a point set \\((n \times d)\\),
where \\(n\\) is the number of points and \\(d\\) is the dimension of each point.

```python
import numpy as np
import rrcf

X = np.random.randn(100, 2)
tree = rrcf.RCTree(X)
```

The resulting tree can be inspected simply by calling the instance object:

```
>>> tree
>>>
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

### Instantiating an empty `RCTree`

A random cut tree can also be instantiated with no points

```python
tree = rrcf.RCTree()
```

## Tree attributes

The tree contains the following attributes:

`root`
: (Branch or Leaf instance)
: Pointer to root of tree.

`leaves`
: (dict)
: Dict containing pointers to all leaves in tree.

`ndim`
: (int)
: Dimension of points in the tree

## Branches and Leaves

The tree is composed of two types of nodes: branches and leaves.

- Branches are nodes with two children and at most one parent.
- Leaves are nodes with no children and at most one parent.

Note that the root node will have no parent node.

### Branches

A branch is a node that represents a partition (cut) in the point set and is represented by the `rrcf.Branch` class.
The branch class contains the following attributes.

`q`
: (int)
: Dimension along which cut is performed.

`p`
: (int)
: Value along which cut is performed.

`l`
: (Branch or Leaf instance)
: Pointer to left child.

`r`
: (Branch or Leaf instance)
: Pointer to right child

`u`
: (Branch or Leaf instance or `None`)
: Pointer to parent

`n`
: (int)
: Number of leaves under branch

`b`
: (numpy ndarray (2 x d))
: Bounding box of points under branch

### Leaf

A leaf is a node that represents an individual point in the point set and is represented by the `rrcf.Leaf` class.
The leaf class contains the following attributes.

`i`
: (int)
: Index of leaf (user-specified)

`d`
: (int)
: Depth of leaf in tree

`u`
: (Branch instance)
: Pointer to parent

`x`:
: (numpy ndarray (1 x d))
: Pointer to original data point

`n`
: (int)
: Number of points in leaf (1 if no duplicates)

`b`
: (numpy ndarray (1 x d))
: Bounding box of point (equivalent to original point)
