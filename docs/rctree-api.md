# RCTree API documentation

This section enumerates all the methods of the `RCTree` class

## Inserting and deleting points

<b>`insert_point`</b>`(point, index, tolerance=None)`
> Inserts a point into the tree, creating a new leaf with given index

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `point`     | numpy ndarray (1 x d) | Data point |
| `index`       | any hashable type | Identifier for new leaf in tree |
| `tolerance`    | float      | Tolerance for determining duplicate points |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `leaf`     | `Leaf` instance | New leaf in tree |

*Example:*

```python
>>> tree = rrcf.RCTree()

# Insert a point
>>> x = np.random.randn(2)
>>> tree.insert_point(x, index=0)

Leaf(0)
```

---

<b>`forget_point`</b>`(index)`
> Delete leaf from tree

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `index`       | any hashable type | Index of leaf in tree |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `leaf`     | `Leaf` instance | Deleted leaf |

*Example:*

```python
>>> tree = rrcf.RCTree()

# Insert a point
>>> x = np.random.randn(2)
>>> tree.insert_point(x, index=0)

# Forget point
>>> tree.forget_point(0)

Leaf(0)
```

## Getting tree info

<b>`query`</b>`(point, node=None)`
> Search for leaf nearest to point

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `point`     | numpy ndarray (1 x d) | Data point |
| `node`    | `Branch` instance      | Node to begin traversal (defaults to root) |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `leaf`     | `Leaf` instance | Leaf nearest to queried point in tree |

*Example:*

```python
# Create RCTree
>>> X = np.random.randn(10, 2)
>>> tree = rrcf.RCTree(X)

# Insert new point
>>> new_point = np.array([4, 4])
>>> tree.insert_point(new_point, index=10)

# Query tree for point with added noise
>>> tree.query(new_point + 1e-5)

Leaf(10)
```

---

<b>`get_bbox`</b>`(branch=None)`
> Compute bounding box of all points underneath a given branch.

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `branch`    | `Branch` instance      | Branch to begin traversal (defaults to root) |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `bbox`     | numpy ndarray (2 x d) | Bounding box of all points underneath branch |

*Example:*

```python
>>> X = np.random.randn(10, 3)
>>> tree = rrcf.RCTree(X)
>>> tree.get_bbox()

array([[-0.8600458 , -1.69756215, -1.16659065],
       [ 2.48455863,  1.02869042,  1.09414144]])
```

---

<b>`find_duplicate`</b>`(point, tolerance=None)`
> If point is a duplicate of existing point in the tree, return the leaf containing the point, else return None.

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `point`     | numpy ndarray (1 x d) | Point to query in tree |
| `tolerance`    | float      | Tolerance for determining whether or not point is a duplicate |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `duplicate` | `Leaf` instance or `None` |  If point is a duplicate, returns the leaf containing the point. If point is not a duplicate, return None. |

*Example:*

```python
# Create RCTree
>>> X = np.random.randn(10, 2)
>>> tree = rrcf.RCTree(X)

# Insert new point
>>> new_point = np.array([4, 4])
>>> tree.insert_point(new_point, index=10)

# Search for duplicates
>>> tree.find_duplicate((3, 3))

>>> tree.find_duplicate((4, 4))

Leaf(10)
```

## Anomaly scoring

<b>`codisp`</b>`(leaf)`
> Compute collusive displacement at leaf.

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `leaf`    | Any hashable type or `Leaf` instance      | Index of leaf or `Leaf` instance |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `codisplacement`     | float | Collusive displacement if leaf is removed |

*Example:*

```python
# Create RCTree
>>> X = np.random.randn(100, 2)
>>> tree = rrcf.RCTree(X)
>>> new_point = np.array([4, 4])
>>> tree.insert_point(new_point, index=100)

# Compute collusive displacement
>>> tree.codisp(100)

31.667
```

---

<b>`codisp`</b>`(leaf)`
> Compute displacement at leaf.

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `leaf`    | Any hashable type or `Leaf` instance      | Index of leaf or `Leaf` instance |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `displacement`     | float | Displacement if leaf is removed |

*Example:*

```python
# Create RCTree
>>> X = np.random.randn(100, 2)
>>> tree = rrcf.RCTree(X)
>>> new_point = np.array([4, 4])
>>> tree.insert_point(new_point, index=100)

# Compute displacement
>>> tree.disp(100)

12
```

## Leaf and Branch operations

<b>`map_leaves`</b>`(node, op=(lambda x: None), *args, **kwargs)`
> Traverse tree recursively, calling operation given by op on leaves

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `node`      | `Leaf` or `Branch` instance | Node in RCTree |
| `op`       | function | Function to call on each leaf (defaults to no-op) |
| `*args`    | any      | Positional arguments to `op` |
| `**kwargs` | any      | Keyword arguments to `op` |

*Returns:*

`None`

*Example:*

```python
# Use map_leaves to print leaves in postorder

>>> X = np.random.randn(10, 2)
>>> tree = RCTree(X)
>>> tree.map_leaves(tree.root, op=print)

Leaf(5)
Leaf(9)
Leaf(4)
Leaf(0)
Leaf(6)
Leaf(2)
Leaf(3)
Leaf(7)
Leaf(1)
Leaf(8)
```

---

<b>`map_branches`</b>`(node, op=(lambda x: None), *args, **kwargs)`
> Traverse tree recursively, calling operation given by op on branches

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `node`      | `Leaf` or `Branch` instance | Node in RCTree |
| `op`       | function | Function to call on each leaf (defaults to no-op) |
| `*args`    | any      | Positional arguments to `op` |
| `**kwargs` | any      | Keyword arguments to `op` |

*Returns:*

`None`

*Example:*

```python
# Use map_branches to collect all branches in a list

>>> X = np.random.randn(10, 2)
>>> tree = RCTree(X)
>>> branches = []
>>> tree.map_branches(tree.root, op=(lambda x, stack: stack.append(x)),
                      stack=branches)
>>> branches

[Branch(q=0, p=-0.53),
 Branch(q=0, p=-0.35),
 Branch(q=1, p=-0.67),
 Branch(q=0, p=-0.15),
 Branch(q=0, p=0.23),
 Branch(q=1, p=0.29),
 Branch(q=1, p=1.31),
 Branch(q=0, p=0.62),
 Branch(q=1, p=0.86)]
```
