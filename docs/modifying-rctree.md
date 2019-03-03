# Inserting points

Trees can be dynamically updated in response to new data by inserting points.

Here, we create an empty `RCTree` and insert 6 new points using the `insert_point` method (where each point is a 2-dimensional Gaussian-distributed random vector).

Note that for each point, we supply a unique index label `i`. In this case `i` is an integer in the range from 0 to 6; however, any hashable type can be used for the index label.

```python
tree = rrcf.RCTree()

for i in range(6):
    x = np.random.randn(2)
    tree.insert_point(x, index=i)
```

Visualizing the result:

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

# Deleting points

Similarly, points can be removed from the `RCTree`'s memory by using the `forget_point` method.

We can remove an individual point by supplying its index label as an argument to `forget_point`.

```
tree.forget_point(2)
```

Visualizing the result:

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

Note that the point with the index label `2` has been removed from the tree.
