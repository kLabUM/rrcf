# Inserting points

Trees can be dynamically updated in response to new data by inserting points.

Here, we create an empty `RCTree` and insert 6 new points (where each point is a 2-dimensional Gaussian-distributed random vector).

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
