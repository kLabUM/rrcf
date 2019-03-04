# RCTree API documentation

This section enumerates all the methods of the `RCTree` class

## Inserting and deleting points

## Other methods

<b>`map_leaves`</b>`(node, op=(lambda x: None), *args, **kwargs)`
> Traverse tree recursively, calling operation given by op on leaves

| Argument | Type | Description |
-----------|------|--------------
| `node`      | Leaf or Branch instance | Node in RCTree |
| `op`       | function | Function to call on each leaf (defaults to no-op) |
| `*args`    | any      | Positional arguments to `op` |
| `**kwargs` | any      | Keyword arguments to `op` |

```python
# Use map leaves to print leaves in postorder

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
