import numpy as np
import rrcf

np.random.seed(0)
n = 100
d = 3
X = np.random.randn(n, d)
tree = rrcf.RCTree(X)

deck = np.arange(n, dtype=int)
np.random.shuffle(deck)
indexes = deck[:5]

def test_batch():
    # Check stored bounding boxes and leaf counts after instantiating from batch
    branches = []
    tree.map_branches(tree.root, op=tree._get_nodes, stack=branches)
    leafcount = tree._count_leaves(tree.root)
    assert (leafcount == n)
    for branch in branches:
        leafcount = tree._count_leaves(branch)
        assert (leafcount == branch.n)
        bbox = tree.get_bbox(branch)
        assert (bbox == branch.b).all()

def test_forget_batch():
    # Check stored bounding boxes and leaf counts after forgetting points
    for index in indexes:
        forgotten = tree.forget_point(index)
        branches = []
        tree.map_branches(tree.root, op=tree._get_nodes, stack=branches)
        for branch in branches:
            leafcount = tree._count_leaves(branch)
            try:
                assert (leafcount == branch.n)
            except:
                print(forgotten.x)
                print('Computed:\n', leafcount)
                print('Stored:\n', branch.n)
                raise
            bbox = tree.get_bbox(branch)
            try:
                assert np.allclose(bbox, branch.b)
            except:
                print(forgotten.x)
                print('Computed:\n', bbox)
                print('Stored:\n', branch.b)
                raise

def test_insert_batch():
    # Check stored bounding boxes and leaf counts after inserting points
    for index in indexes:
        x = np.random.randn(d)
        tree.insert_point(x, index=index)
        branches = []
        tree.map_branches(tree.root, op=tree._get_nodes, stack=branches)
        for branch in branches:
            leafcount = tree._count_leaves(branch)
            try:
                assert (leafcount == branch.n)
            except:
                print(x)
                print('Computed:\n', leafcount)
                print('Stored:\n', branch.n)
                raise
            bbox = tree.get_bbox(branch)
            try:
                assert np.allclose(bbox, branch.b)
            except:
                print(x)
                print('Computed:\n', bbox)
                print('Stored:\n', branch.b)
                raise
