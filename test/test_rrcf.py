import json
import numpy as np
import rrcf

np.random.seed(0)
n = 100
d = 3
X = np.random.randn(n, d)
Z = np.copy(X)
Z[90:, :] = 1

tree = rrcf.RCTree(X)
duplicate_tree = rrcf.RCTree(Z)

tree_seeded = rrcf.RCTree(random_state=0)
duplicate_tree_seeded = rrcf.RCTree(random_state=np.random.RandomState(0))

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

def test_codisp():
    for i in range(100):
        codisp = tree.codisp(i)
        assert codisp > 0

def test_disp():
    for i in range(100):
        disp = tree.disp(i)
        assert disp > 0

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

def test_batch_with_duplicates():
    # Instantiate tree with 10 duplicates
    leafcount = duplicate_tree._count_leaves(tree.root)
    assert (leafcount == n)
    for i in range(90, 100):
        try:
            assert duplicate_tree.leaves[i].n == 10
        except:
            print(i)
            print(duplicate_tree.leaves[i].n)
            raise

def test_insert_duplicate():
    # Insert duplicate point
    point = (1., 1., 1.)
    leaf = duplicate_tree.insert_point(point, index=100)
    assert leaf.n == 11
    for i in range(90, 100):
        try:
            assert duplicate_tree.leaves[i].n == 11
        except:
            print(i)
            print(duplicate_tree.leaves[i].n)
            raise

def test_find_duplicate():
    # Find duplicate point
    point = (1, 1, 1)
    duplicate = duplicate_tree.find_duplicate(point)
    assert duplicate is not None

def test_forget_duplicate():
    # Forget duplicate point
    leaf = duplicate_tree.forget_point(100)
    for i in range(90, 100):
        assert duplicate_tree.leaves[i].n == 10

def test_shingle():
    shingle = rrcf.shingle(X, 3)
    step_0 = next(shingle)
    step_1 = next(shingle)
    assert (step_0[1] == step_1[0]).all()

def test_random_state():
    # The two trees should have the exact same random-cuts
    points = np.random.uniform(size=(100, 5))
    for idx, point in enumerate(points):
        tree_seeded.insert_point(point, idx)
        duplicate_tree_seeded.insert_point(point, idx)
    assert str(tree_seeded) == str(duplicate_tree_seeded)

def test_insert_depth():
    tree = rrcf.RCTree()
    tree.insert_point([0., 0.], index=0)
    tree.insert_point([0., 0.], index=1)
    tree.insert_point([0., 0.], index=2)
    tree.insert_point([0., 1.], index=3)
    tree.forget_point(index=3)
    min_depth = min(leaf.d for leaf in tree.leaves.values())
    assert min_depth >= 0

def test_to_dict():
    tree = rrcf.RCTree()
    tree.insert_point([0., 0.], index=0)
    tree.insert_point([0., 0.], index=1)
    tree.insert_point([0., 0.], index=2)
    tree.insert_point([0., 1.], index=3)
    obj = tree.to_dict()
    X = np.random.randn(10, 3)
    X[5] = X[2]
    tree = rrcf.RCTree(X)
    obj = tree.to_dict()
    with open('tree.json', 'w') as outfile:
        json.dump(obj, outfile)

def test_from_dict():
    num_leaves = 10
    with open('tree.json', 'r') as infile:
        obj = json.load(infile)
    tree = rrcf.RCTree()
    tree.load_dict(obj)
    tree = rrcf.RCTree.from_dict(obj)
    # Ensure we didn't drop any duplicate leaves
    assert len(tree.leaves) == num_leaves

def test_print():
    tree = rrcf.RCTree()
    tree.insert_point([0., 0.], index=0)
    tree.insert_point([0., 0.], index=1)
    tree.insert_point([0., 1.], index=3)
    print(list(tree.leaves.values())[0])
    print(tree.root)
