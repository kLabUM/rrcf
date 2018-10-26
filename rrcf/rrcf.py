import numpy as np

class RCTree:
    def __init__(self, X, root=None):
        self.leaves = {}
        self.root = root
        self.u = None
        S = np.ones(X.shape[0], dtype=np.bool)
        self.mktree(X, S, parent=self)

    def cut(self, X, S, parent=None, side='l'):
        xmax = X[S].max(axis=0)
        xmin = X[S].min(axis=0)
        l = xmax - xmin
        l /= l.sum()
        q = np.random.choice(2, p=l)
        p = np.random.uniform(xmin[q], xmax[q])
        S1 = (X[:,q] <= p) & (S)
        S2 = (~S1) & (S)
        child = Branch(q=q, p=p, u=parent)
        if parent is not None:
            setattr(parent, side, child)
        return S1, S2, child

    def mktree(self, X, S, parent=None, side='root', depth=0):
        depth += 1
        S1, S2, branch = self.cut(X, S, parent=parent, side=side)
        if S1.sum() > 1:
            self.mktree(X, S1, parent=branch, side='l', depth=depth)
        else:
            i = np.asscalar(np.flatnonzero(S1))
            leaf = Leaf(i=i, d=depth, u=branch)
            branch.l = leaf
            self.leaves[i] = leaf
        if S2.sum() > 1:
            self.mktree(X, S2, parent=branch, side='r', depth=depth)
        else:
            i = np.asscalar(np.flatnonzero(S2))
            leaf = Leaf(i=i, d=depth, u=branch)
            branch.r = leaf
            self.leaves[i] = leaf
        depth -= 1

    def delete_leaf(self, i):
        node = self.leaves.pop(i)
        parent = node.u
        grandparent = parent.u
        if node is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        sibling.u = grandparent
        if parent is grandparent.l:
            grandparent.l = sibling
        else:
            grandparent.r = sibling
        # Update depths
        self.traverse(grandparent, op=self._increment_depth, inc=-1)
        del node
        del parent

    def traverse(self, node, op=(lambda x: None), *args, **kwargs):
        if isinstance(node, Branch):
            if node.l:
                self.traverse(node.l, op=op, *args, **kwargs)
            if node.r:
                self.traverse(node.r, op=op, *args, **kwargs)
        else:
            op(node, *args, **kwargs)

    def _increment_depth(self, x, inc=1):
        x.d += (inc)

class Branch:
    __slots__ = ['q', 'p', 'l', 'r', 'u']
    def __init__(self, q, p, l=None, r=None, u=None):
        self.l = l
        self.r = r
        self.u = u
        self.q = q
        self.p = p

class Leaf:
    __slots__ = ['i', 'd', 'u']
    def __init__(self, i, d=None, u=None):
        self.u = u
        self.i = i
        self.d = d

