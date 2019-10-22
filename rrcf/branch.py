# -*- coding: utf-8 -*-


class Branch:
    """
    Branch of RCTree containing two children and at most one parent.

    Attributes:
    -----------
    q: Dimension of cut
    p: Value of cut
    l: Pointer to left child
    r: Pointer to right child
    u: Pointer to parent
    n: Number of leaves under branch
    b: Bounding box of points under branch (2 x d)
    """
    # __slots__ = ['q', 'p', 'l', 'r', 'u', 'n', 'b']

    def __init__(self, q, p, l=None, r=None, u=None, n=0, b=None):
        self.l = l
        self.r = r
        self.u = u
        self.q = q
        self.p = p
        self.n = n
        self.b = b

    def __repr__(self):
        return "Branch(q={}, p={:.2f})".format(self.q, self.p)