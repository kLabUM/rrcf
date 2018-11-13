import numpy as np

class RCTree:
    """
    Robust random cut tree data structure.

    Example usage:
    X = np.random.randn(100,2)
    tree = RCTree(X)

    Attributes:
    -----------
    root: Pointer to root of tree
    leaves: Dict containing pointers to all leaves in tree
    """
    def __init__(self, X, root=None):
        # Initialize dict for leaves
        self.leaves = {}
        # Initialize tree root
        self.root = root
        # Store dimension of dataset
        self.ndim = X.shape[1]
        # Set node above to None in case of bottom-up search
        self.u = None
        # Create RRC Tree
        S = np.ones(X.shape[0], dtype=np.bool)
        self._mktree(X, S, parent=self)
        # Remove parent of root
        self.root.u = None
        # Count all leaves under each branch
        self._count_all_top_down(self.root)

    def _cut(self, X, S, parent=None, side='l'):
        # Find max and min over all d dimensions
        xmax = X[S].max(axis=0)
        xmin = X[S].min(axis=0)
        # Compute l
        l = xmax - xmin
        l /= l.sum()
        # Determine dimension to cut
        q = np.random.choice(self.ndim, p=l)
        # Determine value for split
        p = np.random.uniform(xmin[q], xmax[q])
        # Determine subset of points to left
        S1 = (X[:,q] <= p) & (S)
        # Determine subset of points to right
        S2 = (~S1) & (S)
        # Create new child node
        child = Branch(q=q, p=p, u=parent)
        # Link child node to parent
        if parent is not None:
            setattr(parent, side, child)
        return S1, S2, child

    def _mktree(self, X, S, parent=None, side='root', depth=0):
        # Increment depth as we traverse down
        depth += 1
        # Create a cut according to definition 1
        S1, S2, branch = self._cut(X, S, parent=parent, side=side)
        # If S1 does not contain an isolated point...
        if S1.sum() > 1:
            # Recursively construct tree on S1
            self._mktree(X, S1, parent=branch, side='l', depth=depth)
        # Otherwise...
        else:
            # Create a leaf node from isolated point
            i = np.asscalar(np.flatnonzero(S1))
            leaf = Leaf(i=i, d=depth, u=branch, x=X[i, :])
            # Link leaf node to parent
            branch.l = leaf
            self.leaves[i] = leaf
        # If S2 does not contain an isolated point...
        if S2.sum() > 1:
            # Recursively construct tree on S2
            self._mktree(X, S2, parent=branch, side='r', depth=depth)
        # Otherwise...
        else:
            # Create a leaf node from isolated point
            i = np.asscalar(np.flatnonzero(S2))
            leaf = Leaf(i=i, d=depth, u=branch, x=X[i, :])
            # Link leaf node to parent
            branch.r = leaf
            self.leaves[i] = leaf
        # Decrement depth as we traverse back up
        depth -= 1

    def forget_point(self, x):
        # Pop pointer to leaf out of leaves dict
        leaf = self.leaves.pop(x)
        # Find parent and grandparent
        parent = leaf.u
        grandparent = parent.u
        # Find sibling
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        # Set parent of sibling to grandparent
        sibling.u = grandparent
        # Short-circuit grandparent to sibling
        if parent is grandparent.l:
            grandparent.l = sibling
        else:
            grandparent.r = sibling
        # Update depths
        parent = grandparent
        self.traverse(parent, op=self._increment_depth, inc=-1)
        # Update leaf counts under each branch
        # TODO: Check correctness
        for _ in range(leaf.d):
            if parent:
                parent.n -= 1
                parent = parent.u
            else:
                break

    def traverse(self, node, op=(lambda x: None), *args, **kwargs):
        '''
        Traverse tree recursively, calling operation given by op on leaves

        node: node in RCTree
        op: function to call on each leaf
        *args: positional arguments to op
        **kwargs: keyword arguments to op
        '''
        if isinstance(node, Branch):
            if node.l:
                self.traverse(node.l, op=op, *args, **kwargs)
            if node.r:
                self.traverse(node.r, op=op, *args, **kwargs)
        else:
            op(node, *args, **kwargs)

    def _count_all_top_down(self, node):
        '''
        Traverse tree recursively, calling operation given by op on leaves

        node: node in RCTree
        op: function to call on each branch
        *args: positional arguments to op
        **kwargs: keyword arguments to op
        '''
        if isinstance(node, Branch):
            if node.l:
                self._count_all_top_down(node.l)
            if node.r:
                self._count_all_top_down(node.r)
            node.n = node.l.n + node.r.n

    def query(self, point, node=None):
        '''
        Search for leaf nearest to point

        point: point to search for
        node: node in RCTree
        '''
        if node is None:
            node = self.root
        return self._query(point, node)

    def _count_leaves(self, node):
        num_leaves = np.array(0, dtype=np.int64)
        self.traverse(node, op=self._accumulate, accumulator=num_leaves)
        num_leaves = np.asscalar(num_leaves)
        return num_leaves

    def disp(self, leaf):
        '''
        Compute displacement at leaf

        leaf: index of point
        '''
        if not isinstance(leaf, Leaf):
            try:
                leaf = self.leaves[leaf]
            except:
                raise KeyError('leaf must be a Leaf instance or key to self.leaves')
        parent = leaf.u
        # Find sibling
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        # Count number of nodes in sibling subtree
        displacement = sibling.n
        return displacement

    def codisp(self, leaf):
        '''
        Compute collusive displacement at leaf

        leaf: index of point
        '''
        if not isinstance(leaf, Leaf):
            try:
                leaf = self.leaves[leaf]
            except:
                raise KeyError('leaf must be a Leaf instance or key to self.leaves')
        node = leaf
        results = []
        for _ in range(node.d):
            parent = node.u
            if parent is None:
                break
            if node is parent.l:
                sibling = parent.r
            else:
                sibling = parent.l
            num_deleted = node.n
            displacement = sibling.n
            result = (displacement / num_deleted)
            results.append(result)
            node = parent
        co_displacement = max(results)
        return co_displacement

    def get_bbox(self, node=None):
        if node is None:
            node = self.root
        mins = np.full(self.ndim, np.inf)
        maxes = np.full(self.ndim, -np.inf)
        self.traverse(node, op=self._get_bbox, mins=mins, maxes=maxes)
        bbox = np.vstack([mins, maxes])
        return bbox

    def _query(self, point, node):
        if isinstance(node, Leaf):
            return node
        else:
            if point[:, node.q] <= node.p:
                return self._query(point, node.l)
            else:
                return self._query(point, node.r)

    def _increment_depth(self, x, inc=1):
        x.d += (inc)

    def _accumulate(self, x, accumulator):
        accumulator += 1

    def _get_leaves(self, x, stack):
        stack.append(x.i)

    def _get_bbox(self, x, mins, maxes):
        lt = (x.x < mins)
        gt = (x.x > maxes)
        mins[lt] = x.x[lt]
        maxes[gt] = x.x[gt]

    def insert_point(self, point, i=None):
        node = self.root
        parent = node.u
        maxdepth = max([leaf.d for leaf in self.leaves.values()])
        depth = 0
        # TODO: Check correctness of maxdepth
        for _ in range(maxdepth):
            bbox = self.get_bbox(node)
            cut_dimension, cut = self._insert_point_cut(point, bbox)
            # TODO: Should this be >= or >?
            if (cut <= bbox[0, cut_dimension]):
                leaf = Leaf(x=point, i=i, d=depth)
                branch = Branch(q=cut_dimension, p=cut, l=leaf, r=node,
                                n=(leaf.n + node.n))
                break
            elif (cut >= bbox[1, cut_dimension]):
                leaf = Leaf(x=point, i=i, d=depth)
                branch = Branch(q=cut_dimension, p=cut, l=node, r=leaf,
                                n=(leaf.n + node.n))
                break
            else:
                depth += 1
                if point[:, node.q] <= node.p:
                    parent = node
                    node = node.l
                    side = 'l'
                else:
                    parent = node
                    node = node.r
                    side = 'r'
        # Set parent of new branch and leaf
        node.u = branch
        leaf.u = branch
        branch.u = parent
        # Set child of parent to new branch
        setattr(parent, side, branch)
        # Increment depths below branch
        self.traverse(branch, op=self._increment_depth, inc=1)
        # Increment leaf count above branch
        # TODO: Check correctness
        for _ in range(leaf.d):
            if parent:
                parent.n += 1
                parent = parent.u
            else:
                break
        # Add leaf to leaves dict
        self.leaves[i] = leaf
        # Return inserted leaf for convenience
        return leaf

    def _insert_point_cut(self, point, bbox):
        """
        Generates the cut dimension and cut value
        based on the InsertPoint algorithm
        ----
        Inputs:
        S : Set of point to be split (numpy array (n x d))
        p : New point to be inserted (numpy array (1 x d))

        Returs:
        dimenstion for cut, cut value
        ----
        Example:
        InsertPoint_cut(x_inital, x_new)
        (0, 0.9758881798109296)
        """
        # Generate the bounding box
        bbox_hat = np.empty(bbox.shape)
        # Update the bounding box based on the internal point
        bbox_hat[0, :] = np.minimum(bbox[0, :], point)
        bbox_hat[1, :] = np.maximum(bbox[1, :], point)
        b_span = bbox_hat[1, :] - bbox_hat[0, :]
        b_range = b_span.sum()
        r = np.random.uniform(0, b_range)
        span_sum = np.cumsum(b_span)
        cut_dimension = np.inf
        for j in range(len(span_sum)):
            if span_sum[j] >= r:
                cut_dimension = j
                break
        if not np.isfinite(cut_dimension):
            raise ValueError("Cut dimension is not finite.")
        cut = bbox_hat[0, cut_dimension] + span_sum[cut_dimension] - r
        return cut_dimension, cut

class Branch:
    __slots__ = ['q', 'p', 'l', 'r', 'u', 'n']
    def __init__(self, q, p, l=None, r=None, u=None, n=0):
        self.l = l
        self.r = r
        self.u = u
        self.q = q
        self.p = p
        self.n = n

class Leaf:
    __slots__ = ['i', 'd', 'u', 'x', 'n']
    def __init__(self, i, d=None, u=None, x=None, n=1):
        self.u = u
        self.i = i
        self.d = d
        self.x = x
        self.n = n

