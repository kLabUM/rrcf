# Modifying a random cut tree

Trees are dynamically maintained by inserting and deleting data points. Dynamic
maintenance of trees (i) allows for efficient identification of anomalies on
streaming data, (ii) allows the model to adapt over time as the input data
changes, and (iii) creates a natural framework for scoring anomalies (wherein
anomalies are defined as points that substantially change the tree structure
upon insertion or deletion).

## Deletion of points

The *deletion* operation seeks to take a tree \\(T\\) drawn from \\(RRCF(S)\\)
and remove a point \\(p \in S\\), thereby producing a tree \\(T'\\) drawn from the
distribution \\(RRCF(S - \{p\})\\). As shown in the original paper by
Guha et al. (2016), this goal can be accomplished simply by removing the leaf
\\(P\\) corresponding to point \\(p\\), then removing its parent node, and then finally,
short-circuiting the sibling of \\(P\\) to the grandparent of $P$ (see pseudocode below).

![Algorithm 3](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/alg_3.png)

## Insertion of points

The *insertion*
operation seeks to take a tree \\(T\\) drawn from $RRCF(S)$ along with a point \\(p
\not \in S\\) and produce a tree \\(T'\\) drawn from \\(RRCF(S \cup \{p\})\\). The
algorithm for insertion is considerably more involved: for each iteration, one
must (i) generate a new random cut along a random dimension and check whether
the cut separates \\(S\\) and \\(p\\), (ii) if the cut separates \\(S\\) and \\(p\\), create a
new parent node with \\(P\\) as one child and the subtree \\(T(S)\\) as the other child,
(iii) if the cut does not separate \\(S\\) and \\(p\\), follow the existing cut in the
tree (e.g. if \\(p_i\\) is less than the existing cut value, go to the left child,
else go to the right child) and then start again from step (i) with the subtree
rooted at the new child node. The pseudocode below shows the
insertion algorithm in full detail.

![Algorithm 4](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/alg_4.png)
