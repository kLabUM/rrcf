# Constructing a robust random cut tree

Given a point set \\(S\\), a robust random cut tree (RRCT) is constructed by
recursively partitioning the point set until each point is isolated in its own
bounding box (Guha et al. 2016) [^1]. For each iteration of the tree construction routine, a random
dimension is selected, with the probability of selecting a dimension being
proportional to the difference between its minimum and maximum values. Next, a
partition is selected uniformly at random between the minimum and maximum value
of that dimension. If the partition isolates a point \\(x\\) from the rest of the
point set, a new leaf node for \\(x\\) is created, and the point is removed from the
point set. The algorithm is then recursively applied to each subset of remaining
points on either side of the partition. The algorithm for constructing an RRCT
tree is formally specified below:

![Algorithm 1](https://s3.us-east-2.amazonaws.com/mdbartos-img/rrcf/alg_1.png)

## References
[^1]: Guha, S., Mishra, N., Roy, G., Schrijvers, O., 2016. Robust random cut forest based anomaly detection on streams. In: International conference on machine learning. pp. 2712–2721.
