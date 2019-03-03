# Constructing a robust random cut tree

Given a point set \(S\), a robust random cut tree (RRCT) is constructed by
recursively partitioning the point set until each point is isolated in its own
bounding box. For each iteration of the tree construction routine, a random
dimension is selected, with the probability of selecting a dimension being
proportional to the difference between its minimum and maximum values. Next, a
partition is selected uniformly at random between the minimum and maximum value
of that dimension. If the partition isolates a point \(x\) from the rest of the
point set, a new leaf node for \(x\) is created, and the point is removed from the
point set. The algorithm is then recursively applied to each subset of remaining
points on either side of the partition. The algorithm for constructing an RRCT
tree is formally specified below:

```
 \begin{align*}
 \text{Input: Point set S of size n and dimension d}\\
 \text{Generate a robust random cut tree on a point set S.}\\
 \\
 RRCT(S)\\
 \text{while} |S| \neq 1:\\
 \ell_i \gets \max_{x \in S} x_i - \min_{x \in S} x_i, \forall i \\
 p_i \gets \frac{\ell_i}{\sum_{j} \ell_j}, \forall i \\
 \text{Choose a random dimension} i \in \{1, \dots d\} \text{with probability proportional} to p_i \\
 \text{Choose} X_i \sim \text{Uniform} \ [\min_{x \in S} x_i, \max_{x \in S} x_i] \\
 S_1 \gets \{x|x \in S, x_i \leq X_i\} \\
 S_2 \gets S \setminus S_1 \\
 RRCT(S_1) \\
 RRCT(S_2) \\
 \end{align*}
```
