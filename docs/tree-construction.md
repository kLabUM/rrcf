Given a point set $S$, a robust random cut tree (RRCT) is constructed by
recursively partitioning the point set until each point is isolated in its own
bounding box. For each iteration of the tree construction routine, a random
dimension is selected, with the probability of selecting a dimension being
proportional to the difference between its minimum and maximum values. Next, a
partition is selected uniformly at random between the minimum and maximum value
of that dimension. If the partition isolates a point $x$ from the rest of the
point set, a new leaf node for $x$ is created, and the point is removed from the
point set. The algorithm is then recursively applied to each subset of remaining
points on either side of the partition. The algorithm for constructing an RRCT
tree is formally specified in Algorithm \ref{alg:1}.

$$ \begin{algorithm}
 \KwData{Point set $S$ of size $n$ and dimension $d$\;}
 \KwResult{Generate a robust random cut tree on a point set $S$.}
 \vspace{6pt}
 \SetKwFunction{rrct}{RRCT}
 \rrct{$S$}\\
 \While{$|S| \neq 1$}{
  $\ell_i \gets \max_{x \in S} x_i - \min_{x \in S} x_i, \forall i$\;
  $p_i \gets \frac{\ell_i}{\sum_{j} \ell_j}, \forall i$\;
  Choose a random dimension $i \in \{1, \dots d\}$ with probability proportional to $p_i$\;
  Choose $X_i \sim \text{Uniform} \ [\min_{x \in S} x_i, \max_{x \in S} x_i]$\;
  $S_1 \gets \{x|x \in S, x_i \leq X_i\}$\;
  $S_2 \gets S \setminus S_1$\;
  \rrct{$S_1$}\;
  \rrct{$S_2$}\;
 }
 \caption{Construction of Robust Random Cut Tree (RRCT).}
 \label{alg:1}
%\end{tcolorbox}
\end{algorithm}$$
