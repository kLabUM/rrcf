The anomaly score of a point is defined by its
\textit{collusive displacement}, which measures the change in model complexity
incurred by inserting or deleting a given point $x$. First, observe that because
the RRCT is a binary search tree, the depth of a particular point in the tree is
equivalent to its \textit{bit depth} (e.g. the number of bits needed to store
the point). The model complexity can thus be represented as the sum of bit
depths of all points in the tree. Within this context, an \textit{outlier} is
defined as a point that significantly increases the model complexity when it is
included in the tree. Quantifying this concept, we can define the
\textit{displacement} induced by a point $x$ as the expected change in the bit
depths of all leaves in a RRCT tree if point $x$ is removed:

\begin{equation}
    Disp(x, Z) = \sum_{T, y \in Z - \{x\}} Pr[T] \biggl( f(y, Z, T) - f(y, Z - \{x\}, T \biggr)
\end{equation}

where $x$ is the point to be removed, $Z$ is the point set, $T$ is an RRCT tree,
and $f(y, Z, T)$ the depth of point $y$ in the tree $T$ defined on point set
$Z$. It can be observed that the displacement associated with a point $x$ is
simply equal to the number of leaves in the subtree beneath the sibling node of
$x$ \citep{guha2016robust}.

The \textit{collusive displacement} extends the notion of \textit{displacement}
by accounting for duplicates and near-duplicates that can mask the presence of
outliers. Consider, for example, an outlier $p$ located far away from a large
cluster of inliers. In the absence of other outliers, the displacement
associated with $p$ will be large; however if there is another outlier $q$ close
to this original outlier, then the displacement of $p$ in the presence of $q$
will be small (given that $p$ will likely only displace $q$ when removed from
the tree). To account for this problem, we can compute the displacement that
occurs when removing a set of ``colluders" $C$ alongside the point of interest
$x$. Let $x$ be the point to be removed, and let $C$ be the set of ``collusive''
points to be removed alongside $x$. The collusive displacement is then defined
as the expected change in the depth of points in the tree when a point set $C$
containing $x$ is removed:

\begin{equation}
    CoDisp(x, Z, |S|) = \underset{S \subseteq Z, T}{\mathbb{E}} \biggl[ \underset{x \in C \subseteq S}{\max} \frac{1}{|C|} \sum_{y \in S - C} \biggl( f(y, S, T) - f(y, S - C, T'') \biggr) \biggr]
\end{equation}

Note that this formulation of CoDisp attempts to find the smallest subset of
points $C \supseteq x$ that maximizes the total displacement if all points are
simultaneously removed. In theory, this involves searching over all subsets $C$
containing $x$, which is impractical. While the original paper states that
CoDisp can be estimated efficiently by considering only ``subtrees in the leaf
to root path of $x$'' \citep{guha2016_si}, it does not provide an explicit
algorithm for computing CoDisp. Thus, \textbf{we propose an algorithm for
estimating CoDisp}. Starting at the leaf of interest, this algorithm traverses
the leaf to root path, while at each step computing the number of leaves in the
subtree containing $x$ (equal to $|C|$) and the number of leaves in the sibling
subtree (equal to $Disp(C)$). The maximum ratio of these two quantities
($Disp(C) / |C|$) over all nodes in the leaf-to-root path is equal to the CoDisp
of $x$. A formal description of this algorithm is given in Algorithm
\ref{alg:4}.

\begin{algorithm}
 \KwData{Robust random cut tree $T$ and a point $p$.}
 \KwResult{Collusive displacement score if point $p$ is removed.}
 \vspace{6pt}
 
 \SetKwFunction{codisp}{CoDisp}
 \codisp{$T, p$}

 Find leaf $P$ in $T$ corresponding to point $p$\;
 $N \gets P$\;
 $i \gets 1$\;
 \While{$N$ is not the root node}
 {
  $N \gets$ parent of $N$\;
  $T_1 \gets$ subtree to the left of node $N$\;
  $T_2 \gets$ subtree to the right of node $N$\;
  \eIf{$P \in T_1$}
  {$G_i \gets |T_2| / |T_1|$\;}
  {$G_i \gets |T_1| / |T_2|$\;}
  $i \gets i + 1$\;
 }
 \KwRet{ $\underset{i}{\max}(G_i)$}
 \caption{Efficient algorithm for computing collusive displacement.}
 \label{alg:4}
\end{algorithm}
