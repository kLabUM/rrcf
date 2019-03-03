Trees are dynamically maintained by inserting and deleting data points. Dynamic
maintenance of trees (i) allows for efficient identification of anomalies on
streaming data, (ii) allows the model to adapt over time as the input data
changes, and (iii) creates a natural framework for scoring anomalies (wherein
anomalies are defined as points that substantially change the tree structure
upon insertion or deletion).

The *deletion* operation seeks to take a tree $T$ drawn from $RRCF(S)$
and remove a point $p \in S$, thereby producing a tree $T'$ drawn from the
distribution $RRCF(S - \{p\})$. As shown in the original paper by
\cite{guha2016robust}, this goal can be accomplished simply by removing the leaf
$P$ corresponding to point $p$, then removing its parent node, and then finally,
short-circuiting the sibling of $P$ to the grandparent of $P$ (see Algorithm
\ref{alg:2} in the Appendix for more details). Similarly, the \textit{insertion}
operation seeks to take a tree $T$ drawn from $RRCF(S)$ along with a point $p
\not \in S$ and produce a tree $T'$ drawn from $RRCF(S \cup \{p\})$. The
algorithm for insertion is considerably more involved: for each iteration, one
must (i) generate a new random cut along a random dimension and check whether
the cut separates $S$ and $p$, (ii) if the cut separates $S$ and $p$, create a
new parent node with $P$ as one child and the subtree $T(S)$ as the other child,
(iii) if the cut does not separate $S$ and $p$, follow the existing cut in the
tree (e.g. if $p_i$ is less than the existing cut value, go to the left child,
else go to the right child) and then start again from step (i) with the subtree
rooted at the new child node. Algorithm \ref{alg:3} in the Appendix shows the
insertion algorithm in full detail.

\begin{algorithm}
 \KwData{Robust random cut tree $T$ and a point $p$.}
 \KwResult{Robust random cut tree $T'$ with point $p$ removed.}
 \vspace{6pt}
 
 \SetKwFunction{forgetpoint}{ForgetPoint}
 \forgetpoint{$T, p$}

 Find leaf $P$ in $T$ corresponding to point $p$\;
 Let $R$ be the sibling node of $P$\;
 Delete the parent node of $P$ and replace it with $R$\;
 Return the modified tree $T'$\;
 \caption{Remove a point from the robust random cut tree.}
 \label{alg:2}
\end{algorithm}


\begin{algorithm}
 \KwData{An existing point set $S'$ partitioned by tree $T'$, and a new point $p$}
 \KwResult{New tree $T'(S' \cup \{p\}$.}
 \vspace{6pt}
 
 Let $x_i^\ell$ be the minimum value of $x$ with respect to dimension $i$\;
 Let $x_i^h$ be the maximum value of $x$ with respect to dimension $i$\;
 Let $B(S') = [x_1^\ell, x_1^h] \times [x_2^\ell, x_2^h] \times \dots \times [x_d^\ell, x_d^h]$ be the bounding box of point set $S'$\;
 Let $Q$ be the root node of tree $T'$, with cut dimension $k$ and cut value $q$\;

 \vspace{6pt}

 \SetKwFunction{insertpoint}{InsertPoint}
 \insertpoint{$T', p$)}\\
 \eIf{$S' = \emptyset$}{
   \KwRet{\rrct{$p$}}\;
   }
   {
   $\hat{x}_i^\ell \gets \min(p_i, x_i^\ell), \forall i$\;
   $\hat{x}_i^h \gets \max(p_i, x_i^h), \forall i$\;
   Choose a random number $r \in [0, \sum_{i} (\hat{x}_i^h - \hat{x}_i^\ell)]$\;
   $j \gets \text{argmin} \{ j | \sum_{i=1}^j (\hat{x}_i^h - \hat{x}_i^\ell) \geq r \}$\;
   $c \gets x_j^\ell + \sum_{i=1}^j (\hat{x}_i^h - \hat{x}_i^\ell) - r$\;
   \eIf{$c \not\in [x_j^\ell, x_j^h]$}
   {
    Create a new leaf $P$ corresponding to point $p$\;
    Create a new parent node $C$ corresponding to cut value $c$ along dimension $j$\;
    If root node $Q$ of tree $T'$ has a parent, replace $Q$ with $C$\;
    \eIf{$p_j \leq c$}{Set $P$ as the left child of $C$, and let the right child be the root $Q$ of subtree $T'$}
    {Set $P$ as the right child of $C$, and let the left child be the root $Q$ of subtree $T'$}
   }
   {
   $T_1' \gets$ subtree to the left of root node $Q$\;
   $T_2' \gets$ subtree to the right of root node $Q$\;
   \eIf{$p_k \leq q$}{\insertpoint{$T_1', p$}}{\insertpoint{$T_2', p$}}
   }
   }
 \caption{Insert a point into the robust random cut tree.}
\label{alg:3}
\end{algorithm}
