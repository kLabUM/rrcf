# Related work

Traditional approaches for anomaly detection most often work by constructing a
profile of "normal" data points, then finding the points that deviate from
this profile using a distance or density metric (Liu et al. 2012) [^1]. We
briefly survey some of these methods here:

**One-class support vector machines (OC-SVM)**
: This approach solves for the SVM decision boundary using a single class
(Tax and Duin 2004) [^2]. Data that are similar to the training data are classified as
"normal", while data that are dissimilar are classified as outliers. While
effective for high-dimensional data, this method requires specification of a
kernel function, along with tuning of hyperparameters. Additional methods for
updating the decision boundary are required if the distribution of inliers or
outliers changes over time.

**Robust covariance estimation**
: If the data are assumed to come from a known distribution (e.g. Gaussian),
an elliptic envelope can be used to separate inliers from outliers (Rousseeuw and Driessen 1999) [^3].
This method requires the user to specify the assumed distribution of the data,
which may potentially lead to inaccurate results if the distribution is unknown.

**Local outlier factor detection**
: This technique measures
deviations in local density between neighboring points, and labels as outliers
those points with significantly smaller local density (Breunig et al. 2000) [^4]. This
approach may not be performant for high-dimensional data, and is not well-suited
for streaming data (given that inclusion of a new point changes the anomaly
score of existing points).

**Replicator neural networks**
: This technique uses an autoencoder
to produce a compressed representation of the input data, then classifies
outliers as points that have large reconstruction error (Williams et al. 2002) [^5].
This approach is capable of handling high-dimensional streaming data. However,
the network must be retrained if the distribution of inliers or outliers
changes.

## Isolation Forest

To address problems with the methods mentioned above, the Isolation Forest (IF) algorithm proposes a novel
ensemble method that isolates anomalies directly without relying on an explicit
distance or density metric (Liu et al. 2012) [^1]. The IF algorithm works by
recursively partitioning a point set \\(S\\) such that all points are isolated in
non-overlapping axis-aligned bounding boxes. Partitions are stored in a binary
search tree. Outliers are then identified based on their depth in the tree.
Because outliers are isolated earlier on average, points with a depth that is
much smaller than the average depth are more likely to be outliers.

## Motivation for a new ensemble method

While the Isolation Forest
algorithm has shown promising performance in detecting outliers, it suffers from
a few key limitations. First, it is not designed for use with streaming data,
given that points cannot be inserted or deleted from isolation trees once the
tree has been constructed. Second, the IF algorithm is sensitive to "irrelevant
dimensions", meaning that partitions are often wasted on dimensions that
provide relatively little information. Finally, while the tree depth shows
empirical success in detecting outliers, there is little theoretical
justification for using this metric as an anomaly score (Guha et al. 2016) [^6].
The RRCF algorithm was devised to overcome these limitations by using a novel sketching
algorithm (Guha et al. 2016) [^6].

## References

[^1]: Liu, F. T., Ting, K. M., Zhou, Z.-H., 2012. Isolation-based anomaly detection. ACM Transactions on Knowledge Discovery from Data (TKDD) 6 (1), 3.

[^2]: Tax, D. M., Duin, R. P., 2004. Support vector data description. Machine learning 54 (1), 45–66.

[^3]: Rousseeuw, P. J., Driessen, K. V., 1999. A fast algorithm for the minimum covariance determinant estimator. Technometrics 41 (3), 212–223.

[^4]: Breunig, M. M., Kriegel, H.-P., Ng, R. T., Sander, J., May 2000. Lof: Identifying density-based local outliers. SIGMOD Rec. 29 (2), 93–104.

[^5]: Williams,  G.,  Baxter,  R.,  He,  H.,  Hawkins,  S.,  Gu,  L.,  2002.  A  comparative  study  of  RNN  for outlier detection in data mining. In: Data Mining, 2002. ICDM 2003. Proceedings. 2002 IEEE International Conference on. IEEE, pp. 709–712.

[^6]: Guha, S., Mishra, N., Roy, G., Schrijvers, O., 2016. Robust random cut forest based anomaly detection on streams. In: International conference on machine learning. pp. 2712–2721.
