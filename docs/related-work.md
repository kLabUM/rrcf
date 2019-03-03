# Related work

Traditional approaches for anomaly detection most often work by constructing a
profile of ``normal'' data points, then finding the points that deviate from
this profile using a distance or density metric (Liu 2012). We
briefly survey some of these methods here:

**One-class support vector machines (OC-SVM)**
: This approach solves for the SVM decision boundary using a single class
(Tax 2004). Data that are similar to the training data are classified as
"normal", while data that are dissimilar are classified as outliers. While
effective for high-dimensional data, this method requires specification of a
kernel function, along with tuning of hyperparameters. Additional methods for
updating the decision boundary are required if the distribution of inliers or
outliers changes over time.

**Robust covariance estimation**
: If the data are assumed to come from a known distribution (e.g. Gaussian),
an elliptic envelope can be used to separate inliers from outliers (Rousseeuw 1999}.
This method requires the user to specify the assumed distribution of the data,
which may potentially lead to inaccurate results if the distribution is unknown.

**Local outlier factor detection**
: This technique measures
deviations in local density between neighboring points, and labels as outliers
those points with significantly smaller local density (Breunig 2000). This
approach may not be performant for high-dimensional data, and is not well-suited
for streaming data (given that inclusion of a new point changes the anomaly
score of existing points).

**Replicator neural networks**
: This technique uses an autoencoder
to produce a compressed representation of the input data, then classifies
outliers as points that have large reconstruction error (Williams 2002).
This approach is capable of handling high-dimensional streaming data. However,
the network must be retrained if the distribution of inliers or outliers
changes.

## Isolation Forest

To address problems with the methods mentioned above, the Isolation Forest (IF) algorithm proposes a novel
ensemble method that isolates anomalies directly without relying on an explicit
distance or density metric (Liu 2012). The IF algorithm works by
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
justification for using this metric as an anomaly score (Guha 2016).
