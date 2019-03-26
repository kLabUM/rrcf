# Caveats and gotchas

## Scaling of dimensions

The RRCF algorithm considers the relative scale of each dimension when constructing robust random cut trees. This means that dimensions with less variability (on an absolute scale) will affect the outlier score of a point less than dimensions with higher variability.

This consideration is important to remember if each dimension represents a different categorical property or is measured with a different set of units. Consider, for example the following dataset.

| Person      | Height (in)  | Weight (lb)   | Age (yr)    | 
| ------------| ------------ | ------------- | ----------- |
| Alice       | 61           | 105           | 34          |
| Bob         | 70           | 300           | 50          |
| Timmy       | 48           | 70            | 10          |
| Nosferatu   | 75           | 180           | 170         |

In this case, `Weight` will influence the outlier score most, because the range between the maximum and minimum values is largest (300 - 70 = 230). However, looking at the table, age seems like the most intuitive category for determining the outlier (in this case, Nosferatu is more than three times as old as the second-oldest person).

In cases where each column is measured in different units, or measures a different type of quantity, it may be necessary to scale each column before constructing the random cut tree. For example, min-max scaling each column between zero and one yields:

| Person      | Height (-)   | Weight (-)    | Age (-)     | 
| ------------| ------------ | ------------- | ----------- |
| Alice       | 0.48         | 0.15          | 0.15        |
| Bob         | 0.81         | 1.0           | 0.25        |
| Timmy       | 0.0          | 0.0           | 0.0         |
| Nosferatu   | 1.0          | 0.48          | 1.0         |

Other scaling methods may suit other datasets better (for instance, scaling each dimension to a mean of zero and a standard deviation of one). The user should experiment with different scalings to determine the method that works best for the task at hand.
