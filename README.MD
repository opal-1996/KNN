# K-Nearest Neighbors Algorithm

## Introduction

The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it's typically used as a classification algorithm, working off the assumption that similar points can be found near one another.

KNN is called `lazy learning` algorithm because it doesn't perform any training when you supply the training data. It just stores the data during the training time and doesn't perform any calculations. It doesn't build a model until a query is performed on the dataset. This makes KNN ideal for data mining. 

Consider there are two groups, A and B. To determine whether a data point is in group A or group B, the algorithm looks at the states of the data points near it. If the majority of data points are in group A, it's very likely that the data point in question is in group A and vice versa. 

## Algorithm Example (k=1)

**Training** algorithm:

* for $i=1,2,...,n$ in the $n$-dimensional training dataset $D$:  store training examples $(x_i, f(x_i))$

**Prediction** algorithm:

* closest point = None
* closest distance = $\infin$
* for $i=1,2,...,n:$ 
  * current distance = $d(x_i, x_q)$
  * if current distance < closest distance:
    * closest distance = current distance
    * closet point = $x_i$
* prediction $h(x_q)$ is the target value of closest point

Unless noted otherwise, the default distance metric of nearest neighbor algorithms is the `Euclidean Distance`, which computes the distance between two points $x_a$ and $x_b$:

$$
\begin{aligned}
d\left(\mathbf{x}^{[a]}, \mathbf{x}^{[b]}\right)=\sqrt{\sum_{j=1}^m\left(x_j^{[a]}-x_j^{[b]}\right)^2}
\end{aligned}
$$

## KNN Classification & Regression

Previously, we described the training and inference algorithms of 1-NN, which makes a prediction by assigning the class label or continuous target value of the most similar training example to the query point. 

Instead of basing the prediction of the single, most similar training example, KNN considers the k nearest neighbors when predicting a class label (in classification) or a continuous value (in regression).

### Classification

In the classification setting, the simplest incarnation of the KNN model is to predict the target class label as the class label that is most often represented among the k most similar training examples for a given query point. In other words, the class label can be considered as the `mode` of the k training labels or the outcome of a `plurality voting`:

$$
\begin{aligned}
h\left(\mathbf{x}^{[t]}\right)=\operatorname{mode}\left(\left\{f\left(\mathbf{x}^{[1]}\right), \ldots, f\left(\mathbf{x}^{[k]}\right)\right\}\right)
\end{aligned}
$$

### Regression

In the regression setting, we first find the k nearest neighbors in the dataset. Secondly, we make a prediction based on the labels of the k nearest neighbors. A common approach for computing the continuous target is to compute the mean or average target value over the k nearest neighbors:

$$
\begin{aligned}
h\left(\mathbf{x}^{[t]}\right)=\frac{1}{k} \sum_{i=1}^k f\left(\mathbf{x}^{[i]}\right)
\end{aligned}
$$

## Curse of Dimensionality

The KNN algorithm is particularly susceptible to the `curse of dimensionality`. In machine learning, the curse of dimensionality refers to scenarios with a fixed size of training examples but an increasing number of dimensions and range of feature values in each dimension in a high-dimensional feature space. 

In KNN an increasing number of dimensions becomes increasingly problematic because the more dimensions we add, the larger the volume in the hyperspace needs to be to capture a fixed number of neighbors. As the volume grows larger and larger, the neighbors become less and less similar to the query point as they are now all relatively distant from the query point considering all different dimensions that are included when computing the pairwise distances. 

## Common Use Cases of KNN

While Neural Networks are gaining popularity in the computer vision and pattern recognition field, one area where KNN models are still commonly and successfully being used is in the intersection between computer vision, pattern classification, and biometrics (e.g., to make predictions based on extracted geometrical features).

Other common use cases include recommender system (via collaborative filtering) and outlier detection.

## Other Distance Metrics

Other than the Euclidean distance, we have other distance metrics to use when measuring the pairwise distance between data points:

* Manhattan Distance:

$$
\begin{aligned}
d(x, y)=\left(\sum_{i=1}^m\left|x_i-y_i\right|\right)
\end{aligned}
$$

* Minkowski Distance:
  
$$
\begin{aligned}
d(x, y)= \left(\sum_{i=1}^n\left|x_i-y_i\right|\right)^{1 / p}
\end{aligned}
$$

* Hamming Distance:

$$
\begin{aligned}
D_H=\left(\sum_{i=1}^k\left|x_i-y_i\right|\right) \\
x_i=y_i, D_i=0 \\
x_i \neq y_i, D_i=1 
\end{aligned}
$$

## How to Choose the Factor K?

K value indicates the count of the nearest neighbors and there are no pre-defined statistical methods to find the most favorable value of K. Usually we will plot the error rates with different K values, and choose the favorable K value which is located at the "elbow" of the lineplot.