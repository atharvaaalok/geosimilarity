---
title: Geometric Similarity Measures
subtitle: An Introduction to Similarity Measures
subject: A Motivation
---


# Geometric Similarity Measures


## Introduction
Suppose we have a curve representation method, for example some parameterization scheme such as a
spline. We also have a particular shape specified (say the [Stanford Bunny](#stanford_bunny)) and we
want our parameterization to represent that shape as closely as possible. To represent the given
shape one way would be to tune the parameters of the representation scheme iteratively using a
gradient descent optimization approach. This requires us to define an objective function that can
then be optimized over. An appropriate objective for this task would be some measure of
similarity(or dissimilarity) between the target curve and the one traced out by our parameterization
. The objective function can then be maximized(or minimized) to fit the parameters.

The target of this tutorial is to study **geometric similarity measures**. We discuss different
kinds of measures both for curve and surface similarity and also see their implementations.

:::::{grid} 2

::::{grid-item}
:::{figure} assets/a_basic_spline.svg
:label: a_basic_spline
:height: 150px
:alt: A basic spline drawing to represent a parameterized shape.
Parameterized shape, using a spline.
:::
::::

::::{grid-item}
:::{figure} assets/stanford_bunny.svg
:label: stanford_bunny
:height: 225px
:alt: The Stanford Bunny shown as a target curve.
Target shape, the [Stanford Bunny](https://en.wikipedia.org/wiki/Stanford_bunny).
:::
::::

:::::


## Concrete Problem Setup
First we define the different objects that we deal with:

Shape Parameterization
: We use some parameterization scheme e.g. splines to represent our shapes. We have a set of
parameters $\phi$ that represent our shape. By changing $\phi$ we trace out different curves in the
plane. We will think of $\phi$ as a column vector $[\phi_1, \phi_2, \ldots, \phi_n]^{T}$.

Parameterized/Candidate Curve
: This is the curve that is traced out by the parameterization scheme. We denote it by $C_c$ and is
obtained by sampling the scheme at different points along the actual curve. It is specified in the
form of a $N_c$ length sequence of $(x, y)$ points. These points are ordered along the curve. We
will specify the points in a matrix in $\mathbb{R}^{N_c \times 2}$ where each row corresponds to a
point $(x_i, y_i)$. We denote the matrix as $X_c$.

Target Curve
: This is the curve that we want our parameterization scheme to represent. We denote it by $C_t$ and
it is specified in the form of a $N_t$ length sequence of $(x, y)$ points. These points are ordered
along the curve. We will specify the points in a matrix in $\mathbb{R}^{N_t \times 2}$ as we did for
the parameterized curve. We denote the matrix as $X_t$.

Loss Function
: A function denoted as $\mathcal{L}(X_c, X_t)$ that measures the degree of dissimilarity between
the target curve and the candidate curve. It should be differentiable to allow us to find gradients
$\frac{d\mathcal{L}}{d\phi}$ that can then be used to run gradient descent.

**_Goal_**: To tune $\phi$ such that our representation scheme traces out the target curve.


## Similarity Measures
We now discuss the different curve and surface similarity measures. For each measure we describe the
exact mathematical definition, practical considerations, modifications to make them differentiable
and implementations in [PyTorch](https://pytorch.org/).


### Mean Squared Error (MSE)

#### Description
:::{note} Assumption
$N_c = N_t = N$. That is, we sample the candidate curve at exactly $N_t$ points.
:::

The mean squared error loss function computes the average of the squared distance between the
corresponding points on the two curves. Mathematically,
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( d(X_{c}[i], X_{t}[i]) \right)^2
$$
where, $X[i]$ denotes $i${sup}`th` row, i.e. $(x_i, y_i)$ and $d$ is a distance function.

Though the measure is quite naive and not very robust, it is very simple and quick to implement and
is also differentiable without any modifications.

:::{figure} assets/mse_visualization.svg
:width: 70%
:alt: Two curves defined by an equal number of points and an MSE loss between corresponding points.

Mean Squared Error (MSE) between two curves visualized.
:::

#### Implementation

```python
import torch

def mse_loss(Xc, Xt):
    loss = torch.mean((Xc - Xt) ** 2)
    return loss
```