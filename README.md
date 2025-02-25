<div align="center">
    <img alt="GeoSimilarity Logo with text on the side." src="https://raw.githubusercontent.com/atharvaaalok/geosimilarity/refs/heads/main/docs/assets/logos/logo_with_text_inside.svg" height="300px">
</div>

GeoSimilarity is a differentiable curve and surface similarity loss function library.

It is built on top of [PyTorch](https://github.com/pytorch/pytorch) and is designed with
modularity and extensibility in mind.

> [!NOTE]  
> We are in the very early stages of development and are inviting domain experts for collaboration.

For a flavor of the documentation please refer:
[Documentation Webpage](https://atharvaaalok.github.io/geosimilarity/)


## Inviting Collaborators
We are in the very early stages of development are are inviting collaborators for:
- **Code Optimization:** Benchmark your results and show a clear improvement.
- **Novel Loss Functions:** Developing new curve/surface/point cloud similarity measures.
- **Testing:** Incorporating testing codes.
- **Improving Documentation:** Improving doc-string clarity and including doc tests. Also for
  improving the documentation website and adding explanations of similarity measures.

We'll use [Github issues](https://github.com/atharvaaalok/geosimilarity/issues) for tracking pull
requests and bugs.


## Installation
To pip install run:
```
$ pip install geosimilarity
```


## Basic Usage
Given a candidate curve $Xc$ and a target curve $Xt$ we can measure the similarity using a given
measure as follows:
```python
import torch
import geosimilarity as gs

Xc = torch.randn(10, 2, requires_grad = True)
Xt = torch.randn(10, 2)

# Define a loss object and compute the similarity between curves
loss_fn = gs.MSELoss()
loss = loss_fn(Xc, Xt)

# To modify Xc to fit the target use autograd capacity for gradient stepping
loss.backward()
```


## License
Distributed under the [MIT License](License).