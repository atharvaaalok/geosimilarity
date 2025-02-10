<div align="center">
    <img alt="GeoSimilarity Logo with text on the side." src="https://raw.githubusercontent.com/atharvaaalok/geosimilarity/refs/heads/main/docs/assets/logos/logo_with_text_inside.svg" height="300px">
</div>

GeoSimilarity is a differentiable curve and surface similarity loss function library.

It is built on top of [PyTorch](https://github.com/pytorch/pytorch) and is designed with
modularity and extensibility in mind.

> [!NOTE]  
> We are in the very early stages of development and are inviting domain experts for collaboration.
> The usage examples below are stand-ins for future api.

For a flavor of the documentation please refer:
[Documentation Webpage](https://atharvaaalok.github.io/geosimilarity/)


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


## Inviting Collaborators
We are in the very early stages of development are are inviting collaborators for:
- **Code Optimization:** Benchmark your results and show a clear improvement.
- **Novel Loss Functions:** Developing new curve/surface/point cloud similarity measures.
- **Testing:** Incorporating testing codes.
- **Improving Documentation:** Improving doc-string clarity and including doc tests. Also for
  improving the documentation website and adding explanations of similarity measures.

We'll use [Github issues](https://github.com/atharvaaalok/geosimilarity/issues) for tracking pull
requests and bugs.


## License
Distributed under the [MIT License](License).


## Project Plan:
- [x] Create first loss function. Mean Squared Error (MSE).
- [x] Generate proper documentation.
    - [x] Proper docstrings. Follow Google python coding style guide.
    - [x] Take inspiration from pytorch.
    - [x] Use math equations.
    - [x] Use type annotations.
- [x] Add .gitignore file.
- [x] Add testing code.
    - [x] Use linear splines. Create spline generator function.
    - [x] Create automated training function.
    - [x] Create plot function. Parameterized and Target shape comparison.
    - [x] Generate a bunch of target shapes. Use SVGs.
- [ ] Create documentation first cut using MyST markdown.
    - [x] Create motivation/introduction.
    - [ ] Include function docstrings in the documentation.
    - [x] Launch web page for documentation using github pages.
- [x] Add license.