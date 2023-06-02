import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}

# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        super().__init__()
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]

        
        layers = []
        activations = []
        prev_dim = in_dim
        for dim, nonlin in zip(dims, nonlins):
            layers.append(nn.Linear(prev_dim, dim))
            if isinstance(nonlin, str):
                activation = ACTIVATIONS[nonlin](**ACTIVATION_DEFAULT_KWARGS[nonlin])
            elif isinstance(nonlin, nn.Module):
                activation = nonlin
            else:
                raise ValueError(f"Unknown activation function: {nonlin}")
            activations.append(activation)
            prev_dim = dim
        
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def forward(self, x: Tensor) -> Tensor:
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x
