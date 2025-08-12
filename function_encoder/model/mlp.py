from typing import List, Callable, Union
import torch


class MLP(torch.nn.Module):
    """A simple multi-layer perceptron neural network with optional SIREN-style initialization.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions.
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.ReLU().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
        omega_0 (float, optional): Frequency factor for sine activation (SIREN). Defaults to 1.0.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: Union[torch.nn.Module, Callable] = torch.nn.ReLU(),
        bias: bool = True,
        omega_0: float = 1.0,
    ):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.omega_0 = omega_0
        self.layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias)

            if self.activation == torch.sin:
                if i == 0:
                    self._init_first_layer(layer)
                else:
                    self._init_siren_layer(layer)

            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == torch.sin:
                x = self.omega_0 * x
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

    def _init_first_layer(self, layer: torch.nn.Linear):
        fan_in = layer.in_features
        with torch.no_grad():
            layer.weight.uniform_(-1 / fan_in, 1 / fan_in)
            layer.weight.mul_(self.omega_0)
            if layer.bias is not None:
                layer.bias.uniform_(-1 / fan_in, 1 / fan_in)

    def _init_siren_layer(self, layer: torch.nn.Linear):
        fan_in = layer.in_features
        bound = torch.sqrt(torch.tensor(6 / fan_in)) / self.omega_0
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound)
            if layer.bias is not None:
                layer.bias.uniform_(-bound, bound)


class MultiHeadedMLP(torch.nn.Module):
    """A multi-headed MLP that outputs multiple vectors in parallel.

    The network shares parameters for all layers except the final one,
    which produces multiple outputs (heads) simultaneously.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions
        num_heads (int): Number of output heads to produce
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.ReLU().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        num_heads: int,
        activation: Callable = torch.nn.Tanh(),  # default to sine for SIREN
        bias: bool = True,
        omega_0: float = 30,
    ):
        super(MultiHeadedMLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.omega_0 = omega_0
        self.layers = torch.nn.ModuleList()

        # Create hidden layers (except final multi-head layer)
        for i in range(len(layer_sizes) - 2):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias)
            # Only apply custom SIREN initialization if activation is sine.
            if self.activation == torch.sin:
                if i == 0:
                    self._init_first_layer(layer)
                else:
                    self._init_siren_layer(layer)
            self.layers.append(layer)

        # Create the final multi-head layer.
        final_layer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1] * num_heads, bias=bias)
        if self.activation == torch.sin:
            self._init_siren_layer(final_layer)
        self.layers.append(final_layer)

    def forward(self, x):
        # For each hidden layer, if using sine activation, multiply by omega_0 before applying sine.
        for layer in self.layers[:-1]:
            if self.activation == torch.sin:
                x = self.omega_0 * layer(x)
            else:
                x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        # Reshape output to have shape [*, layer_sizes[-1], num_heads]
        x = x.view(*x.shape[:-1], self.layer_sizes[-1], -1)
        return x

    @staticmethod
    def _init_first_layer(layer: torch.nn.Linear, omega_0: float = 30):
        """Initialize the first layer with weights from U(-1/fan_in, 1/fan_in) and then scale by omega_0."""
        fan_in = layer.in_features
        with torch.no_grad():
            layer.weight.uniform_(-1 / fan_in, 1 / fan_in)
            layer.weight.mul_(omega_0)
            if layer.bias is not None:
                layer.bias.uniform_(-1 / fan_in, 1 / fan_in)

    def _init_siren_layer(self, layer: torch.nn.Linear):
        """Initialize subsequent layers with weights from U(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)."""
        fan_in = layer.in_features
        bound = torch.sqrt(6 / torch.tensor(fan_in)) / self.omega_0
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound)
            if layer.bias is not None:
                layer.bias.uniform_(-bound, bound)

class MultiHeadedMLP_Freeze(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        num_heads: int,
        activation: Callable = torch.nn.Tanh(),
        bias: bool = True,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.shared_layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 2):
            self.shared_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))

        self.output_heads = torch.nn.ModuleList([
            torch.nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=bias)
            for _ in range(num_heads)
        ])

        self.frozen_head_indices = set()

    def freeze_heads(self, indices: List[int]):
        for idx in indices:
            for param in self.output_heads[idx].parameters():
                param.requires_grad = False
            self.frozen_head_indices.add(idx)

    def add_head(self):
        new_head = torch.nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])
        self.output_heads.append(new_head)

    def forward(self, x):
        for layer in self.shared_layers:
            x = self.activation(layer(x))

        outputs = [head(x) for head in self.output_heads]
        x = torch.stack(outputs, dim=-1)  # [batch, output_dim, num_heads]
        return x