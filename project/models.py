import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

torch.manual_seed(42)
ACTIVATIONS = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu6': nn.ReLU6,
    'softsign': nn.Softsign,
    'selu': nn.SELU,
    'elu': nn.ELU,
    'relu': nn.ReLU,
    'none': None
}


class MLP(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, activation='none'):
        super().__init__()
        self.activation = ACTIVATIONS[activation]
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Linear(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim
                )
            )
            if self.activation is not None and i != num_layers - 1:
                layers.append(self.activation())
        self.model = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

        self.__name__ = f"MLP with {activation} activation"

    def forward(self, x):
        return self.model(x)


class NeuralAccumulatorCell(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.W_hat)
        nn.init.kaiming_uniform_(self.M_hat)

    def forward(self, x):
        W = F.tanh(self.W_hat) * F.sigmoid(self.M_hat)
        return F.linear(x, W, self.bias)


class NAC(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralAccumulatorCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim
                )
            )
        self.model = nn.Sequential(*layers)
        self.__name__ = 'NAC'

    def forward(self, x):
        return self.model(x)


class NeuralArithmeticLogicUnitCell(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.eps = 1e-10
        self.nac = NeuralAccumulatorCell(in_dim, out_dim)
        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.register_parameter('G', self.G)
        self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.G)

    def forward(self, x):
        a = self.nac(x)
        g = F.sigmoid(F.linear(x, self.G, self.bias))
        log_x = torch.log(torch.abs(x) + self.eps)
        m = torch.exp(self.nac(log_x))
        return g * a + (1 - g) * m


class NALU(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralArithmeticLogicUnitCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim
                )
            )
        self.model = nn.Sequential(*layers)
        self.__name__ = 'NALU'

    def forward(self, x):
        return self.model(x)

