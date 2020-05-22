import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class LocalLinear(nn.Module):
    def __init__(self, in_features, local_features, kernel_size, padding=0, stride=1, bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fold_num = (in_features+2*padding-self.kernel_size)//self.stride+1
        self.weight = nn.Parameter(torch.randn(fold_num, kernel_size,local_features))
        self.bias = nn.Parameter(torch.randn(fold_num, local_features)) if bias else None

    def forward(self, x:torch.Tensor):
        x = F.pad(x, [self.padding]*2, value=0)
        x = x.unfold(-1, size=self.kernel_size, step=self.stride)
        x = torch.matmul(x.unsqueeze(2), self.weight).squeeze(2) + self.bias
        return x


class LocalSynthesizer(nn.Module):
    """Local `Synthesizer` based on an MLP with ReLU activation, but only "local" receptive fields.

    Args:
        output_dim: Dimensionality of the synthesized `messages`.
        n_hidden (optional): Number of hidden layers. Defaults to 0.
        hidden_dim (optional): Dimensionality of the hidden layers. Defaults to
            `output_dim`.
        trigger_dim (optional): Dimensionality of the trigger. Defaults to
            `output_dim`.
        context_dim (optional): Dimensionality of the context. If `None`, do
            not use context. Defaults to `None`.
    """

    def __init__(self, output_dim, n_hidden=0, hidden_dim=None, context_dim=None, non_zero_init=False):
        super().__init__()

        self.output_dim = output_dim
        self.context_dim = context_dim

        if hidden_dim is None:
            hidden_dim = output_dim

        top_layer_dim = 1 if n_hidden == 0 else hidden_dim

        self.input_layer = LocalLinear(
            in_features=2*output_dim,
            local_features=top_layer_dim,
            kernel_size=2+context_dim if context_dim is not None else 2,
            stride=2+context_dim if context_dim is not None else 2
        )

        self.layers = torch.nn.ModuleList([
            LocalLinear(
                in_features=2*output_dim,
                local_features=output_dim,
                kernel_size=2,
                stride=2
            )
            for layer_index in range(n_hidden)
        ])

        # zero-initialize the last layer, as in the paper
        if not non_zero_init:
            if n_hidden > 0:
                init.constant_(self.layers[-1].weight, 0)
                init.constant_(self.layers[-1].bias, 0)
            else:
                init.constant_(self.input_layer.weight, 0)
                init.constant_(self.input_layer.bias, 0)

    def forward(self, trigger, context):
        """Synthesizes a `message` based on `trigger` and `context`.

        Args:
            trigger: `trigger` to synthesize the `message` based on. Size:
                (`batch_size`, `trigger_dim`).
            context: `context` to condition the synthesizer. Ignored if
                `context_dim` has not been specified in the constructor. Size:
                (`batch_size`, `context_dim`).

        Returns:
            The synthesized `message`.
        """

        mean = trigger.sum(1).unsqueeze(1).repeat(1, self.output_dim)
        mean = (mean - trigger)/(self.output_dim-1)

        if self.context_dim is not None:
            # TODO implement with context
            context = context.unsqueeze(2).repeat(1, 1, self.output_dim)
            interleaved = torch.stack((trigger.unsqueeze(-1), mean.unsqueeze(-1)), dim=-1).view(-1, 2*self.output_dim)
            last = self.input_layer(interleaved)
        else:
            interleaved = torch.stack((trigger.unsqueeze(-1), mean.unsqueeze(-1)), dim=-1).view(-1, 2*self.output_dim)
            last = self.input_layer(interleaved).squeeze(-1)

        for layer in self.layers:
            mean = last.sum(1).unsqueeze(1).repeat(1, self.output_dim, 1)
            mean = (mean - last)/(self.output_dim-1)

            interleaved = torch.stack((last.unsqueeze(-1), mean.unsqueeze(-1)), dim=-1).view(-1, 2*self.output_dim)
            last = self.input_layer(interleaved)

            last = layer(F.relu(last))

        return last
