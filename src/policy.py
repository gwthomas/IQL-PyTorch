import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .util import mlp


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        scale_tril = torch.diag(std)
        if obs.ndim > 1:
            batch_size = len(obs)
            return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        else:
            return MultivariateNormal(mean, scale_tril=scale_tril)
