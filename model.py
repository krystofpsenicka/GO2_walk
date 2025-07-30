import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, GaussianHead, CriticHead
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
    
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()
        
        self.encoder, dim = make_mlp_layers(obs_dim, hidden_dims, F.silu, True)
        self.policy = GaussianHead(dim, action_dim, log_std_min=-20, log_std_max=2, state_dependent_std=False)
        
    def forward(self, obs, action=None):
        obs = self.encoder(obs)
        
        step: StochasticContinuousPolicyStep = self.policy(obs, action)
        
        return step
    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims):
        super().__init__()
        
        self.encoder, dim = make_mlp_layers(obs_dim, hidden_dims, F.silu, True)
        self.value = CriticHead(dim)
        
    def forward(self, obs):
        obs = self.encoder(obs)
        
        step: ValueStep = self.value(obs)
        
        return step