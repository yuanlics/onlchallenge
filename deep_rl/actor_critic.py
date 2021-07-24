#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np  


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, exploration_param=0.05, device="cpu"):
        super(ActorCritic, self).__init__()
        # output of actor in (0, 1) range
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
#                 nn.Sigmoid()  # DEBUG 不需要sigmoid
                )
        # critic V(s)
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU(),
                nn.Linear(32, 1),
                )
        self.device = device
        self.MAX_ACTION = 20  # DEBUG 限定最大估计 mbps
        self.action_var = torch.full((action_dim,), exploration_param**2).to(self.device)  # 在均值附近随机探索。exploration_param即随机探索的标准差。
        self.random_action = True

    def forward(self, state):
        value = self.critic(state)
        action_mean = self.actor(state)
        
        # DEBUG action是receiving rate的增量，且限定最大估计
#         if state.size(0) == 1000:
#             print(state[:, 0].shape, action_mean.shape)
#             raise
#         action_mean = torch.clamp(state[:, 0].detach() + action_mean, 0, self.MAX_ACTION)
        action_mean = torch.clamp(state.flatten()[0].detach() + action_mean, 0, self.MAX_ACTION)
        
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        if not self.random_action:
            action = action_mean
        else:
            action = dist.sample()  # 随机探索

        action_logprobs = dist.log_prob(action)

        return action.detach(), action_logprobs, value

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        # DEBUG
        action_mean = torch.clamp(state.flatten()[0].detach() + action_mean, 0, self.MAX_ACTION)
        
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(state)

        return action_logprobs, torch.squeeze(value), dist_entropy

