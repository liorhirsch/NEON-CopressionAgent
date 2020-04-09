import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


num_envs = 8
env_name = "CartPole-v0"


def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

# envs = [make_env() for i in range(num_envs)]

def test_env(vis=False):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = actor_critic_model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns



num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.n

# Hyper params:
lr = 1e-3
num_steps = 10

actor_critic_model = ActorCritic(num_inputs, num_outputs).to(device)
optimizer = optim.Adam(actor_critic_model.parameters())

num_episodes = 20000
episode_idx = 0



while episode_idx < num_episodes:
    state = envs.reset()
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    # rollout trajectory
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = actor_critic_model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

        state = next_state
        episode_idx += 1

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = actor_critic_model(next_state)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
