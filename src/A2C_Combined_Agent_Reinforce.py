import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.Configuration.StaticConf import StaticConf
from src.Model.Critic import Critic
from src.Model.Actor import Actor
from src.Model.ActorCritic import ActorCritic
from src.NetworkEnv import NetworkEnv
from src.PrioritizedReplay import PrioritizedReplayMemory


class A2C_Combined_Agent_Reinforce():
    # , experience_replay_size, priority_alpha, priority_beta_start, priority_beta_frames
    def __init__(self, models_path):
        # Hyper params:
        self.discount_factor = 0.9
        self.lr = 1e-3
        self.num_steps = 10
        self.device = StaticConf.getInstance().conf_values.device
        self.num_actions = StaticConf.getInstance().conf_values.num_actions
        self.num_episodes = 100
        self.episode_idx = 0

        self.actor_critic_model = ActorCritic(self.device, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), self.lr)

        self.env = NetworkEnv(models_path)

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def train(self):
        writer = SummaryWriter()
        frame_idx = 0

        all_rewards_episodes = []
        max_reward_in_all_episodes = -np.inf
        reward_not_improving = False
        min_epochs = 100
        action_to_compression = StaticConf.getInstance().conf_values.action_to_compression_rate

        while self.episode_idx < min_epochs or (not reward_not_improving):
            print("Episode {}/{}".format(self.episode_idx, self.num_episodes))
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []

            # rollout trajectory
            for _ in range(self.num_steps):
                dist, value = self.actor_critic_model(state)

                action = dist.sample()
                compression_rate = action_to_compression[action.cpu().numpy()[0]]
                next_state, reward, done = self.env.step(compression_rate)

                log_prob = dist.log_prob(action)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))

                state = next_state

                if done:
                    break

            writer.add_scalar('Total Reward in Episode', sum(rewards), self.episode_idx)
            self.episode_idx += 1
            # next_state = torch.FloatTensor(next_state).to(self.device)
            returns = self.compute_returns(0, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            writer.add_scalar('Actor Loss', v(actor_loss), self.episode_idx)
            writer.add_scalar('Critic Loss', v(critic_loss), self.episode_idx)

            # loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            # loss_val = loss.data.detach().cpu().numpy().min()
            # writer.add_scalar('Loss', loss_val, self.episode_idx)

            total_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            all_rewards_episodes.append(returns[-1])
            curr_reward = all_rewards_episodes[-1]

            if max_reward_in_all_episodes < v(curr_reward):
                max_reward_in_all_episodes = v(curr_reward)

            if len(all_rewards_episodes) > min_epochs and max_reward_in_all_episodes >= max(all_rewards_episodes[-20:]):
                reward_not_improving = True

def v(a):
    return a.data.detach().cpu().numpy().min()