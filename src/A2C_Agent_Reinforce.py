import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# def test_env(vis=False):
#     state = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         dist, _ = actor_critic_model(state)
#         next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
#         state = next_state
#         total_reward += reward
#     return total_reward
from src.Configuration.StaticConf import StaticConf
from src.Model.Critic import Critic
from src.Model.Actor import Actor
from src.Model.ActorCritic import ActorCritic
from src.NetworkEnv import NetworkEnv
from src.PrioritizedReplay import PrioritizedReplayMemory


class A2C_Agent_Reinforce():
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
        # self.actor_critic_model = ActorCritic(self.device, self.num_actions).to(self.device)
        # self.optimizer = optim.Adam(self.actor_critic_model.parameters(), self.lr)

        self.actor_model = Actor(self.device, self.num_actions).to(self.device)
        self.critic_model = Critic(self.device, self.num_actions).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), self.lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.lr)

        self.env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.can_do_more_then_one_loop)

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
        action_to_compression = StaticConf.getInstance().conf_values.action_to_compression_rate

        while self.episode_idx < (len(self.env.all_networks) * 10) or (not reward_not_improving):
            print("Episode {}/{}".format(self.episode_idx, self.num_episodes))
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            done = False

            # rollout trajectory
            while not done:
                value = self.critic_model(state)
                dist = self.actor_model(state)

                action = dist.sample()
                compression_rate = action_to_compression[action.cpu().numpy()[0]]
                next_state, reward, done = self.env.step(compression_rate)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))

                state = next_state
                # self.episode_idx += 1

                if done:
                    break

                # dist = self.actor_model(state)
                # action = dist.sample()

                # dist, value_cur_s = self.actor_critic_model(state)

                # value_cur_s = self.critic_model(state)
                # probs = dist._param.detach().numpy()[0]
                # action = np.random.choice(len(self.action_to_compression), 1, p=probs)[0]
                # action = dist.sample()
                # compression_rate = self.action_to_compression[action.cpu().numpy()[0]]
                # next_state, reward, done = self.env.step(compression_rate)

                # if done:
                #     value_next_s = 0
                # else:
                #     # _, value_next_s = self.actor_critic_model(next_state)
                #     value_next_s = self.critic_model(next_state)

                # target = reward + self.discount_factor * value_next_s
                # rewards.append(reward)
                # advantage = target - value_cur_s
                # critic_loss = advantage.pow(2)
                # loss_val = critic_loss.data.detach().cpu().numpy().min()
                # writer.add_scalar('Critic Loss', loss_val, frame_idx)

                # actor_loss = -dist.log_prob(torch.Tensor([action])) * advantage.detach()
                # loss_val = actor_loss.data.detach().cpu().numpy().min()
                # writer.add_scalar('Actor Loss', loss_val, frame_idx)

                # self.critic_optimizer.zero_grad()
                # critic_loss.backward()
                # self.critic_optimizer.step()


                # self.actor_optimizer.zero_grad()
                # actor_loss.backward()
                # self.actor_optimizer.step()

                # loss = actor_loss + critic_loss
                # loss_val = loss.data.detach().cpu().numpy().min()
                # writer.add_scalar('Loss', loss_val, frame_idx)

                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

                # state = next_state
                # frame_idx += 1
                #
                # if done:
                #     break

                # entropy += dist.entropy().mean()
                #
                # log_probs.append(log_prob)
                # values.append(value_cur_s)
                # rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                # masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))
                #
                # state = next_state
                # frame_idx += 1
                #
                # if done:
                #     break

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

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            all_rewards_episodes.append(returns[-1])
            curr_reward = all_rewards_episodes[-1]

            if max_reward_in_all_episodes < v(curr_reward):
                max_reward_in_all_episodes = v(curr_reward)



            if len(all_rewards_episodes) > 20 and max_reward_in_all_episodes >= max(all_rewards_episodes[-20:]):
                reward_not_improving = True

def v(a):
    return a.data.detach().cpu().numpy().min()