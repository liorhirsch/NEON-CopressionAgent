import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


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
from src.Model.ActorCritic import ActorCritic
from src.NetworkEnv import NetworkEnv


class A2C_Agent():

    def __init__(self, models_path):
        # Hyper params:
        self.lr = 1e-3
        self.num_steps = 10
        self.device = StaticConf.getInstance().conf_values.device
        self.num_actions = StaticConf.getInstance().conf_values.num_actions
        self.num_episodes = 200
        self.episode_idx = 0
        self.actor_critic_model = ActorCritic(self.device, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_model.parameters())
        self.env = NetworkEnv(models_path)
        self.action_to_compression = {
            0: 0.9,
            1:0.75,
            2: 0.6,
            3:0.4
        }

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def train(self):
        writer = SummaryWriter()

        while self.episode_idx < self.num_episodes:
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            # rollout trajectory
            for _ in range(self.num_steps):
                # state = torch.FloatTensor(state).to(device)
                dist, value = self.actor_critic_model(state)

                action = dist.sample()
                compression_rate  = self.action_to_compression[action.cpu().numpy()[0]]
                next_state, reward, done = self.env.step(compression_rate)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))

                state = next_state
                self.episode_idx += 1

                if done:
                    break

            # next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.actor_critic_model(next_state)
            returns = self.compute_returns(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            loss_val = loss.data.detach().cpu().numpy().min()
            writer.add_scalar('Loss', loss_val, self.episode_idx)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
