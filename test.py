import torch

import main
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from main import init_conf_values, load_models_path
from src.Configuration.StaticConf import StaticConf
from src.Model.ActorCritic import ActorCritic
from src.NetworkEnv import NetworkEnv

init_conf_values()

device = StaticConf.getInstance().conf_values.device
num_actions = StaticConf.getInstance().conf_values.num_actions

actor_critic_model = ActorCritic(device, num_actions)
checkpoint = torch.load("./ac_model.pt")
actor_critic_model.load_state_dict(checkpoint)

models_path = load_models_path("./Fully Connected Training/")
env = NetworkEnv(models_path)

done = False
state = env.reset()

action_to_compression = {
    0: 0.9,
    1: 0.75,
    2: 0.6,
    3: 0.4
}

while not done:
    dist, value = actor_critic_model(state)
    action = dist.sample()
    compression_rate = action_to_compression[action.cpu().numpy()[0]]
    next_state, reward, done = env.step(compression_rate)
    state = next_state
