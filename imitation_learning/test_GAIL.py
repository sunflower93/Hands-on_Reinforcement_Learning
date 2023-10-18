import random

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import rl_utils
from PPO.PPO import PPO
from imitation_learning.generative_adversarial_imitation_learning import GAIL

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 120
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name)
env.reset(seed=0)
torch.manual_seed(0)
lr_d = 1e-3
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
gail = GAIL(agent, state_dim, hidden_dim, action_dim, lr_d, device)
n_episode = 500
return_list = []


return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

def sample_expert_data(n_episode):
    states = []
    actions = []
    for episode in range(n_episode):
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
    return np.array(states), np.array(actions)

env.reset(seed=0)
torch.manual_seed(0)
random.seed(0)
n_episode = 1
expert_s, expert_a = sample_expert_data(n_episode)

n_samples = 30  # 采样30个数据
random_index = random.sample(range(expert_s.shape[0]), n_samples)
expert_s = expert_s[random_index]
expert_a = expert_a[random_index]

with tqdm(total=n_episode, desc="进度条") as pbar:
    for i in range(n_episode):
        episode_return = 0
        state = env.reset()[0]
        done = False
        state_list = []
        action_list = []
        next_state_list = []
        done_list = []
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            done_list.append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        gail.learn(expert_s, expert_a, state_list, action_list, next_state_list, done_list)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10])})
        pbar.update(1)

iteration_list = list(range(len(return_list)))
plt.plot(iteration_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GAIL on {}'.format(env_name))
plt.show()

