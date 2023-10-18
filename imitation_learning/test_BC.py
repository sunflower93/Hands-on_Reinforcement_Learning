import random

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import rl_utils
from PPO.PPO import PPO
from imitation_learning.behavior_cloning import BehaviorClone

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 100
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
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)

def sample_expert_data(n_episode):
    states = []
    actions = []
    for episode in range(n_episode):
        state = env.reset()[0]
        done = False
        while not done:
            action = ppo_agent.take_action(state)
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

# print(expert_s)
# print(expert_a)

def test_agent(agent, env, n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.array(return_list)

env.reset(seed=0)
torch.manual_seed(0)
np.random.seed(0)

lr = 1e-3
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)
n_iterations = 1000
batch_size = 64
test_returns = []

with tqdm(total = n_iterations, desc="进度条") as pbar:
    for i in range(n_iterations):
        sample_indices = np.random.randint(low=0, high=expert_s.shape[0], size=batch_size)
        bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
        current_return = test_agent(bc_agent, env, 5)
        test_returns.append(current_return)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10])})
        pbar.update(1)


iteration_list = list(range(len(test_returns)))
plt.plot(iteration_list, test_returns)
plt.xlabel('Iterations')
plt.ylabel('Returns')
plt.title('BC on {}'.format(env_name))
plt.show()