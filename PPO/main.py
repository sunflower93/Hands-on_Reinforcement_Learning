import gym
import torch
from matplotlib import pyplot as plt

import rl_utils
from PPO import PPO
from PPOContinuous import PPOContinuous

# actor_lr = 1e-3
# critic_lr = 1e-2
# num_episodes = 500
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# env_name = 'CartPole-v0'
# env = gym.make(env_name)
# env.reset(seed=0)
# torch.manual_seed(0)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
#
# return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
#
# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()
#
# mv_return = rl_utils.moving_average(return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()


actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 21)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()