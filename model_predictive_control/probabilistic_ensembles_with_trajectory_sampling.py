import numpy as np

from model_predictive_control.EnsembleDynamicsModel import EnsembleDynamicsModel
from model_predictive_control.FakeEnv import FakeEnv
from model_predictive_control.cross_entropy_method import CEM


class PETS:
    '''PETS算法'''
    def __init__(self, env, repaly_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes):
        self._env = env
        self._env_pool = repaly_buffer

        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim)
        self._fake_env = FakeEnv(self._model)
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        self._cem = CEM(n_sequence, elite_ratio, self._fake_env, self.upper_bound, self.lower_bound)
        self.plan_horizon = plan_horizon
        self.num_episodes = num_episodes

    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = env_samples[0]
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2])
        next_obs = env_samples[3]
        inputs = np.concatenate((obs, actions), axis=-1)
        labels = np.concatenate((rewards, next_obs - obs), axis=-1)
        self._model.train(inputs, labels)

    def mpc(self):
        mean = np.tile((self.upper_bound + self.lower_bound) / 2.0, self.plan_horizon)
        var = np.tile(np.square(self.upper_bound - self.lower_bound) / 16, self.plan_horizon)
        obs, done, episode_return = self._env.reset()[0], False, 0
        while not done:
            actions = self._cem.optimize(obs, mean, var)
            action = actions[:self._action_dim]  # 选取第一个动作
            next_obs, reward, done, _, _ = self._env.step(action)
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            mean = np.concatenate([np.copy(actions)[self._action_dim:], np.zeros(self._action_dim)])
        return episode_return

    def explore(self):
        obs, done, episode_return = self._env.reset()[0], False, 0
        while not done:
            action = self._env.action_space.sample()
            next_obs, reward, done, _, _ = self._env.step(action)
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
        return episode_return

    def train(self):
        return_list = []
        explore_return = self.explore()  # 先进行随机策略的探索来收集一条序列的数据
        print('episode: 1, return: %d' % explore_return)
        return_list.append(explore_return)

        for i_episode in range(self.num_episodes - 1):
            self.train_model()
            episode_return = self.mpc()
            return_list.append(episode_return)
            print('episode: %d, return: %d' % (i_episode + 2, episode_return))
        return return_list