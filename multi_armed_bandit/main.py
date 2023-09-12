# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from multi_armed_bandit.Solver.BernoulliBandit import BernoulliBandit
from multi_armed_bandit.Solver.DecayingEpsilonGreedy import DecayingEpsilonGreedy
from multi_armed_bandit.Solver.EpsilonGreedy import EpsilonGreedy
from multi_armed_bandit.Solver.UCB import UCB
from multi_armed_bandit.Solver.ThompsonSampling import ThompsonSampling


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label = solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))


#epsilon取0.01时情况
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累计懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ['psilonGreedy'])


#epsilon取不同值时对比
np.random.seed(0)
epsilon = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilon]
epsilon_greedy_solver_names = ['epsilon={}'.format(e) for e in epsilon]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


#采用衰减的epsilon
np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累计懊悔值为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ['DecayingEpsilonGreedy'])


#采用置信上界算法
np.random.seed(1)
coef = 1
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累计懊悔值为：', UCB_solver.regret)
plot_results([UCB_solver], ['UCB'])


#采用汤普森采样
np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累计懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ['ThompsonSampling'])