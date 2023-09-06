from multi_armed_bandit.Solver import Solver
class EpsilonGreedy(Solver):
    """epsilon贪婪算法，继承Solver"""
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
