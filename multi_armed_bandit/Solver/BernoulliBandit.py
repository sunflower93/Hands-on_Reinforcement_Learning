import numpy as np

class BernoulliBandit:
    """伯努利多臂老虎机，输入K表示拉杆个数"""
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)

        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


