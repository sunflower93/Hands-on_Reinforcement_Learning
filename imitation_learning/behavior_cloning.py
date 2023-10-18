import torch.optim

from PPO.PPO import PolicyNet


class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimitizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions.type(torch.int64)))
        bc_loss = torch.mean(-log_probs)

        self.optimitizer.zero_grad()
        bc_loss.backward()
        self.optimitizer.step()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
