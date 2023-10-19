import itertools

import numpy as np
import torch

from model_predictive_control.EnsembleModel import EnsembleModel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EnsembleDynamicsModel:
    '''环境模型集成，加入精细化的训练'''
    def __init__(self, state_dim, action_dim, num_networks=5):
        self._num_networks = num_networks
        self._state_dim, _action_dim = state_dim, action_dim
        self.model = EnsembleModel(state_dim, action_dim, ensemble_size=num_networks)
        self._epoch_since_last_update = 0

    def train(self, inputs, labels, batch_size=64, holdout_ratio=0.1, max_iter=20):
        # 设置训练集与验证集
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_networks, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_networks, 1, 1])

        # 保留最好的结果
        self._snapshots = {i: (None, 1e10) for i in range(self._num_networks)}

        for epoch in itertools.count():
            # 定义每一个网络的训练数据
            train_index = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self._num_networks)])
            # 所有真实数据都用来训练
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):
                bathc_index = train_index[:, batch_start_pos:batch_start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[bathc_index]).float().to(device)
                train_label = torch.from_numpy(train_labels[bathc_index]).float().to(device)

                mean, logvar = self.model(train_input, return_log_var=True)
                loss, _ = self.model.loss(mean, logvar, train_label)
                self.model.train(loss)

            with torch.no_grad():
                mean, logvar = self.model(holdout_inputs, return_log_var=True)
                _, holdout_losses = self.model.loss(mean, logvar, holdout_labels, use_var_loss=False)
                holdout_losses = holdout_losses.cpu()
                break_condition = self._save_best(epoch, holdout_losses)
                if break_condition or epoch > max_iter:
                    break

    def _save_best(self, epoch, losses, threshold=0.1):
        updated = False
        for i in range(len(losses)):
            current = losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > threshold:
                self._snapshots[i] = (epoch, current)
                updated = True
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5

    def predict(self, inputs, batch_size=64):
        mean, var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            cur_mean, cur_var = self.model(input[None, :, :].repeat([self._num_networks, 1, 1]), return_log_var=False)
            mean.append(cur_mean.detach().cpu().numpy())
            var.append(cur_var.detach().cpu().numpy())
        return np.hstack(mean), np.hstack(var)


















