import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape, mean=None, std=None):
        if mean is not None and std is not None:
            # 测试阶段直接使用提供的 mean 和 std
            self.running_ms = None
            self.mean = np.array(mean)
            self.std = np.array(std)
        else:
            # 训练阶段动态计算 mean 和 std
            self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if self.running_ms is not None:  # 如果在训练阶段
            if update:
                self.running_ms.update(x)
            x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        else:  # 如果在测试阶段，直接使用保存的 mean 和 std
            x = (x - self.mean) / (self.std + 1e-8)
        return x


# class Normalization:
#     def __init__(self, shape):
#         self.running_ms = RunningMeanStd(shape=shape)

#     def __call__(self, x, update=True):
#         # Whether to update the mean and std,during the evaluating,update=False
#         if update:
#             self.running_ms.update(x)
#         x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

#         return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
