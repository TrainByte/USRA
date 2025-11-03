import torch
import random
import collections
import numpy as np

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)
    

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.frame = 1

    def add(self, state, action, reward, next_state, done, td_error=1.0):
        # Set initial priority to a high value to ensure new samples are prioritized
        max_priority = max(self.priorities.max(), 10.0) if self.buffer else 10.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities[self.position] = max(max_priority, td_error)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Adjust beta with exponential decay to improve sample weights over time
        self.beta = min(1.0, self.beta_start * (1.0 + (self.frame / self.beta_frames)))
        self.frame += 1
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones,
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, batch_indices, batch_td_errors):
        # Apply small epsilon to prevent zero priority issues
        for idx, td_error in zip(batch_indices, batch_td_errors):
            self.priorities[idx] = td_error + 1e-5

    def size(self):
        return len(self.buffer)