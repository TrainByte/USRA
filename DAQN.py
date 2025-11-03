import random
import gym
import numpy as np
import collections
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

# # Trick 8: orthogonal initialization
# def orthogonal_init(layer, gain=1.0):
#     nn.init.orthogonal_(layer.weight, gain=gain)
#     nn.init.constant_(layer.bias, 0)

class AttentionLayer(nn.Module):
    """简化版的自注意力层，用于增强特征表达"""
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.attn_weights = None

        # orthogonal_init(self.query)
        # orthogonal_init(self.key)
        # orthogonal_init(self.value)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力权重
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(Q.size(-1))  # QK^T / sqrt(d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        self.attn_weights = attn_weights
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        return output
    
class Qnet(nn.Module):
    ''' Q网络，结合了Attention机制 '''
    def __init__(self, state_dim, hidden_dim, action_dim, attention_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.attn = AttentionLayer(hidden_dim, attention_dim)  # Attention层
        self.fc2 = nn.Linear(attention_dim, action_dim)  # 修改为 attention_dim 作为输入维度

        # orthogonal_init(self.fc1)
        # orthogonal_init(self.fc2, gain=0.01)
        
    def forward(self, x):
        # 将输入状态x通过隐藏层，得到hidden_dim维度的表示
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        
        # 需要调整x的形状，添加一个维度以适应注意力层
        x = x.unsqueeze(1)  # 增加一个维度，变为(batch_size, 1, hidden_dim)

        attn_output = self.attn(x)  # Attention层的输出

        # 将注意力输出送入最终的全连接层，生成动作的Q值
        output = self.fc2(attn_output.squeeze(1))  # 去掉序列维度，变为(batch_size, action_dim)
        return output


class DQNWithAttention:
    ''' DQN算法，加入Attention机制 '''
    def __init__(self, state_dim, hidden_dim, action_dim,learning_rate, gamma,
                 epsilon, target_update, device, attention_dim = 8):
        
        self.lr = learning_rate
        # self.max_train_steps = int(5e5)
    
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim, attention_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim, attention_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict,total_steps):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        actions = actions.long()
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    #     # Trick 6:learning rate Decay
    #     self.lr_decay(total_steps)

    # def lr_decay(self, total_steps):
    #     lr_now = self.lr * (1 - total_steps / self.max_train_steps)
    #     for p in self.optimizer.param_groups:
    #         p['lr'] = lr_now

    def save_model(self, save_path):
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
        }, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {load_path}")
