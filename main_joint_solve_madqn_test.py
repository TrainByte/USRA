"""
@author: Ning zhe Shi
@time: 2025年4月16日
根据审稿人意见补充DRL对比算法,测试脚本
"""
import sys
import numpy as np
import geatpy as ea
import argparse
import json
from SEGA_KKT_Drop import MyProblem_Drop
from normalization import Normalization, RewardScaling
import power_allocation_cvx as pa_cvx
import project_backend as pb
import os
import US_env
from DQN import *
import collections
import time
import torch
import random

def detect_channel_changes(channel_previous, channel_current):
    """
    检测并返回信道分配发生变化的用户索引列表。

    :param channel_previous: 上一轮迭代的信道分配，NumPy 数组格式。
    :param channel_current: 当前轮迭代的信道分配，NumPy 数组格式。
    :return: 一个包含变化用户索引的列表。如果没有变化，则返回空列表。
    """
    # 将信道分配转换为二元状态（1表示分配了信道，0表示未分配或被丢弃）
    binary_previous = np.where(channel_previous > 0, 1, 0)
    binary_current = np.where(channel_current > 0, 1, 0)

    # 检查信道分配变化
    changed_users = np.where(binary_previous != binary_current)[0]

    return changed_users.tolist()


def calculate_user_externality(channel_opt, x_opt, p_opt, H_all_2, task_matrix, u, W, m, cell_mapping, T_p, NOMA, dropped, weighted, cpu_matrix, cpu_cycles, T_requirement_matrix, t,Utility_m_k_matrix_t):
    num_users = len(channel_opt)
    rewards = []

    for current_user in range(num_users):
        total_externality = 0
        if channel_opt[current_user] == 0:
            rewards.append(Utility_m_k_matrix_t[current_user])
            continue

        # 找出同信道的其他用户
        same_channel_users = [i for i in range(num_users) if i != current_user and channel_opt[i] == channel_opt[current_user]]
        if not same_channel_users:
            rewards.append(Utility_m_k_matrix_t[current_user])
            continue

        # 复制当前的信道分配、资源分配等数据
        new_channel_opt = channel_opt.copy()
        new_x_opt = x_opt.copy()
        new_p_opt = p_opt.copy()

        # 将当前用户置为 0
        new_channel_opt[current_user] = 0
        # new_x_opt[current_user] = 0
        # new_p_opt[current_user] = 0

        # 重新计算上传时间和计算时间
        T_m_k_uplink_matrix = np.zeros((num_users, len(H_all_2)))
        T_m_k_c_matrix = np.zeros((num_users, len(H_all_2)))
        
        T_m_k_uplink,_ = pb.user_uplink_time(H_all_2[t], new_channel_opt, new_p_opt, task_matrix[:, t], u,
                                                        W, m, cell_mapping[:, t], T_p, NOMA, dropped, weighted)
        T_m_k_uplink_matrix[:, t]= T_m_k_uplink.flatten()
        T_m_k_c_matrix[:, t], _ = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t], task_matrix[:, t],
                                                                         cpu_cycles, cell_mapping[:, t],
                                                                         new_channel_opt)

        # 计算新的端到端时延
        T_m_k_uplink_c_matrix_t = T_m_k_uplink_matrix[:, t] + T_m_k_c_matrix[:, t]

        # 计算没有当前用户干扰时同信道其他用户的满意度
        without_interference_Utility_m_k_matrix_t, _, _, _,_ = pb.analyze_and_penalize(
            T_m_k_uplink_c_matrix_t, T_requirement_matrix[:, t], new_channel_opt, T_p, weighted)
        without_interference_satisfaction = np.sum(without_interference_Utility_m_k_matrix_t[same_channel_users])

        # 当前同信道其他用户的满意度
        current_Utility_m_k_matrix_t = Utility_m_k_matrix_t.copy()
        current_satisfaction = np.sum(current_Utility_m_k_matrix_t[same_channel_users])

        # 计算用户外部性
        user_externality = without_interference_satisfaction - current_satisfaction
        total_externality += user_externality

        # 计算当前用户的奖励
        user_reward = current_Utility_m_k_matrix_t[current_user] - total_externality
        rewards.append(user_reward)

    return rewards

def main(args):
    if not args.output_dir:
        print("Error: Output directory is not specified.")
        sys.exit(1)
    json_file_trained = args.json_file_trained
    json_file = args.json_file
    json_file_policy = args.json_file_policy
    print(json_file)

    with open('./config/deployment/' + json_file + '.json', 'r') as f:
        options = json.load(f)
    
    with open ('./config/policy/'+json_file_policy+'.json','r') as f:
        options_policy = json.load(f)

    #  Number of samples
    total_samples = options['simulation']['total_samples']
    # simulation parameters
    train_episodes = options['train_episodes']
    u = options['simulation']['U']  
    m = options['simulation']['M']
    B = args.B
    n = options['simulation']['N']
    P_max_dBm = args.P_max
    noise_dbm = options['simulation']['noise']
    Pmax_dB = P_max_dBm - 30
    Pmax = np.power(10.0, Pmax_dB / 10)
    n0_dB = noise_dbm - 30
    noise_var_Hz = np.power(10.0, n0_dB / 10)
    W = B / n  # 每个子频带的带宽
    noise_var = noise_var_Hz * W
    cpu_cycles = args.cpu_cycles
    T_p = args.T_p
    # NIND = args.NIND
    NOMA = args.NOMA
    dropped = args.drop
    # NOMA = True
    # dropped = True
    z_n = args.z_n
    beta = args.beta
    print("drop:", dropped)
    print("NOMA:", NOMA)
    '''
    读取random_deployment中的数据
    '''
    file_path = './simulations/channel/%s_network' % (json_file)
    data = np.load(file_path + '.npz', allow_pickle=True)
    H_all = data['arr_0']
    # 获取用户小区映射关系
    mirrors = data['arr_1'].item()
    cell_mapping = mirrors['cell_mapping']
    # 获取任务大小的矩阵
    task_matrix = data['arr_2']  # 矩阵大小为(用户数*时隙数)
    # 获取每bit数据所需CPU轮数大小的矩阵
    cpu_matrix = data['arr_3']  # 矩阵大小为(用户数*时隙数)
    T_requirement_matrix = data['arr_4']  # 矩阵大小为(用户数*时隙数)

    '''
    joint communication and computing allocation:including computing_resource、power and sub-channel
    '''
    H_all_2 = []
    epsilon = 1e-4  # 收敛阈值
    max_iter = 20  # 最大迭代步数
    np.random.seed(u + n + 1000)
    # 读取信道增益并进行平方及归一化
    for t in range(total_samples):
        H_all_2.append(H_all[t] ** 2 / noise_var)  # H_all为开根号的数据，需要进行平方
        # TODO：完成信道增益矩阵和噪声功率的归一化,上行除以noise_var即做了归一化

    # 每个时隙最优功率和最优子信道的上传时间
    T_m_k_uplink_matrix = np.zeros(cpu_matrix.shape)
    T_m_k_c_matrix = np.zeros(cpu_matrix.shape)

    ppo_user_channel_opt_all = []
    ppo_x_opt_all = []
    ppo_p_opt_all = []
    # 初始化
    Utility_m_k_uplink_c_matrix_copy = 0  # 用于运行过程中显示截至第t4个时隙的效用值(端到端时延)结果的中间变量
    num_sum_not_requirement = 0  # 用于运行过程中显示截至第t个时隙的失败结果的中间变量
    num_sum_not_service = 0  # 用于运行过程中显示截至第t个时隙的未服务用户数的中间变量
    num_sum_service_but_failure = 0  # 用于运行过程中显示截至第t个时隙的服务但是失败的用户数中间变量
    weighted_initial = np.ones((u,1))  # 初始化用户权重为1
    
    # DRL 参数读取
    lr = args.lr
    hidden_dim = args.hidden_width
    gamma = args.gamma
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)

    
    # DRL 初始化内容
    policy = US_env.userselection_subchannelscheduling(options,options_policy,u,m,n,noise_var,seed=100)
    args.state_dim = policy.policynum_input
    args.action_dim = policy.policynum_actions
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent_DQN = DQN(args.state_dim,hidden_dim,args.action_dim,lr,gamma,epsilon,target_update,device)
    
    # 加载模型参数
    model_dir = os.path.join(args.output_dir, 'train', 'model')
    save_model_path = os.path.join(model_dir,
                             f"{json_file_trained}_B{args.B / 1e6}_Pmax{args.P_max}_Cpu{args.cpu_cycles / 1e9}_Tp{args.T_p}_z_n{args.z_n}_NOMA{args.NOMA}_drop{args.drop}_beta{format(args.beta,'g')}_hiddenWidth{args.hidden_width}_lr{args.lr}_gamma{args.gamma}")
    agent_DQN.load_model(save_model_path)
    
    US_strategy = np.array
    time_calculating_strategy_takes = [] # 用于决策的时间
    time_optimization_at_each_slot_takes = []
    sum_utility_list = collections.deque([],2) # 总效用值
    status_matrix_list = collections.deque([],2) # 总效用值
    rewards_list = []  # 奖励列表
    US_list = []  # US列表
    forcezero = False
    # 初始化 state_norm
    state_norm = Normalization(
    shape=(policy.policynum_input,),
    mean=agent_DQN.state_norm_mean,
    std=agent_DQN.state_norm_std
    )
    # state_norm = Normalization(shape=policy.policynum_input)  # Trick 2:state normalization
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma) # Trick 4:reward scaling

    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    for t in range(total_samples):  # 对于整个total_samples时间轴上的数据分帧进行处理
        if (t == 0):
            weighted = weighted_initial
        else:
            weighted = weighted_previous
        print(f"##########################第{t}帧#################################")
        print(
            f"{json_file}_B{args.B / 1e6}_Pmax{args.P_max}_Cpu{args.cpu_cycles / 1e9}_Tp{args.T_p}_z_n{args.z_n}_NOMA{args.NOMA}_drop{args.drop}_beta{format(args.beta,'g')}.npz")
        
        
        if (t > 0):
            for agent in range (u):
                current_state = policy.local_state(agent,H_all_2[t],cpu_matrix[:, t], task_matrix[:, t], cell_mapping[:, t],T_requirement_matrix[:, t],weighted,US_list[-1],rewards_list[-1],status_matrix_list[-1])
                current_state = state_norm(current_state)
                a_time = time.time()
                strategy = agent_DQN.evaluate(current_state)
                print(strategy)
                time_calculating_strategy_takes.append(time.time()-a_time)
                                
                # Pick the action
                US_strategy[agent] =  strategy

                # Add current state to the short term memory to observe it during the next state
                policy.previous_state[agent,:] = current_state
                policy.previous_action[agent] = strategy
        
        if(t < 1):
            US_strategy = np.array([(i % n) + 1 for i in range(u)])  # 初始化用户信道矩阵即u个用户循环选择1-n号子信道

        US_strategy_current = np.array(US_strategy) 

        # 根据动作得到当前用户选择和子信道调度方案
        channel_current = US_strategy_current

        '''
        computing resource allocation
        '''
        t_m_k_c_current, x_current = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t],
                                                                            task_matrix[:, t],
                                                                            cpu_cycles, cell_mapping[:, t],
                                                                            channel_current)  # 根据当前信道分配情况计算时延
        '''
        compute utility and decide whether to replace channel_previous and x_previous
        '''
        p_initial = Pmax * np.ones((u, 1))  # 初始化为最大功率
        p_previous = p_initial
        latency_current,_ = pb.user_uplink_time(H_all_2[t], channel_current, p_previous, task_matrix[:, t], u, W,
                                                m, cell_mapping[:, t], T_p, NOMA,
                                                dropped,weighted)
        latency_current = latency_current.flatten() + t_m_k_c_current
        sum_utility_previous = np.sum(
            pb.penalize_service_failures_and_drops(latency_current, T_requirement_matrix[:, t], channel_current,
                                                    T_p,weighted))
        '''
        power allocation
        '''
        sum_utility_current, p_current, power_no_sloved,_ = pa_cvx.power_allocation_cvx(max_iter,
                                                                                        p_previous,
                                                                                        channel_current,
                                                                                        H_all_2[t],
                                                                                        u, n, m,
                                                                                        cell_mapping[
                                                                                        :, t],
                                                                                        Pmax,
                                                                                        epsilon,
                                                                                        sum_utility_previous,
                                                                                        t_m_k_c_current,
                                                                                        T_requirement_matrix[:,
                                                                                        t],
                                                                                        T_p,
                                                                                        task_matrix[
                                                                                        :, t],
                                                                                        W,
                                                                                        NOMA,
                                                                                        dropped,weighted)

        print(f"******sum_utility_previous:{sum_utility_previous}")
        print(f"*********sum_utility_current:{sum_utility_current}")
        if power_no_sloved == 0 or sum_utility_previous >= sum_utility_current:  # 如果无解或者没有更好的解
            print('功率无更好解/解导致utility未更新')
            sum_utility_current, p_current = sum_utility_previous, p_previous
        else:
            sum_utility_previous, p_previous = sum_utility_current, p_current



        channel_opt = channel_current
        x_opt = x_current
        p_opt = p_current
        # 每个帧的上传时间、计算时间、总时间以及最优资源分配数据如下
        T_m_k_uplink,_ = pb.user_uplink_time(H_all_2[t], channel_opt, p_opt, task_matrix[:, t], u,
                                                        W, m, cell_mapping[:, t], T_p, NOMA, dropped,weighted)
        T_m_k_uplink_matrix[:, t]= T_m_k_uplink.flatten()

        T_m_k_c_matrix[:, t], _ = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t], task_matrix[:, t],
                                                                         cpu_cycles, cell_mapping[:, t],
                                                                         channel_opt)
        ppo_user_channel_opt_all.append(channel_opt)
        ppo_x_opt_all.append(x_opt)
        ppo_p_opt_all.append(p_opt)
        print(f"##########################截至第{t}时隙的平均端到端时延#################################")
        T_m_k_uplink_c_matrix_t = T_m_k_uplink_matrix[:, t] + T_m_k_c_matrix[:, t]  # 中间变量，为了消除20ms以上的时延的影响
        Utility_m_k_matrix_t, num_service_but_failure, num_not_service, num_not_requirement,status_matrix = pb.analyze_and_penalize(
            T_m_k_uplink_c_matrix_t, T_requirement_matrix[:, t], channel_opt, T_p, weighted)
        # # 计算外部性 (计算与当前用户在同一个子信道的用户如果没有当前用户干扰时的满意度,然后减去当前的满意度得到一个用户外部性,对当前信道下的用户外部性求和得到总
        # # 外部性,然后用当前用户的满意度减去总外部性得到当前用户的reward,子信道为0的用户没有外部性)
        # rewards = calculate_user_externality(channel_opt, x_opt, p_opt, H_all_2, task_matrix, u, W, m, cell_mapping, T_p, NOMA, dropped, weighted, cpu_matrix, cpu_cycles, T_requirement_matrix, t,Utility_m_k_matrix_t)
        # sum_utility_list.append(np.array(rewards))
        status_matrix_list.append(status_matrix)
        rewards_list.append(Utility_m_k_matrix_t*1000)  # 不管是丢弃还是服务失败都是惩罚，这是DRL的reward
        US_list.append(channel_opt)
        # Accumulate counts
        num_sum_service_but_failure += num_service_but_failure
        num_sum_not_service += num_not_service
        num_sum_not_requirement += num_not_requirement
        Utility_m_k_uplink_c_matrix_copy += np.mean(Utility_m_k_matrix_t)
        print(f"瞬时当前平均效用值为: {np.mean(Utility_m_k_matrix_t)*1000:.4f}")
        print(f"最近1000个平均效用值为: {np.mean(rewards_list[-1000:]):.4f}")
        print(f"丢弃用户所占比例: {num_not_service / u:.2%}")
        print(f"服务但失败的用户所占比例: {num_service_but_failure / u:.2%}")
        print(f"平均效用值为: {Utility_m_k_uplink_c_matrix_copy / (t + 1):.4f}")
        print(f"丢弃用户所占比例: {num_sum_not_service / (t + 1) / u:.2%}")
        print(f"服务但失败的用户所占比例: {num_sum_service_but_failure / (t + 1) / u:.2%}")
        print(f"总未满足要求的用户所占比例(包括丢弃和服务失败): {num_sum_not_requirement / (t + 1) / u:.2%}")
        # 该时隙多域资源分配结束
        # 更新weighted
        weighted_previous = pb.update_weights(ppo_user_channel_opt_all,beta)
        # print(f"weighted_previous:{weighted_previous}")
        # print("权重",weighted_previous)
        # print(ppo_user_channel_opt_all)
    
    # 创建 train/data 目录，如果不存在的话
    data_dir = os.path.join(args.output_dir, 'test', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)   
    save_data_path = os.path.join(data_dir,
                             f"{json_file}_B{args.B / 1e6}_Pmax{args.P_max}_Cpu{args.cpu_cycles / 1e9}_Tp{args.T_p}_z_n{args.z_n}_NOMA{args.NOMA}_drop{args.drop}_beta{format(args.beta,'g')}_hiddenWidth{args.hidden_width}_lr{args.lr}_gamma{args.gamma}.npz")
    # os.makedirs(data_dir, exist_ok=True)
    np.savez(save_data_path,
             user_channel_opt_all=ppo_user_channel_opt_all,
             p_opt_all=ppo_p_opt_all,
             x_opt_all=ppo_x_opt_all,
             T_m_k_uplink_matrix=T_m_k_uplink_matrix,
             T_m_k_c_matrix=T_m_k_c_matrix,
             T_requirement_matrix=T_requirement_matrix,
             sum_utility_list=sum_utility_list,  # 当看收敛性能时存储
             rewards_list=rewards_list
             )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='give train scenarios.')
    parser.add_argument('--json-file-trained', type=str,
                        default='train_M7_U42_N5_Task_u15000_Ucpu_e1000_Tdemand_u15',
                        help='json file for the deployment')
    parser.add_argument('--json-file', type=str,
                        default='M7_U42_N5_Task_u15000_Ucpu_e1000_Tdemand_u15',
                        help='json file for the deployment')
    parser.add_argument('--json-file-policy', type=str, default='ppo',
                       help='json file for the hyperparameters, including hsac and dsac')
    parser.add_argument("--output_dir", default="madqn_output", type=str, help="Output directory.")
    # 新添加的参数
    parser.add_argument("--B", type=float, default=2e7, help="System bandwidth.")
    parser.add_argument("--P_max", type=int, default=23, help="Maximum transmission power.")
    parser.add_argument("--cpu_cycles", type=float, default=1.5e10, help="CPU cycles.")
    parser.add_argument("--T_p", type=float, default=0.01, help="penalty constant.")
    parser.add_argument("--z_n", type=int, default=2, help="max channel number.")
    # parser.add_argument("--NIND", type=int, default=500, help="Genetic algorithm population size.")
    # parser.add_argument("--NOMA", action="store_false", default=True, help="Disable NOMA if specified.")
    # parser.add_argument("--drop", action="store_false", default=True, help="Disable drop if specified.")
    parser.add_argument("--NOMA", action="store_true", help="Enable NOMA if specified.")
    parser.add_argument("--drop", action="store_true", help="Enable drop if specified.")
    parser.add_argument("--beta", type=float, default=1, help="penalty hyperparameter.")
    
    # PPO specific parameters
    # parser.add_argument('--lmbda', type=float, default=0.95, help='Lambda for GAE-Lambda')
    # parser.add_argument('--epochs', type=int, default=32, help='Number of epochs for training')
    # parser.add_argument('--eps', type=float, default=0.6, help='Clipping epsilon for PPO')
    # parser.add_argument('--minibatch-size', type=int, default=64, help='mini batch size')
    
    # # Network parameters (actor and critic)
    # parser.add_argument('--actor-lr', type=float, default=1e-4, help='Learning rate for the actor network')
    # parser.add_argument('--critic-lr', type=float, default=1e-3, help='Learning rate for the critic network')
    # parser.add_argument('--hidden-dim', type=int, default=128, help='Number of neurons in each layer of the network')
    # parser.add_argument('--gamma', type=float, default=0.2, help='Discount factor for future rewards')
    parser.add_argument("--max_train_steps", type=int, default=int(5e5), help=" Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=2142, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate of dqn")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.1, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    args = parser.parse_args()
    main(args)
