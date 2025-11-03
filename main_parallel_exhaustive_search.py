"""
@author: Ning zhe Shi
@time: 2024年3月22日
"""
import numpy as np
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import power_allocation_cvx as pa_cvx
import project_backend as pb
import os
import itertools
from tqdm import tqdm
import sys

def process_allocation(params):
    alloc, t, H_all_2, task_matrix, cpu_matrix, cell_mapping, u, n, m, B, Pmax, cpu_cycles, T_p, NOMA, dropped, W, max_iter, epsilon, T_requirement_matrix, weight = params
    # 将alloc从元组转换为期望的格式
    channel_current = np.array(alloc)
    '''
    computing resource allocation
    '''
    t_m_k_c_current, x_current = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t], task_matrix[:, t],
                                                                     cpu_cycles, cell_mapping[:, t],
                                                                     channel_current)
    '''
    power allocation
    '''
    p_initial = Pmax * np.ones((u, 1))  # 初始化为最大功率
    # 根据当前遍历的信道和当前遍历信道对应最优的计算资源分配情况下,计算总效用值
    latency_current,_ = pb.user_uplink_time(H_all_2[t], channel_current, p_initial, task_matrix[:, t], u, W,
                                          m, cell_mapping[:, t], T_p, NOMA,
                                          dropped, weight)
    latency_current = latency_current.flatten() + t_m_k_c_current
    sum_utility_previous = np.sum(
        pb.penalize_service_failures_and_drops(latency_current, T_requirement_matrix[:, t], channel_current,
                                               T_p, weight))
    sum_utility_current, p_current, power_no_sloved,_ = pa_cvx.power_allocation_cvx(max_iter,
                                                                                  p_initial,
                                                                                  channel_current,
                                                                                  H_all_2[t],
                                                                                  u, n, m,
                                                                                  cell_mapping[
                                                                                  :, t],
                                                                                  Pmax,
                                                                                  epsilon,
                                                                                  sum_utility_previous,
                                                                                  t_m_k_c_current,
                                                                                  T_requirement_matrix[:,t],
                                                                                  T_p,
                                                                                  task_matrix[
                                                                                  :, t],
                                                                                  W,
                                                                                  NOMA,
                                                                                  dropped,weight)
    if power_no_sloved == 0 or sum_utility_previous >= sum_utility_current:  # 如果无解或者没有更好的解
        sum_utility_current, p_current = sum_utility_previous, p_initial

    return sum_utility_current, channel_current, p_current, x_current


def is_valid_allocation_vectorized(alloc, cell_mapping, z_m, n):
    # 假设alloc是一个表示信道分配的数组，cell_mapping是每个用户对应的小区编号
    num_users = len(alloc)
    num_cells = np.max(cell_mapping) + 1  # 假设cell_mapping从0开始

    # 创建一个矩阵，用于统计每个小区每个信道上的用户数
    cell_channel_count = np.zeros((num_cells, n + 1), dtype=int)

    for user in range(num_users):
        cell = cell_mapping[user]
        channel = alloc[user]
        cell_channel_count[cell, channel] += 1

    # 检查是否有任何小区的任何信道上的用户数超过z_m
    return np.all(cell_channel_count[:, 1:] <= z_m)  # 排除了未分配信道的情况


def main(args):
    json_file = args.json_file
    print(json_file)

    with open('./config/deployment/' + json_file + '.json', 'r') as f:
        options = json.load(f)

    #  Number of samples
    total_samples = options['simulation']['total_samples']
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
    NIND = args.NIND
    NOMA = args.NOMA
    dropped = args.drop
    z_n = args.z_n
    beta = args.beta
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

    user_channel_opt_all = []
    x_opt_all = []
    p_opt_all = []

    Utility_m_k_uplink_c_matrix_copy = 0  # 用于运行过程中显示截至第t4个时隙的效用值(端到端时延)结果的中间变量
    num_sum_not_requirement = 0  # 用于运行过程中显示截至第t个时隙的失败结果的中间变量
    num_sum_not_service = 0  # 用于运行过程中显示截至第t个时隙的未服务用户数的中间变量
    num_sum_service_but_failure = 0  # 用于运行过程中显示截至第t个时隙的服务但是失败的用户数中间变量
    weighted_initial = np.ones((u,1))  # 初始化用户权重为1
    for t in range(total_samples):  # 对于整个total_samples时间轴上的数据分帧进行处理
        if t == 0:
            weighted = weighted_initial
        else:
            weighted = weighted_previous
        print(f"##########################第{t}帧#################################")
        print(f"{json_file}_B{args.B / 1e6}_Pmax{args.P_max}_Cpu{args.cpu_cycles / 1e9}_Tp{args.T_p}_z_n{args.z_n}_NIND{args.NIND}_NOMA{args.NOMA}_drop{args.drop}_beta{args.beta}.npz")
        '''
        joint communication and computing allocation:including computing_resource、power and sub-channel
        其中穷搜法解决信道分配/丢弃
        '''
        '''
        channel allocation
        '''
        # 根据条件选择范围
        # 根据drop的值动态设置范围
        # dropped = True
        if dropped:
            start = 0  # 如果可以丢弃，信道范围包含0
        else:
            start = 1  # 否则从1开始
        # 使用动态范围生成所有可能的分配
        allocs = list(itertools.product(range(start, n + 1), repeat=u))
        valid_allocs = [alloc for alloc in allocs if is_valid_allocation_vectorized(alloc, cell_mapping[:, t], z_n, n)]
        task_params = [(alloc, t, H_all_2, task_matrix, cpu_matrix, cell_mapping, u, n, m, B, Pmax, cpu_cycles, T_p,
                        NOMA, dropped, W, max_iter, epsilon, T_requirement_matrix, weighted) for alloc in valid_allocs]
        # allocs = list(itertools.product(range(start, n + 1), repeat=u))  # 因为range不包括右端的n+1
        # task_params = [(alloc, t, H_all_2, task_matrix, cpu_matrix, cell_mapping, u, n, m, B, Pmax, cpu_cycles, T_p,
        #                 NOMA, dropped, W, max_iter, epsilon, T_requirement_matrix) for alloc in allocs]

        optimal_result = (-np.inf, None, None, None)  # 初始化为无穷大的延迟，表示还未找到最优解
        found_valid_result = False  # 增加一个标志来检测是否找到了有效的结果
        # 使用ProcessPoolExecutor并行处理，这里使用tqdm显示进度
        with ProcessPoolExecutor() as executor:
            # 使用tqdm包装任务参数，以显示进度
            futures = [executor.submit(process_allocation, param) for param in task_params]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing frame {t}"):
                potential_result = f.result()
                if potential_result[0] > optimal_result[0]:
                    optimal_result = potential_result
                    found_valid_result = True  # 更新标志，表示找到了至少一个有效的结果

        # 检查是否有任何有效的结果
        if not found_valid_result:
            # 如果没有找到有效的结果，则警告并提前停止
            print("Warning: No valid results were found. Stopping early.")
            sys.exit()  # 调用sys.exit()来终止程序

        optimal_channel_alloc = optimal_result[1]
        optimal_power = optimal_result[2]
        optimal_compute = optimal_result[3]
        # 每个帧的上传时间、计算时间、总时间以及最优资源分配数据如下
        T_m_k_uplink,_ = pb.user_uplink_time(H_all_2[t], optimal_channel_alloc, optimal_power,
                                                        task_matrix[:, t], u,
                                                        W, m, cell_mapping[:, t], T_p, NOMA, dropped, weighted)
        T_m_k_uplink_matrix[:, t] = T_m_k_uplink.flatten()
        T_m_k_c_matrix[:, t], _ = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t], task_matrix[:, t],
                                                                      cpu_cycles, cell_mapping[:, t],
                                                                      optimal_channel_alloc)
        user_channel_opt_all.append(optimal_channel_alloc)
        x_opt_all.append(optimal_compute)
        p_opt_all.append(optimal_power)
        print(f"##########################截至第{t}时隙的平均端到端时延#################################")
        T_m_k_uplink_c_matrix_t = T_m_k_uplink_matrix[:, t] + T_m_k_c_matrix[:, t]  # 中间变量，为了消除20ms以上的时延的影响
        Utility_m_k_uplink_c_matrix_t, num_service_but_failure, num_not_service, num_not_requirement,_ = pb.analyze_and_penalize(
            T_m_k_uplink_c_matrix_t, T_requirement_matrix[:, t], optimal_channel_alloc, T_p, weighted)
        # Accumulate counts
        num_sum_service_but_failure += num_service_but_failure
        num_sum_not_service += num_not_service
        num_sum_not_requirement += num_not_requirement
        Utility_m_k_uplink_c_matrix_copy += np.mean(Utility_m_k_uplink_c_matrix_t)
        print(f"平均效用值为: {Utility_m_k_uplink_c_matrix_copy / (t + 1):.4f}")
        print(f"丢弃用户所占比例: {num_sum_not_service / (t + 1) / u:.2%}")
        print(f"服务但失败的用户所占比例: {num_sum_service_but_failure / (t + 1) / u:.2%}")
        print(f"总未满足要求的用户所占比例(包括丢弃和服务失败): {num_sum_not_requirement / (t + 1) / u:.2%}")
        # 多域资源分配结束
        # 更新weighted
        weighted_previous = pb.update_weights(user_channel_opt_all,beta)
        print("权重",weighted_previous)
        print(user_channel_opt_all)
    save_path = os.path.join(args.output_dir,
                             f"{json_file}_B{args.B / 1e6}_Pmax{args.P_max}_Cpu{args.cpu_cycles / 1e9}_Tp{args.T_p}_z_n{args.z_n}_NIND{args.NIND}_NOMA{args.NOMA}_drop{args.drop}_beta{args.beta}.npz")
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(save_path,
             user_channel_opt_all=user_channel_opt_all,
             p_opt_all=p_opt_all,
             x_opt_all=x_opt_all,
             T_m_k_uplink_matrix=T_m_k_uplink_matrix,
             T_m_k_c_matrix=T_m_k_c_matrix,
             )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='give test scenarios.')
    parser.add_argument('--json-file', type=str,
                        default='M3_U12_N2_Task_u15000_Ucpu_e1000_Tdemand_u15_slot1',
                        help='json file for the deployment')
    parser.add_argument("--output_dir", default="exhaustive_search_output", type=str, help="Output directory.")
    # 新添加的参数
    parser.add_argument("--B", type=float, default=2e7, help="System bandwidth.")
    parser.add_argument("--P_max", type=int, default=23, help="Maximum transmission power.")
    parser.add_argument("--cpu_cycles", type=float, default=1.5e10, help="CPU cycles.")
    parser.add_argument("--T_p", type=float, default=0.01, help="penalty constant.")
    parser.add_argument("--z_n", type=int, default=2, help="max channel number.")
    parser.add_argument("--NIND", type=int, default=500, help="Genetic algorithm population size.")
    parser.add_argument("--NOMA", action="store_true", help="Enable NOMA if specified.")
    parser.add_argument("--drop", action="store_true", help="Enable drop if specified.")
    parser.add_argument("--beta", type=float, default=1, help="penalty hyperparameter.")
    args = parser.parse_args()
    main(args)
