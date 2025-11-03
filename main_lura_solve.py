"""
@author: Ning zhe Shi
@time: 2023年12月28日
"""
import sys
import numpy as np
import geatpy as ea
import argparse
import json
from SEGA_KKT_Drop import MyProblem_Drop
import power_allocation_cvx as pa_cvx
import project_backend as pb
import os


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


def main(args):
    if not args.output_dir:
        print("Error: Output directory is not specified.")
        sys.exit(1)
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
    user_order_matrix = data['arr_5']  # 用户到达顺序

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
    rate_uplink_matrix = np.zeros(cpu_matrix.shape)

    ceao_user_channel_opt_all = []
    ceao_x_opt_all = []
    ceao_p_opt_all = []
    ceao_selection_users_num_opt_all = []
    # 初始化
    Utility_m_k_uplink_c_matrix_copy = 0  # 用于运行过程中显示截至第t4个时隙的效用值(端到端时延)结果的中间变量
    num_sum_not_requirement = 0  # 用于运行过程中显示截至第t个时隙的失败结果的中间变量
    num_sum_not_service = 0  # 用于运行过程中显示截至第t个时隙的未服务用户数的中间变量
    num_sum_service_but_failure = 0  # 用于运行过程中显示截至第t个时隙的服务但是失败的用户数中间变量
    weighted_initial = np.ones((u,1))  # 初始化用户权重为1
    for t in range(total_samples):  # 对于整个total_samples时间轴上的数据分帧进行处理
        print(f"##########################第{t}帧#################################")
        print(
            f"{json_file}_B{args.B / 1e6}_Pmax{args.P_max}_Cpu{args.cpu_cycles / 1e9}_Tp{args.T_p}_z_n{args.z_n}_NIND{args.NIND}_NOMA{args.NOMA}_drop{args.drop}_beta{args.beta}.npz")
        low, high = 0, u
        max_selection_users_num = 0
        # 帧内通信资源分配初始化
        p_initial = Pmax * np.ones((u, 1))  # 初始化为最大功率
        channel_initial = np.array([(i % n) + 1 for i in range(u)])  # 初始化用户信道矩阵即u个用户循环选择1-n号子信道
        while low <= high:
            selection_users_num = (low + high) // 2
            print(f"尝试选择用户数量: {selection_users_num}")

            # 根据用户截至时间选择前 selection_users_num 个用户
            # 排序索引按照用户截至时间
            sorted_indices = np.argsort(T_requirement_matrix[:,t])
            selected_indices = sorted_indices[:selection_users_num]

            # 创建全零的信道矩阵
            channel_updated = np.zeros_like(channel_initial)

            # 为选择的用户设置轮询信道
            for idx, selected_index in enumerate(selected_indices):
                channel_updated[selected_index] = (idx % n) + 1
            # 打印选择后的信道矩阵
            print("Updated channel matrix:", channel_updated)
            # 信道重新初始化
            channel_initial = channel_updated
            # 帧内计算资源分配初始化
            t_m_k_c_initial, x_initial = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t], task_matrix[:, t],
                                                                                cpu_cycles, cell_mapping[:, t],
                                                                                channel_initial)  # 初始化为全调度最优分配
            
            #  开始尝试是否该用户数量下可以满足所有用户需求
            num_iter = 0
            sum_utility_list = []  # 记录总效用的列表
            if t == 0:
                weighted = weighted_initial
            else:
                weighted = weighted_previous
            '''
            joint communication and computing allocation:including computing_resource、power and sub-channel
            '''
            while num_iter < max_iter:
                '''
                打印执行情况
                '''
                print(f"#######################第{num_iter}轮####################")
                '''
                将初始化的数据列为上一个时隙结束的分配情况
                '''
                if num_iter == 0:
                    channel_previous = channel_initial
                    p_previous = p_initial
                    x_previous = x_initial
                    t_m_k_c_previous = t_m_k_c_initial
                    # latency_previous = pb.user_uplink_time(H_all_2[t], channel_previous, p_previous, task_matrix[:, t], u,W, m, cell_mapping[:, t], T_p, NOMA,dropped).flatten() + t_m_k_c_previous
                    # sum_utility_previous = np.sum(pb.penalize_service_failures_and_drops(latency_previous, T_requirement_matrix[:, t],channel_previous, T_p))
                    sum_utility_previous = -np.inf  # 不然会出现初始化有用户不满足需求，但是总效用值低，导致第一轮子信道分配虽然满足的所有约束但是效用值高，导致最后用初始化的算法，导致出现服务但失败的情况
                    sum_utility_list.append(sum_utility_previous)
                else:
                    '''
                    channel allocation
                    '''
                    problem = MyProblem_Drop(H_all_2[t], p_previous, u, n, m, cpu_cycles, cell_mapping[:, t],
                                            T_requirement_matrix[:, t],
                                            T_p, cpu_matrix[:, t], task_matrix[:, t], W, NOMA, dropped, z_n, weighted, channel_updated=channel_updated)
                    algorithm = ea.soea_SEGA_templet(problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=100,
                                                    logTras=0, trappedValue=1e-7, maxTrappedCount=10)
                    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=0, outputMsg=False, drawLog=False,
                                    saveFlag=False, dirName='result')
                    channel_current = res['Vars'][0] if res['ObjV'] is not None else channel_previous
                    # print(channel_current)
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
                    latency_current,_ = pb.user_uplink_time(H_all_2[t], channel_current, p_previous, task_matrix[:, t], u, W,
                                                        m, cell_mapping[:, t], T_p, NOMA,
                                                        dropped,weighted)
                    latency_current = latency_current.flatten() + t_m_k_c_current
                    sum_utility_current = np.sum(
                        pb.penalize_service_failures_and_drops(latency_current, T_requirement_matrix[:, t], channel_current,
                                                            T_p,weighted))
                    if sum_utility_current > sum_utility_previous:
                        sum_utility_list.append(sum_utility_current)
                        sum_utility_previous, channel_previous, x_previous, t_m_k_c_previous = sum_utility_current, channel_current, x_current, t_m_k_c_current
                    else:  # 无解
                        print('子信道无更好解:', sum_utility_current)
                        sum_utility_list.append(sum_utility_previous)
                        sum_utility_current, channel_current, x_current, t_m_k_c_current = sum_utility_previous, channel_previous, x_previous, t_m_k_c_previous
                    '''
                    power allocation
                    '''
                    # 是否每个子问题都要计算下sum_latency.我觉得需要，这样才能保证收敛。明天看论文，然后继续改代码。
                    sum_utility_current, p_current, power_no_sloved,_ = pa_cvx.power_allocation_cvx(max_iter,
                                                                                                p_previous,
                                                                                                channel_previous,
                                                                                                H_all_2[t],
                                                                                                u, n, m,
                                                                                                cell_mapping[
                                                                                                :, t],
                                                                                                Pmax,
                                                                                                epsilon,
                                                                                                sum_utility_previous,
                                                                                                t_m_k_c_previous,
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
                        sum_utility_list.append(sum_utility_previous)
                        sum_utility_current, p_current = sum_utility_previous, p_previous
                    else:
                        sum_utility_list.append(sum_utility_current)
                        sum_utility_previous, p_previous = sum_utility_current, p_current

                print(f"%%%%%%%%%sum_delay:{sum_utility_list}")
                print(f"%%%%%%%%%sum_delay[-1]:{sum_utility_list[-1]}")
                if num_iter > 0:
                    print(f"%%%%%%%%%sum_delay[-2]:{sum_utility_list[-2]}")

                if num_iter > 1 and abs(sum_utility_list[-1] - sum_utility_list[-2]) / sum_utility_list[-2] < epsilon:
                    channel_opt = channel_current
                    x_opt = x_current
                    p_opt = p_current
                    break
                num_iter += 1
            # 每个帧的上传时间、计算时间、总时间以及最优资源分配数据如下
            T_m_k_uplink,_ = pb.user_uplink_time(H_all_2[t], channel_opt, p_opt, task_matrix[:, t], u,
                                                            W, m, cell_mapping[:, t], T_p, NOMA, dropped,weighted)
            T_m_k_uplink_matrix[:, t] = T_m_k_uplink.flatten()
            T_m_k_c_matrix[:, t], _ = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t], task_matrix[:, t],
                                                                            cpu_cycles, cell_mapping[:, t],
                                                                            channel_opt)
            T_m_k_uplink_c_matrix_t = T_m_k_uplink_matrix[:, t] + T_m_k_c_matrix[:, t]  # 中间变量，为了消除20ms以上的时延的影响
            Utility_m_k_matrix_t, num_service_but_failure, num_not_service, num_not_requirement,_ = pb.analyze_and_penalize(T_m_k_uplink_c_matrix_t, T_requirement_matrix[:, t], channel_opt, T_p, weighted)
                  
            # 判断是否满足资源分配约束条件，如不满足则更新low和high
            if num_service_but_failure == 0 :
                low = selection_users_num + 1
                best_channel_opt = channel_opt
                best_x_opt = x_opt
                best_p_opt = p_opt 
                best_selection_users_num = selection_users_num
            else:
                high = selection_users_num - 1
        print("users num:",best_selection_users_num)  
        ceao_user_channel_opt_all.append(best_channel_opt)
        ceao_x_opt_all.append(best_x_opt)
        ceao_p_opt_all.append(best_p_opt)
        ceao_selection_users_num_opt_all.append(best_selection_users_num)
        print(f"##########################lura截至第{t}时隙的平均端到端时延#################################")
        # 每个帧的上传时间、计算时间、总时间以及最优资源分配数据如下
        T_m_k_uplink,rate_uplink = pb.user_uplink_time(H_all_2[t], best_channel_opt, best_p_opt, task_matrix[:, t], u,
                                                        W, m, cell_mapping[:, t], T_p, NOMA, dropped,weighted)
        T_m_k_uplink_matrix[:, t],rate_uplink_matrix[:,t] = T_m_k_uplink.flatten(),rate_uplink.flatten()
        T_m_k_c_matrix[:, t], _ = pb.optimal_compute_resource_allocation(m, cpu_matrix[:, t], task_matrix[:, t],
                                                                        cpu_cycles, cell_mapping[:, t],
                                                                        best_channel_opt)
        T_m_k_uplink_c_matrix_t = T_m_k_uplink_matrix[:, t] + T_m_k_c_matrix[:, t]  # 中间变量，为了消除20ms以上的时延的影响
        Utility_m_k_matrix_t, num_service_but_failure, num_not_service, num_not_requirement,_ = pb.analyze_and_penalize(
            T_m_k_uplink_c_matrix_t, T_requirement_matrix[:, t], best_channel_opt, T_p, weighted)
        # Accumulate counts
        num_sum_service_but_failure += num_service_but_failure
        num_sum_not_service += num_not_service
        num_sum_not_requirement += num_not_requirement
        Utility_m_k_uplink_c_matrix_copy += np.mean(Utility_m_k_matrix_t)
        print(f"平均效用值为: {Utility_m_k_uplink_c_matrix_copy / (t + 1):.4f}")
        print(f"丢弃用户所占比例: {num_sum_not_service / (t + 1) / u:.2%}")
        print(f"服务但失败的用户所占比例: {num_sum_service_but_failure / (t + 1) / u:.2%}")
        print(f"总未满足要求的用户所占比例(包括丢弃和服务失败): {num_sum_not_requirement / (t + 1) / u:.2%}")
        # 该时隙多域资源分配结束
        # 更新weighted
        weighted_previous = pb.update_weights(ceao_user_channel_opt_all,beta)
        # print("权重",weighted_previous)
        # print(ceao_user_channel_opt_all)
    save_path = os.path.join(args.output_dir,
                             f"{json_file}_B{args.B / 1e6}_Pmax{args.P_max}_Cpu{args.cpu_cycles / 1e9}_Tp{args.T_p}_z_n{args.z_n}_NIND{args.NIND}_NOMA{args.NOMA}_drop{args.drop}_beta{args.beta}.npz")
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(save_path,
             user_channel_opt_all=ceao_user_channel_opt_all,
             p_opt_all=ceao_p_opt_all,
             x_opt_all=ceao_x_opt_all,
             selection_users_num_opt_all=ceao_selection_users_num_opt_all,
             T_m_k_uplink_matrix=T_m_k_uplink_matrix,
             T_m_k_c_matrix=T_m_k_c_matrix,
             T_requirement_matrix=T_requirement_matrix,
             rate_uplink_matrix=rate_uplink_matrix,
             )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='give test scenarios.')

    parser.add_argument('--json-file', type=str,
                        default='M7_U42_N5_Task_u15000_Ucpu_e1000_Tdemand_u15_rate',
                        help='json file for the deployment')
    parser.add_argument("--output_dir", default="lura_output", type=str, help="Output directory.")
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
