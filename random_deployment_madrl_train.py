"""
@author: Ning zhe Shi
回复审稿人意见,保持每个episode具有相同的用户分布情况及任务分布情况(每个episode里面多个(例如100)time slot是不同的),用于收敛曲线
小尺度先不管了先试试
"""

import argparse
import json
import numpy as np
import project_backend as pb
from scipy import special
import copy
import os


def main(args):
    json_file_name = args.json_file
    with open('./config/deployment/' + json_file_name + '.json', 'r') as f:
        options = json.load(f)
    m = options['simulation']['M']  # bs number
    u = options['simulation']['U']  # user number
    compute_params = options['compute_params']
    # Multi channel scenario, n denotes number of channels.
    if 'N' in options['simulation']:
        n = options['simulation']['N']
    else:
        n = 1
    equal_number_for_bs = options['simulation']['equal_number_for_BS']
    r_defined = options['simulation']['R_defined']  # Hexagonal radius (center to side plumb line)
    r = r_defined * (2.0 / np.sqrt(3))  # bs max radius (for generating range vertex coordinates)
    min_dist = options['simulation']['min_dist']  # min distance between user and bs
    total_samples = options['simulation']['total_samples']
    shadowing_dev = options['simulation']['shadowing_dev']
    dcor = options['simulation']['dcor']
    T = options['simulation']['T']
    train_episodes = options['train_episodes']
    isTrain = options['simulation']['isTrain']
    mobility_params = options['mobility_params']
    mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi  # radian/sec
    max_doppler = options['mobility_params']['max_doppler']
    v_c = options['mobility_params']['v_c']  # speed of light
    f_c = options['mobility_params']['f_c']
    # uplink_sinr = options['simulation']['uplink_sinr']
    # train_sequence_uplink = options['simulation']['train_sequence_uplink']
    # channel_error_var = 1.0 / (1 + (uplink_sinr * train_sequence_uplink))
    # channel_error_var = options['simulation']['channel_error_var']
    
    # 检查 options['simulation']['channel_error_var'] 是否存在,设置标记
    save_H_all = 'channel_error_var' in options['simulation']

    # 计算 channel_error_var
    if save_H_all:
        channel_error_var = options['simulation']['channel_error_var']
    else:
        uplink_sinr = options['simulation']['uplink_sinr']
        train_sequence_uplink = options['simulation']['train_sequence_uplink']
        channel_error_var = 1.0 / (1 + (uplink_sinr * train_sequence_uplink))

    if isTrain:
        np.random.seed(1000)
    else:
        np.random.seed(1000 + u)
    # Generate wireless environment related information
    '''
    gains 为u*u*total_samples的矩阵，存放的是信道大尺度增益值(无单位)
    '''
    if isTrain:
        gains, TX_loc, RX_loc, TX_xhex, TX_yhex, TX_neighbors, mirrors = pb.get_gains_hexagon_neighbors_shadowing(u, m,
                                                                                                                  r,
                                                                                                                  min_dist,
                                                                                                                  total_samples,
                                                                                                                  shadowing_dev,
                                                                                                                  dcor,
                                                                                                                  equal_number_for_BS=equal_number_for_bs,
                                                                                                                  draw=False,
                                                                                                                  T=T,
                                                                                                                  train_episodes=train_episodes,
                                                                                                                  mobility_params=mobility_params)
    else:
        gains = [np.zeros((u, u, total_samples))]
        RX_loc = np.zeros((2, u, total_samples))
        mirrors = {}
        mirrors['cell_mapping'] = np.zeros((u, total_samples)).astype(int)
        mirrors['RX_displacement'] = np.zeros((4, u, total_samples))
        mirrors['RX_loc_all'] = np.zeros((2, u, total_samples))
        mirrors['cell_mapping_all'] = np.zeros((u, total_samples)).astype(int)
        mirrors['RX_displacement_all'] = np.zeros((4, u, total_samples))
        tot_test_episodes = int(total_samples / train_episodes['T_train'])
        '修改如下,将pb.get_gains_hexagon_neighbors_shadowing函数从循环for ep in range(tot_test_episodes):中移出来'
        i_gains, TX_loc, i_RX_loc, TX_xhex, TX_yhex, TX_neighbors, i_mirrors = pb.get_gains_hexagon_neighbors_shadowing(
                u, m, r, min_dist, train_episodes['T_train'], shadowing_dev, dcor,
                equal_number_for_BS=equal_number_for_bs, draw=False,
                T=T,
                train_episodes=train_episodes,
                mobility_params=mobility_params)
        for ep in range(tot_test_episodes):
            cursor1 = int(ep * train_episodes['T_train'])
            cursor2 = int((ep + 1) * train_episodes['T_train'])
            gains[0][:, :, cursor1:cursor2] = copy.copy(i_gains[0])
            RX_loc[:, :, cursor1:cursor2] = copy.copy(i_RX_loc)
            mirrors['cell_mapping'][:, cursor1:cursor2] = copy.copy(i_mirrors['cell_mapping'])
            mirrors['RX_displacement'][:, :, cursor1:cursor2] = copy.copy(i_mirrors['RX_displacement'])
            mirrors['RX_loc_all'][:, :, cursor1:cursor2] = copy.copy(i_mirrors['RX_loc_all'])
            mirrors['cell_mapping_all'][:, cursor1:cursor2] = copy.copy(i_mirrors['cell_mapping_all'])
            mirrors['RX_displacement_all'][:, :, cursor1:cursor2] = copy.copy(i_mirrors['RX_displacement_all'])

    rayleigh_var = 1.0  # 瑞利衰落因子单位方差

    weights = []
    for loop in range(total_samples):
        weights.append(np.array(np.ones(u)))
    # Coefficients for shadowing
    if max_doppler == 'mixed':
        f_d = np.random.uniform(2, 15, (total_samples, m, u))
    elif max_doppler == 'independent':
        f_d = total_samples * [1e10]
    elif max_doppler is None:
        f_d = total_samples * [0]  # placeholder...
    else:
        f_d = total_samples * [max_doppler]

    # Init Optimizer results
    H_all = []
    H_all_real = []
    H_large = []  # 大尺度衰落
    H_error = []  # 信道估计误差值
    # Optimum solution with no CSI delay
    # Also extract cell mapping and displacement
    f_d[0] = np.zeros((m, u))  # 这块没看懂为什么赋一个这么大维度的数组，直接赋值0不就可以了
    cell_mapping = mirrors['cell_mapping']
    RX_displacement = mirrors['RX_displacement']
    tmp_channel_b = [pb.generate_complex_gaussian_variance(rayleigh_var, u, n, m)]  # will return a matrix of m times u.
    tmp_channel_error = [pb.generate_complex_gaussian_variance(channel_error_var, u, n, m)]  # 获得信道误差
    tmp_channel = [pb.generate_complex_gaussian_variance(rayleigh_var, u, n)]  # Just to initialze this array.
    tmp_channel_real = [pb.generate_complex_gaussian_variance(rayleigh_var, u, n)]  # Just to initialze this array.
    for k in range(u):
        tmp_channel[0][k, :, :] = tmp_channel_b[0][cell_mapping[:, 0], k, :] - tmp_channel_error[0][
                                                                               cell_mapping[:, 0], k, :]  # 获得h估计值
        tmp_channel_real[0][k, :, :] = tmp_channel_b[0][cell_mapping[:, 0], k, :]
    tmp_H_all = np.zeros((u, u, n))  # 开方后的大尺度乘以小尺度，估计值
    tmp_H_all_real = np.zeros((u, u, n))  # 开方后的大尺度乘以小尺度,实际值
    tmp_H_large = np.zeros((u, u, n))  # 存储大尺度衰落
    tmp_h_estimation = np.zeros((u, u, n))  # 存储不同时隙信道估计误差
    for n_index in range(n):  # 对所有子信道都进行赋值操作，tmp_H_all和tmp_h_estimation是不同子信道是不同的，但是大尺度在所有子信道上都一样
        tmp_H_all[:, :, n_index] = np.multiply(np.sqrt(gains[0][:, :, 0]),
                                               ((abs(tmp_channel[-1][:, :, n_index]))))  # 获得总体H的估计值
        tmp_H_all_real[:, :, n_index] = np.multiply(np.sqrt(gains[0][:, :, 0]),
                                                    ((abs(tmp_channel_real[-1][:, :, n_index]))))  # 获得总体H的实际值
        tmp_H_large[:, :, n_index] = gains[0][:, :, 0]  # 大尺度衰落因子在所有子信道上均一致
        tmp_h_estimation[:, :, n_index] = abs(tmp_channel[-1][:, :, n_index])
    H_large.append(tmp_H_large)
    H_error.append(tmp_h_estimation)
    H_all.append(tmp_H_all)  # 后续需要进行平方，因为这里是开方后的大尺度乘以估计小尺度
    H_all_real.append(tmp_H_all_real)  # 后续需要进行平方，因为这里是开方后的大尺度乘以真实小尺度

    for i in range(1, total_samples):  # 每一个时隙的数据一共total_samples个时隙的数据，上面已经存了第一个时隙索引为0，这里索引从1开始
        if mobility_params['v_max'] == 0:
            if max_doppler == 'independent':
                correlation = 0.0
            else:
                correlation = special.j0(2.0 * np.pi * f_d[i] * T)
        else:
            f_d[i] = np.zeros((m, u))
            for k in range(u):
                f_d[i][:, k] = np.sqrt(RX_displacement[0, k, i] ** 2 + RX_displacement[1, k, i] ** 2) * f_c / (
                        T * v_c)
            correlation = special.j0(2.0 * np.pi * f_d[i] * T)
            correlation = np.dstack(tuple([correlation] * n))
        if train_episodes is not None and i % train_episodes['T_train'] == 0:
            tmp_tmp_channel_b = pb.generate_complex_gaussian_variance(rayleigh_var, u, n, m)
        else:
            tmp_tmp_channel_b = pb.get_markov_rayleigh_variable(tmp_channel_b[-1], correlation, rayleigh_var, u, n,
                                                                m)
        tmp_tmp_channel_error = pb.generate_complex_gaussian_variance(channel_error_var, u, n, m)  # 获得信道误差
        tmp_tmp_channel = np.zeros(np.shape(tmp_channel[0])) + 0j
        tmp_tmp_real = np.zeros(np.shape(tmp_channel[0])) + 0j
        for k in range(u):
            tmp_tmp_channel[k, :] = tmp_tmp_channel_b[cell_mapping[:, i], k] - tmp_tmp_channel_error[
                cell_mapping[:, i], k]  # 后续时隙的h的估计值
            tmp_tmp_real[k, :] = tmp_tmp_channel_b[cell_mapping[:, i], k]  # 后续时隙的h的实际值
        tmp_channel_b.append(tmp_tmp_channel_b)
        tmp_H_all_real = np.zeros((u, u, n))  # 实际的
        tmp_H_all = np.zeros((u, u, n))  # 估计的
        tmp_H_large = np.zeros((u, u, n))  # 存储大尺度衰落
        tmp_h_estimation = np.zeros((u, u, n))  # 存储不同时隙信道估计误差
        for n_index in range(n):  # 选择不同的子频带不影响大尺度衰落的值，因此对于M个子频带的每个子带都是相同的损耗
            tmp_H_all_real[:, :, n_index] = np.multiply(np.sqrt(gains[0][:, :, i]),
                                                        ((abs(tmp_tmp_real[:, :, n_index]))))
            tmp_H_all[:, :, n_index] = np.multiply(np.sqrt(gains[0][:, :, i]), ((abs(tmp_tmp_channel[:, :, n_index]))))
            tmp_H_large[:, :, n_index] = gains[0][:, :, i]
            tmp_h_estimation[:, :, n_index] = abs(tmp_tmp_channel[:, :, n_index])
        H_large.append(tmp_H_large)  # 存储全部时隙的大尺度衰落
        H_error.append(tmp_h_estimation)  # 存储全部时隙的信道小尺度估计值
        H_all.append(tmp_H_all)
        H_all_real.append(tmp_H_all_real)
        tmp_channel.append(tmp_tmp_channel)

    # No need to save the deployment.
    # np_save_path = './simulations/deployment/%s_network%d'%(json_file,overal_sims)
    # if type(mirrors) is dict:
    #     np.savez(np_save_path,options,f_d,gains,TX_loc,RX_loc,TX_xhex,TX_yhex,TX_neighbors,H_all,mirrors['cell_mapping_all'],
    #              mirrors['RX_loc_all'],mirrors['RX_displacement'],mirrors['RX_displacement_all'],mirrors['cell_mapping'])
    # else:
    #     np.savez(np_save_path,options,gains,TX_loc,RX_loc,TX_xhex,TX_yhex,TX_neighbors,mirrors)
    # print('Saved to %s'%(np_save_path))
    '''
    再次seed保证只有子信道不同时不会影响用户的任务大小和强度，只有u不同时才可能影响
    '''
    if isTrain:
        np.random.seed(1000)
    else:
        np.random.seed(1000 + u)
    # 获取计算参数
    task_params = compute_params.get('task', {})
    cpu_params = compute_params.get('cpu', {})
    requirement_params = compute_params.get('T_requirement', {})
    if isTrain:
        # 生成任务大小的矩阵
        task_matrix = pb.generate_task_cpu_requirement_sizes(u, total_samples, task_params)

        # 生成CPU轮数大小的矩阵
        cpu_matrix = pb.generate_task_cpu_requirement_sizes(u, total_samples, cpu_params)

        # 生成用户需求大小的矩阵
        requirement_matrix = pb.generate_task_cpu_requirement_sizes(u, total_samples, requirement_params)
        
        # 生成用户相对到达顺序的矩阵
        user_order_matrix = pb.generate_user_order(u, total_samples)

    else:
        task_matrix = np.zeros((u, total_samples))
        cpu_matrix = np.zeros((u, total_samples))
        requirement_matrix = np.zeros((u, total_samples))
        user_order_matrix = np.zeros((u, total_samples))
        tot_test_episodes = int(total_samples / train_episodes['T_train'])
        
        "类似的,将下列生成的需求从循环中取出"
        # 生成任务大小的矩阵
        i_task_matrix = pb.generate_task_cpu_requirement_sizes(u, train_episodes['T_train'], task_params)

        # 生成CPU轮数大小的矩阵
        i_cpu_matrix = pb.generate_task_cpu_requirement_sizes(u, train_episodes['T_train'], cpu_params)

        # 生成用户需求-减去用户waiting time大小的矩阵
        i_requirement_matrix = pb.generate_task_cpu_requirement_sizes(u, train_episodes['T_train'], requirement_params)
        
        i_user_order_matrix = pb.generate_user_order(u, train_episodes['T_train'])

        for ep in range(tot_test_episodes):

            cursor1 = int(ep * train_episodes['T_train'])
            cursor2 = int((ep + 1) * train_episodes['T_train'])
            task_matrix[:, cursor1:cursor2] = copy.copy(i_task_matrix)
            cpu_matrix[:, cursor1:cursor2] = copy.copy(i_cpu_matrix)
            requirement_matrix[:, cursor1:cursor2] = copy.copy(i_requirement_matrix)
            user_order_matrix[:, cursor1:cursor2] = copy.copy(i_user_order_matrix)
            
    # 定义文件夹路径
    np_save_folder = './simulations/channel'
    os.makedirs(np_save_folder, exist_ok=True)

    np_save_path = os.path.join(np_save_folder, f'{json_file_name}_network')
    if save_H_all: np.savez(np_save_path, H_all_real, mirrors, task_matrix, cpu_matrix,requirement_matrix,user_order_matrix,H_all)
    else: np.savez(np_save_path, H_all_real, mirrors, task_matrix, cpu_matrix,requirement_matrix,user_order_matrix)
    print('Saved to %s' % (np_save_path))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # import os
    # import json
    #
    # # 获取目录下所有 JSON 文件的文件名
    # directory = './config/deployment/'
    # json_files = [file for file in os.listdir(directory) if file.endswith('.json')]
    #
    # # 遍历所有 JSON 文件并修改特定键值下的数据为 500
    # for json_file_name in json_files:
    #     with open(os.path.join(directory, json_file_name), 'r') as f:
    #         options = json.load(f)
    #
    #     # 修改特定键值下的数据为 500
    #     options['simulation']['total_samples'] = 500
    #
    #     # 保存修改后的内容回到文件
    #     with open(os.path.join(directory, json_file_name), 'w') as f:
    #         json.dump(options, f, indent=4)  # 使用 indent 参数使数据更加可读

    parser = argparse.ArgumentParser(description="give test scenarios")
    parser.add_argument('--json-file', type=str,
                        default='M3_U21_N5_Task_u15000_Ucpu_e1000_Tdemand_u15_error0_hetero_fix',
                        help='json file for the deployment')
    args = parser.parse_args()
    main(args)
