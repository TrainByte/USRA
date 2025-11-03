"""
@author: Ning zhe Shi
u: users number
m: bs number
n: sub-channel number
"""
import copy

import numpy as np
import matplotlib.pyplot as plt


def get_gains_hexagon_neighbors_shadowing(u, m, R, min_dist, total_samples, shadowing_dev=10, dcor=10,
                                          equal_number_for_BS=True, draw=False, T=20e-3, train_episodes=None,
                                          mobility_params=None):
    TX_loc, RX_loc, TX_xhex, TX_yhex, neighbors, mirrors, u = get_random_locationshexagon_neighbors(u, m, R,
                                                                                                    min_dist,
                                                                                                    equal_number_for_BS=equal_number_for_BS,
                                                                                                    total_samples=total_samples,
                                                                                                    T=T,
                                                                                                    train_episodes=train_episodes,
                                                                                                    mobility_params=mobility_params)
    distance_vector = get_distance(u, TX_loc, RX_loc, mirrors=mirrors, total_samples=total_samples)

    gains = []
    # Get 2D distance pathloss, original pathloss tried in the previous versions
    # Get channel gains
    g_dB2 = - (128.1 + 37.6 * np.log10(0.001 * distance_vector))  # 存储的是total_samples个时隙，其中第i列存储的为第i个用户所在小区基站与所有用户的信道增益

    # init_shadowing
    tmp_g_dB = np.zeros((u, u, total_samples))
    shadowing = np.zeros((m, u, total_samples))
    shadowing[:, :, 0] = np.random.randn(m, u)
    RX_displacement = mirrors['RX_displacement']
    cell_mapping = mirrors['cell_mapping']
    for sample in range(1, total_samples):
        for n in range(m):
            correlation = np.exp(
                - np.sqrt(RX_displacement[0, :, sample] ** 2 + RX_displacement[1, :, sample] ** 2) / dcor)
            shadowing[n, :, sample] = np.multiply(correlation, shadowing[n, :, sample - 1]) + np.multiply(
                np.sqrt(1.0 - np.square(correlation)), np.random.randn(1, u))
    for sample in range(total_samples):
        for k in range(u):
            tmp_g_dB[k, :, sample] = g_dB2[k, :, sample] + shadowing_dev * shadowing[
                cell_mapping[:, sample], k, sample]  # 存储的是total_samples个时隙，其中第k行存储的为第k个用户与所有用户所在小区基站的信道增益
    gains.append(np.power(10.0, tmp_g_dB / 10.0))

    if (draw == True):
        #        plt.plot(TX_loc.T, 'b^')
        #        plt.plot(RX_loc.T, 'ro')
        for i in range(m):
            plt.plot(TX_loc[0, i], TX_loc[1, i], 'g^')
            plt.text(TX_loc[0, i], TX_loc[1, i], str(i), fontsize=8)  # 在点的旁边添加标识
            plt.plot(TX_xhex[:, i], TX_yhex[:, i], 'k-')
        for i in range(u):
            plt.plot(RX_loc[0, i, 0], RX_loc[1, i, 0], 'ro')
            plt.text(RX_loc[0, i, 0], RX_loc[1, i, 0], str(i), fontsize=8)  # 在点的旁边添加标识
        plt.show()
    return gains, TX_loc, RX_loc, TX_xhex, TX_yhex, neighbors, mirrors


def get_random_locationshexagon_neighbors(u, m, R, min_dist, equal_number_for_BS=True,
                                          total_samples=None, bvar=1.0, T=20e-3, train_episodes=None,
                                          mobility_params=None):
    if equal_number_for_BS is True:
        assert u % m == 0, 'u needs to be divisible by UE_perBS!'
    mirrors = []
    cell_mapping = []
    # Brownian motion case...
    neighbors = []  # neighboring cells

    # IMAC Case: we have the mirror BS at the same location.
    mirrors = []
    max_dist = R
    x_hexagon = R * np.array([0, -np.sqrt(3) / 2, -np.sqrt(3) / 2, 0, np.sqrt(3) / 2, np.sqrt(3) / 2, 0])
    y_hexagon = R * np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1])

    TX_loc = np.zeros((2, m))  # m个基站的中心横坐标及纵坐标
    TX_xhex = np.zeros((7, m))  # m个基站的六个顶点的横坐标，第七个点与第一个点一样
    TX_yhex = np.zeros((7, m))  # m个基站的六个顶点的纵坐标，第七个点与第一个点一样

    RX_loc = np.zeros((2, u, total_samples))  # u个用户的位置，随total_samples(时间)变化
    cell_mapping = np.zeros((u, total_samples)).astype(int)  # u个用户所对应的基站号，随total_samples(时间)变化
    RX_displacement = np.zeros((4, u, total_samples))  # displacement and angle

    ############### DROP KLL KKTERS
    generated_hexagons = 0
    i = 0
    # if (u>0):
    for k in range(1):  # 第一轮只生成第一个基站的坐标
        TX_loc[0, generated_hexagons * 1 + k] = 0.0  # 赋值第一个基站的中心横坐标
        TX_loc[1, generated_hexagons * 1 + k] = 0.0  # 赋值第一个基站的中心纵坐标
        TX_xhex[:, generated_hexagons * 1 + k] = x_hexagon  # 赋值第一个基站的六个顶点的横坐标
        TX_yhex[:, generated_hexagons * 1 + k] = y_hexagon  # 赋值第一个基站的六个顶点的纵坐标
    generated_hexagons += 1

    while (generated_hexagons < m):

        for j in range(6):  # 第一个基站周围的6个基站，当基站不足1+6时也不用担心，因为if (generated_hexagons >= m):语句会退出该循环
            tmp_xloc = TX_loc[0, i] + np.sqrt(3) * R * np.cos(j * np.pi / (3))  # 第1个基站周围的6个基站的中心横坐标
            tmp_yloc = TX_loc[1, i] + np.sqrt(3) * R * np.sin(j * np.pi / (3))  # 第1个基站周围的6个基站的中心纵坐标
            tmp_xhex = tmp_xloc + x_hexagon
            tmp_yhex = tmp_yloc + y_hexagon
            was_before = False
            # 判断生成的小区是否已经存在
            for inner_loop in range(generated_hexagons):
                if (abs(tmp_xloc - TX_loc[0, inner_loop * 1]) < R * 1e-2 and abs(
                        tmp_yloc - TX_loc[1, inner_loop * 1]) < R * 1e-2):
                    was_before = True
                    break
            # 若不存在则生成
            if (not was_before):
                for k in range(1):  # 将上面计算的结果赋值生成即可
                    TX_loc[0, generated_hexagons * 1 + k] = tmp_xloc
                    TX_loc[1, generated_hexagons * 1 + k] = tmp_yloc
                    TX_xhex[:, generated_hexagons * 1 + k] = tmp_xhex
                    TX_yhex[:, generated_hexagons * 1 + k] = tmp_yhex
                generated_hexagons += 1
            if (generated_hexagons >= m):
                break
        i += 1  # 若需要生成的基站数大于1+6，则从第二个基站为中心再次进行如第一个基站类似的操作，以此类推。
    # Then find the neighbors
    for i in range(m):
        tmp_neighbors = []
        for j in range(6):
            tmp_xloc = TX_loc[0, i] + np.sqrt(3) * R * np.cos(j * np.pi / (3))
            tmp_yloc = TX_loc[1, i] + np.sqrt(3) * R * np.sin(j * np.pi / (3))
            tmp_xhex = tmp_xloc + x_hexagon
            tmp_yhex = tmp_yloc + y_hexagon
            for inner_loop in range(m):
                if (inner_loop != i and abs(tmp_xloc - TX_loc[0, inner_loop]) < R * 1e-2 and abs(
                        tmp_yloc - TX_loc[1, inner_loop]) < R * 1e-2):
                    tmp_neighbors.append(inner_loop)
        for j in range(1):
            neighbors.append(tmp_neighbors)  # 存储的是m个list，第m个list对应第m个基站的邻居基站的编号(周围一群的基站编号).注意基站是从0开始编号的.
    ############### DROP USERS  第一个时隙的数据
    a_max = mobility_params['a_max']
    v_max = mobility_params['v_max']
    alpha_angle = mobility_params['alpha_angle']
    T_mobility = mobility_params['T_mobility']
    for i in range(u):
        # Randomly assign initial cell placement
        if equal_number_for_BS:
            assert u % m == 0
            UE_perBS = int(u / m)
            cell_mapping[i, 0] = int(i / UE_perBS)  # 存放的是用户对应基站的序号
        else:
            cell_mapping[i, 0] = np.random.randint(m)  # 如果不是等比例放置用户，则用户随机分布在m个基站
        this_cell = cell_mapping[i, 0]
        # Place UE within that cell.
        constraint_minx_UE = min(TX_xhex[:, this_cell])  # 该小区的横坐标中最小值
        constraint_maxx_UE = max(TX_xhex[:, this_cell])  # 该小区的横坐标中最大值
        constraint_miny_UE = min(TX_yhex[:, this_cell])  # 该小区的纵坐标中最小值
        constraint_maxy_UE = max(TX_yhex[:, this_cell])  # 该小区的纵坐标中最大值
        inside_checker = True
        while (inside_checker):
            RX_displacement[2, i, 0] = np.random.uniform(0, v_max)  # Initial speed.
            RX_displacement[3, i, 0] = np.random.uniform(-np.pi, np.pi)  # Initial angle.
            RX_loc[0, i, 0] = np.random.uniform(constraint_minx_UE, constraint_maxx_UE)
            RX_loc[1, i, 0] = np.random.uniform(constraint_miny_UE, constraint_maxy_UE)
            tmp_distance2center = np.sqrt(
                np.square(RX_loc[0, i, 0] - TX_loc[0, this_cell]) + np.square(RX_loc[1, i, 0] - TX_loc[1, this_cell]))
            if (inside_hexagon(RX_loc[0, i, 0], RX_loc[1, i, 0], TX_xhex[:, this_cell], TX_yhex[:, this_cell])
                    and tmp_distance2center > min_dist and tmp_distance2center < max_dist):  # 判断是否在定义的区间里
                inside_checker = False  # 在定义区间内就跳出检测器，生成第一个时隙中的下一个user的位置
    ############### MOVE USERS 后面的时隙
    step_size = T_mobility

    sleep_step_size = 1.0 + float(train_episodes['T_sleep']) / float(train_episodes['T_train'])
    RX_loc_all = np.zeros((2, u, int(total_samples * sleep_step_size)))
    RX_loc_all[:, :, 0] = RX_loc[:, :, 0]
    cell_mapping_all = np.zeros((u, int(total_samples * sleep_step_size))).astype(int)
    cell_mapping_all[:, 0] = cell_mapping[:, 0]
    cell_request_change = (-1 * np.ones(u)).astype(int)
    cell_request_counter = train_episodes['T_register'] * np.ones(u)
    RX_displacement_all = np.zeros((4, u, int(total_samples * sleep_step_size)))  # displacement and angle
    RX_displacement_all[:, :, 0] = RX_displacement[:, :, 0]  # Initial speed and angle.
    is_mode_sleep = False
    is_mode_train = True
    sample_train = 0
    for sample in range(1, int(total_samples * sleep_step_size)):  # 上面生成的第0个时隙的数据，下面从第1个时隙开始生成
        # Get the mode if necessary
        if sample != 1 and (sample) % train_episodes['T_train'] == 0:
            is_mode_sleep = True
            is_mode_train = False
        if (sample) % (train_episodes['T_sleep'] + train_episodes['T_train']) == 0:
            is_mode_sleep = False
            is_mode_train = True
        if is_mode_train:
            sample_train += 1
        for i in range(u):
            if sample % step_size == 0:  # 每隔step_size个时隙用户的移动速度和方向重新随机化
                delta_v = np.random.uniform(-a_max, a_max)
                delta_angle = np.random.uniform(-alpha_angle, alpha_angle)
                RX_displacement_all[2, i, sample] = min(max(RX_displacement_all[2, i, sample - 1] + delta_v, 0.0),
                                                        v_max)  # v
                RX_displacement_all[3, i, sample] = RX_displacement_all[3, i, sample - 1] + delta_angle  # angle
            else:  # 不到step_size个时隙则用户的移动速度和方向不变
                RX_displacement_all[2, i, sample] = RX_displacement_all[2, i, sample - 1]
                RX_displacement_all[3, i, sample] = RX_displacement_all[3, i, sample - 1]
            if cell_request_change[i] == -1:
                prev_cell = cell_mapping_all[i, sample - 1]  # 读取第i个用户在上个时隙的所处小区号
                prev_cell_map = prev_cell
            elif cell_request_counter[i] > 0:
                cell_request_counter[i] = cell_request_counter[i] - 1
                prev_cell = cell_request_change[i]
                prev_cell_map = cell_mapping_all[i, sample - 1]
            else:
                prev_cell = cell_request_change[i]
                prev_cell_map = prev_cell
                cell_request_change[i] = -1
                cell_request_counter[i] = train_episodes['T_register']

            constraint_minx_UE = min(TX_xhex[:, prev_cell])
            constraint_maxx_UE = max(TX_xhex[:, prev_cell])
            constraint_miny_UE = min(TX_yhex[:, prev_cell])
            constraint_maxy_UE = max(TX_yhex[:, prev_cell])
            inside_checker = True
            while (inside_checker):
                RX_displacement_all[0, i, sample] = T * RX_displacement_all[2, i, sample] * np.cos(
                    RX_displacement_all[3, i, sample])  # displacement x
                RX_displacement_all[1, i, sample] = T * RX_displacement_all[2, i, sample] * np.sin(
                    RX_displacement_all[3, i, sample])  # displacement y
                RX_loc_all[0, i, sample] = RX_loc_all[0, i, sample - 1] + RX_displacement_all[
                    0, i, sample]  # 上一个时隙的x轴位置加上移动后的新位置
                RX_loc_all[1, i, sample] = RX_loc_all[1, i, sample - 1] + RX_displacement_all[
                    1, i, sample]  # 上一个时隙的y轴位置加上移动后的新位置
                tmp_distance2center = np.sqrt(np.square(RX_loc_all[0, i, sample] - TX_loc[0, prev_cell]) + np.square(
                    RX_loc_all[1, i, sample] - TX_loc[1, prev_cell]))  # 计算新位置与小区的基站的距离
                if is_mode_train:
                    RX_displacement[:, i, sample_train] = RX_displacement_all[:, i, sample]
                    RX_loc[0, i, sample_train] = RX_loc_all[0, i, sample - 1]
                    RX_loc[1, i, sample_train] = RX_loc_all[1, i, sample - 1]
                if (inside_hexagon(RX_loc_all[0, i, sample], RX_loc_all[1, i, sample], TX_xhex[:, prev_cell],
                                   TX_yhex[:, prev_cell])
                        and tmp_distance2center > min_dist and tmp_distance2center < max_dist):  # 判断新位置是否在原小区内及是否满足内外圈要求
                    inside_checker = False
                    cell_mapping_all[i, sample] = prev_cell_map  # The UE is still inside the prev cell
                    if is_mode_train:
                        cell_mapping[i, sample_train] = prev_cell_map
                elif (is_mode_train and train_episodes['cell_passing_training']) or (is_mode_sleep and train_episodes[
                    'cell_passing_sleeping']):  # Kow check the immediate neighbors of the cell to see whether UE is passing to another cell or it is out of bounds.
                    for neigh in neighbors[prev_cell]:
                        tmp_distance2center = np.sqrt(
                            np.square(RX_loc_all[0, i, sample] - TX_loc[0, neigh]) + np.square(
                                RX_loc_all[1, i, sample] - TX_loc[1, neigh]))
                        if (inside_hexagon(RX_loc_all[0, i, sample], RX_loc_all[1, i, sample], TX_xhex[:, neigh],
                                           TX_yhex[:, neigh])
                                and tmp_distance2center > min_dist and tmp_distance2center < max_dist):
                            inside_checker = False
                            cell_mapping_all[i, sample] = prev_cell_map  # The UE is still inside the prev cell
                            if is_mode_train:
                                cell_mapping[i, sample_train] = prev_cell_map
                            if neigh == prev_cell_map:  # Cell get back to original cell, dismiss register.
                                cell_request_change[i] = -1
                                cell_request_counter[i] = train_episodes['T_register']
                            else:
                                cell_request_change[i] = neigh
                                cell_request_counter[i] = train_episodes['T_register']
                            break
                    if inside_checker:
                        # If none of the edges worked boucne back with a random angle.
                        RX_displacement_all[3, i, sample] = np.random.uniform(-np.pi, np.pi)
                else:  # cell passing is not allowed boucne back with a random angle.
                    RX_displacement_all[3, i, sample] = np.random.uniform(-np.pi, np.pi)
                    # If user is out of bounds, redo the motion.
    # Don't want to modify the input structure hence the mirrors used for other cases will be used as a dictionary.
    mirrors = {}
    mirrors['cell_mapping'] = cell_mapping
    mirrors['RX_displacement'] = RX_displacement
    if train_episodes is not None:
        mirrors['RX_loc_all'] = RX_loc_all
        mirrors['cell_mapping_all'] = cell_mapping_all
        mirrors['RX_displacement_all'] = RX_displacement_all
    return TX_loc, RX_loc, TX_xhex, TX_yhex, neighbors, mirrors, u


def get_distance(u, TX_loc, RX_loc, mirrors=None, total_samples=1):
    distance_vector = np.zeros((u, u, total_samples))  # 除去total_samples这个维度，u*u的矩阵，对于同一小区下的user，列数据是相同的
    cell_mapping = mirrors['cell_mapping']
    tmp_TX_loc = np.zeros((2, u))
    for sample in range(total_samples):

        tmp_TX_loc = TX_loc[:, cell_mapping[:, sample]]
        for i in range(u):
            distance_vector[:, i, sample] = np.sqrt(np.square(tmp_TX_loc[0, i] - RX_loc[0, :, sample]) +
                                                    np.square(tmp_TX_loc[1, i] - RX_loc[1, :, sample]))

    return distance_vector  # 存储的是total_samples个时隙，其中第i列存储的为第i个用户所在小区基站与所有用户的距离


# Ray tracing 判断生成的用户是否在小区内
def inside_hexagon(x, y, TX_xhex, TX_yhex):
    n = len(TX_xhex) - 1
    inside = False
    p1x, p1y = TX_xhex[0], TX_yhex[0]
    for i in range(n + 1):
        p2x, p2y = TX_xhex[i % n], TX_yhex[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def generate_complex_gaussian_variance(channel_error_variance, u, n=1, m=None):
    # Generate real and imaginary parts from independent standard normal distributions
    if m is None:
        real_part = np.random.randn(u, u, n) * channel_error_variance
        imag_part = np.random.randn(u, u, n) * channel_error_variance
    else:
        real_part = np.random.randn(m, u, n) * channel_error_variance
        imag_part = np.random.randn(m, u, n) * channel_error_variance

    # Construct complex Gaussian random variables
    complex_gaussian = np.sqrt(2.0 / np.pi) * (real_part + 1j * imag_part)

    return complex_gaussian


def get_markov_rayleigh_variable(state, correlation, rayleigh_var, u, n=1, m=None):
    if m is None:
        return correlation * state + np.sqrt(1 - np.square(correlation)) * np.sqrt(2.0 / np.pi) * rayleigh_var * (
                np.random.randn(u, u, n) +
                1j * np.random.randn(u, u, n))
    else:
        return correlation * state + np.sqrt(1 - np.square(correlation)) * np.sqrt(2.0 / np.pi) * rayleigh_var * (
                np.random.randn(m, u, n) +
                1j * np.random.randn(m, u, n))


def compute_allocation(X, L):
    result = [element * X[i] for i, element in enumerate(L)]
    x_m_k = np.sqrt(result) / np.sum(np.sqrt(result))
    return x_m_k


# def generate_random_array(size):
#     attempts = 0  # 记录生成的尝试次数
#     while True:
#         attempts += 1
#         print_debug(debug=False, 生成随机数组次数=attempts)
#         random_array = np.random.rand(size)
#         if np.sum(random_array) <= 1:
#             return random_array

'''
上述需要多次生成，可能总和小于1.下面的这个生成函数生成的总和一定为1，但可以只生成一次
'''


def generate_random_array(size):
    random_array = np.random.uniform(0, 1, size)
    normalized_array = random_array / np.sum(random_array)
    return normalized_array


def print_debug(debug=False, **kwargs):
    if debug:
        for var_name, var_value in kwargs.items():
            print(f"{var_name} = {var_value}")


def user_channel_mapping_matrix(user_channel, u, n):
    user_channel_matrix = np.zeros((u, n+1))
    for i in range(u):
        channel = user_channel[i]
        user_channel_matrix[i, channel] = 1
    return user_channel_matrix


def generate_task_cpu_requirement_sizes(u, t, params):  # 生成u个用户在t个时隙的数据包的大小，里面包括三种分布
    matrix = np.zeros((u, t))
    distribution = params.get('distribution', "uniform")
    for u_idx in range(u):
        for t_idx in range(t):
            if distribution == "uniform":  # 均匀分布
                min_val = params.get('min_val', 100)  # 表示从 params 字典中获取键为 'min_val' 的值，如果键不存在，则返回默认值 0。
                max_val = params.get('max_val', 1500)  # 均匀分布的最大任务大小
                task_size = np.random.uniform(min_val, max_val)
            elif distribution == "exponential":
                avg_task_size = params.get('avg_size', 1000)  # exponential分布的平均任务大小
                task_size = np.random.exponential(avg_task_size)
            elif distribution == "poisson":
                avg_task_size = params.get('avg_size', 1000)  # poisson分布的平均任务大小
                task_size = np.random.poisson(avg_task_size)
            else:
                raise ValueError('Invalid distribution specified.')

            matrix[u_idx, t_idx] = task_size

    return matrix


def user_uplink_time(H_gain_data, user_channel_temp, p_temp, task, u, W, m, cell_mapping, T_p, NOMA, dropped,weight):
    H_large_t = np.zeros((u, u))
    for i in range(u):
        H_large_t[i, :] = H_gain_data[i, :,
                          user_channel_temp[i] - 1]  # 取出第i个用户在对应信道上与其余所有用户所在小区基站的信道增益矩阵,如果被丢弃即信道号为0则随机取即可，不影响，所以这里直接取-1
    H_gain = H_large_t.T  # 转置对应上求解方式中的信道增益矩阵
    # 构建干扰矩阵
    interference_matrix = np.zeros((u, u))
    # 根据用户选择的子信道号设置干扰矩阵
    for i in range(u):
        for j in range(u):
            if user_channel_temp[i] == user_channel_temp[j] and user_channel_temp[i] != 0:
                interference_matrix[i, j] = 1

    OMA_temp = np.multiply(interference_matrix, 1 - np.eye(u))
    OMA_temp[OMA_temp >= 1] = 1

    if NOMA is True:
        A_temp = NOMA_handle(user_channel_temp, OMA_temp, m, cell_mapping, H_gain, p_temp)
    else:
        A_temp = OMA_temp

    y_star = np.sqrt(np.diag(H_gain).reshape(-1, 1) * p_temp) / (
            H_gain * A_temp @ p_temp + 1)  # 公式（33）
    rate = (W * (np.log2(
        1 + 2 * np.multiply(y_star, np.sqrt(
            np.multiply(np.diag(H_gain).reshape(-1, 1), p_temp))) - np.multiply(
            np.square(y_star), (
                    H_gain * A_temp @ p_temp + 1)))))  # 公式（32）
    # 识别被丢弃的用户（信道号为0）
    discarded_users = user_channel_temp.reshape(-1, 1) == 0
    if dropped == True:
        latency = np.where(discarded_users, -weight*T_p, task.reshape(-1, 1) / rate)
    else:
        latency = task.reshape(-1, 1) / rate
    
    # 将被丢弃的用户的速率设为0,2025年4月20日添加
    rate[discarded_users.flatten()] = 0
    return latency,rate


def user_uplink_time_se_ee(H_gain_data, user_channel_temp, p_temp, task, u, W, m, cell_mapping, NOMA):
    H_large_t = np.zeros((u, u))
    for i in range(u):
        H_large_t[i, :] = H_gain_data[i, :, user_channel_temp[i] - 1]  # 取出第i个用户在对应信道上与其余所有用户所在小区基站的信道增益矩阵
    H_gain = H_large_t.T  # 转置对应上求解方式中的信道增益矩阵
    # 构建干扰矩阵
    interference_matrix = np.zeros((u, u))
    # 根据用户选择的子信道号设置干扰矩阵
    for i in range(u):
        for j in range(u):
            if user_channel_temp[i] == user_channel_temp[j]:
                interference_matrix[i, j] = 1

    OMA_temp = np.multiply(interference_matrix, 1 - np.eye(u))
    OMA_temp[OMA_temp >= 1] = 1

    if NOMA is True:
        A_temp = NOMA_handle(user_channel_temp, OMA_temp, m, cell_mapping, H_gain, p_temp)
    else:
        A_temp = OMA_temp

    y_star = np.sqrt(np.diag(H_gain).reshape(-1, 1) * p_temp) / (
            H_gain * A_temp @ p_temp + 1)  # 公式（33）
    rate = (W * (np.log2(
        1 + 2 * np.multiply(y_star, np.sqrt(
            np.multiply(np.diag(H_gain).reshape(-1, 1), p_temp))) - np.multiply(
            np.square(y_star), (
                    H_gain * A_temp @ p_temp + 1)))))  # 公式（32）
    delay = task.reshape(-1, 1) / rate
    se = rate / W
    ee = se / p_temp
    return delay, se, ee, rate


def find_all_duplicate_positions(arr):
    position_dict = {}

    for i, val in enumerate(arr):
        position_dict.setdefault(val, []).append(i)

    return {val: positions for val, positions in position_dict.items() if len(positions) > 1}


def NOMA_handle(user_channel_temp, OMA_temp, m, cell_mapping, H_gain, p_temp):
    Noma_snr = True
    for target_cell in range(m):
        target_cell_users_channel = user_channel_temp[
            cell_mapping == target_cell]  # 时隙t位于target_cell小区的用户的每bit数据所需的CPU轮数大小
        one_cell_users_index = np.where(cell_mapping == target_cell)[0]  # 第target_cell小区所有用户的索引
        # 查找所有重复元素的位置,duplicate_positions为字典，字典keys为第target_cell号小区的重复信道使用的信道号，字典values为对应的用户号。例如 {4: [0, 10], 6: [2, 5, 7]}表示4号子信道被该小区的第0个和第10个用户重复使用
        duplicate_positions = find_all_duplicate_positions(target_cell_users_channel)

        if not duplicate_positions:
            continue
            # print(f"小区{target_cell}中没有找到重复元素")
        else:
            # print(f"小区{target_cell}中重复元素及其位置")
            for channel_index, same_channel_users_index in duplicate_positions.items():
                # print(f"小区{target_cell}中使用子信道{channel_index} 的用户号:", one_cell_users_index[same_channel_users_index])
                # 取出使用相同子信道在全局用户编号的值
                mapping_users_index = one_cell_users_index[same_channel_users_index]
                # 取出该小区使用相同子信道channel_index的用户所对应的信道增益
                H_gain_noma = H_gain[mapping_users_index, mapping_users_index]
                if Noma_snr is True:
                    p_temp_noma = p_temp[mapping_users_index].squeeze()
                    H_gain_noma = H_gain_noma * p_temp_noma
                # 从小到大对H_gain数组排序，同时保持全局用户索引位置变化
                sorted_indices = np.argsort(H_gain_noma)
                sorted_H_gain_noma = H_gain_noma[sorted_indices]
                sorted_one_cell_users_index = mapping_users_index[sorted_indices]
                for i in range(len(sorted_one_cell_users_index)):
                    current_value = sorted_one_cell_users_index[i]
                    remaining_values = sorted_one_cell_users_index[i + 1:]  # 取出比current_value号用户信道质量强的用户号
                    OMA_temp[current_value, remaining_values] = 0  # 比current_value号（全局用户索引）用户信道质量强的赋值0消除干扰对信道质量较好的影响
    return OMA_temp


# 各个小区内各个子信道上的用户最大数量
def subchannel_users_count(vars_temp, cell_mapping, m):
    target_cell_channel_users_max = np.zeros((m, 1))
    for target_cell in range(m):
        # `minlength` 参数在 `np.bincount` 中确保返回数组的最小长度。当设置为2时，无论输入数组的内容如何，`np.bincount` 至少返回包含两个元素的数组，从而避免在处理稀疏数据（如全为0的情况）时出现数组越界的错误。
        target_cell_channel_users_max[target_cell] = max(np.bincount(vars_temp[cell_mapping == target_cell],minlength=2)[1:])
    return target_cell_channel_users_max


# 定义一个函数来检查数组是否全为零
def is_array_non_zero(arr):
    return np.any(arr != 0)


def compute_allocation_new(execution_total_cycles):
    x_m_k = np.sqrt(execution_total_cycles) / np.sum(np.sqrt(execution_total_cycles))
    return x_m_k


def find_all_min_indices(arr):
    # 找到数组中的最小值
    min_value = np.nanmin(arr)
    # 使用布尔掩码找到所有最小值的索引
    mask = (arr == min_value)
    # 使用 np.where 来找到所有最小值的索引
    min_indices = np.transpose(np.where(mask))
    return min_value, min_indices


def compute_real_time_allocation(t, m, task_matrix, cpu_matrix, cpu_cycles, cell_mapping, user_uplink_time):
    '''
    compute allocation

    参数：
    t (int)：当前帧，通常是一个时间戳或时刻。
    m (int)：MEC个数。
    task_matrix (array)：任务的数据量大小（bit），代表任务大小。
    cpu_matrix (array)：CPU需求情况，任务每bit所需cpu轮数，代表任务类型。
    cpu_cycles (float)：CPU计算总周期，表示MEC的CPU在一个时间单位内可以执行的任务轮数。
    cell_mapping (array)：用户和MEC之间的映射关系，记录哪些用户被分配到哪个MEC上。
    user_uplink_time (array)：用户的上行传输时间，即每个用户到达MEC开始计算的时刻。

    返回：
    返回计算后的计算所需时间
    '''
    T_m_k_c_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户在最优的分配比例下的计算时间，矩阵大小为（用户数）
    for target_cell in range(m):  # 对m个小区遍历执行
        X_m_k = cpu_matrix[cell_mapping[:, t] == target_cell, t]  # 时隙t位于target_cell小区的用户的每bit数据所需的CPU轮数大小
        L_m_k = task_matrix[cell_mapping[:, t] == target_cell, t]  # 时隙t位于target_cell小区的用户的任务的bit大小
        Uplink_time_m_k = user_uplink_time[cell_mapping[:,
                                           t] == target_cell]  # 开始时间，在while循环中将会改变，即任务进入MEC开始计算后将开始时间赋值NAN，避免对find_all_min_indices函数的影响
        Uplink_time_m_k_mapping = copy.deepcopy(Uplink_time_m_k)  # 深拷贝一份计算开始时间，用于最后计算每个用户的计算时间（用结束时间减去该开始时间）
        Expected_end_time = np.full(Uplink_time_m_k.shape, float('inf'))  # 预计结束时间，随着新任务进入和旧任务出来会发生改变，初始化为inf
        end_time_mapping = np.full(Uplink_time_m_k.shape, float('inf'))  # 最终实际结束时间，初始化为inf
        Remaining_execution_total_cycles = X_m_k * L_m_k  # 时隙t位于target_cell小区的用户的剩余任务所需总的CPU轮数大小，初始化即为总需求，随着时间的推进会减少
        target_users = []  # 当前在MEC中执行的小区内用户号（从0-k编号），最后再通过target_users编号映射回去
        combined_matrix = np.vstack((Uplink_time_m_k, Expected_end_time))  # 开始时间和预计结束时间的合体，这里和570行的用处差不多，这里主要是为了初始化
        min_value, _ = find_all_min_indices(combined_matrix)  # 寻找开始时间和结束时间里面的最小值进行任务进入和出来的操作
        last_time = min_value  # 初始化上一次任务进出的时刻
        current_time = min_value  # 初始化当前操作的时刻
        count = 0  # 用于计数while循环次数，且正常循环次数为m中用户数的二倍，因为用户任务一进一出共两次，所以是二倍。如果有用户到达/离开时刻重叠则减少
        while is_array_non_zero(Remaining_execution_total_cycles):  # 当所有用户的剩余任务所需CPU轮数均为0时结束该小区的操作
            count += 1  # 从1开始计数，每次+1
            combined_matrix = np.vstack((Uplink_time_m_k, Expected_end_time))  # 开始时间和预计结束时间，最终执行结束里面应该全是NAN
            min_value, min_value_position = find_all_min_indices(combined_matrix)  # 寻找开始时间和结束时间里面的最小值进行任务进入和出来的操作
            last_time = current_time  # 上一时刻等于上一次循环的当前时刻
            current_time = combined_matrix[
                tuple(min_value_position[0])]  # 当前时刻为开始时间和预计结束时间里面最小的那个，有多个最小的当前时刻也是一样，所以没有bug
            #  进行资源分配初始化
            if count == 1:  # 如果是第一次进入，需要初始化一次任务分配比例x_m_k，虽然对577行的剩余任务数量执行没有影响，因为(current_time - last_time)=0，但是需要对x_m_k进行初始化
                x_m_k = compute_allocation_new(Remaining_execution_total_cycles[target_users])  # 初始化
            Remaining_execution_total_cycles[target_users] -= x_m_k * cpu_cycles * (
                        current_time - last_time)  # 从上一时刻到该时刻用上一时刻(上一轮循环)结束时分配的x_m_k进行计算，减去这些时间计算轮数即可得到剩余任务所需计算轮数
            Remaining_execution_total_cycles[
                Remaining_execution_total_cycles < 1e-2] = 0  # 1e-2为了误差，因为计算时间用了除号，这里又用任务来算所以一般会有1e-10的误差
            for i in range(len(min_value_position)):  # len(min_value_position)为了防止一个时刻有多个用户出/入，虽然概率很小，因为很多位小数
                if min_value_position[i][0] == 0:  # 如果最小时刻是combined_matrix矩阵的第一行即任务到达，则执行任务到达的操作
                    target_users.append(
                        min_value_position[i][1])  # target_users记录该小区的第几个用户的任务到达了，该列表记录的就是目前MEC中的任务是哪些用户的
                    Uplink_time_m_k[min_value_position[i][1]] = float("nan")  # 到达后将该用户对应的Uplink_time_m_k变量赋值NAN，方面后续处理
                else:
                    target_users.remove(min_value_position[i][1])  # 如果最小时刻是combined_matrix矩阵的第一行即任务离开，则执行任务离开的操作
                    end_time_mapping[min_value_position[i][1]] = current_time  # target_users中移除该用户编号，记录该小区的该用户的任务结束了
                    Expected_end_time[min_value_position[i][1]] = float("nan")  # 并给期望结束时间赋值NAN，方便后续处理
            unique_elements = set(target_users)  # 为了下面判断是否MEC同时处理相同的任务，如果是的话其实就是有bug
            # 比较原始列表和去重后的列表
            if len(unique_elements) < len(target_users):
                print("警告：列表中包含重复元素")
            x_m_k = compute_allocation_new(Remaining_execution_total_cycles[
                                               target_users])  # 采用闭式解进行计算资源划分，采用target_users即MEC中目前在执行的用户编号的剩余任务量Remaining_execution_total_cycles
            T_m_k_c = Remaining_execution_total_cycles[target_users] / (x_m_k * cpu_cycles)  # 所需时间
            Expected_end_time[target_users] = current_time + T_m_k_c  # 当前时刻+所需计算时间=预计结束时间

        T_m_c = end_time_mapping - Uplink_time_m_k_mapping  # 计算时隙t小区m中的各个用户的计算时间
        T_m_k_c_matrix[cell_mapping[:, t] == target_cell] = T_m_c  # 存储时隙t小区m中用户的计算时间至原始位置

    return T_m_k_c_matrix  # 返回最优分配的用户处理时间


def penalize_service_failures_and_drops(T_m_k_uplink_c_matrix_t, T_requirement_matrix, channel, T_p, weight):
    """
    Calculates utility and penalizes service failures and drops in a network simulation.

    Parameters:
    - T_m_k_uplink_c_matrix_t: 2D numpy array representing the service times.
    - T_requirement_matrix: 2D numpy array of service time requirements.
    - channel: 1D or 2D numpy array indicating the channel option for users; 0 indicates dropped service.
    - T_p: Penalty value to be applied for failures and drops.
    - weight: Array or scalar that scales the penalty value based on specific conditions.

    Returns:
    - utility_matrix: 2D numpy array containing utilities for services that met requirements, 
      and penalties for failures and dropped services.
    """
    # Mask to identify non-zero entries in channel (service not dropped)
    mask = channel != 0
    
    # Initialize utility matrix
    utility_matrix = np.zeros_like(T_m_k_uplink_c_matrix_t, dtype=float)
    
    # Mask to identify services that met the requirements
    success_mask = (T_m_k_uplink_c_matrix_t <= T_requirement_matrix) & mask
    utility_matrix[success_mask] = T_requirement_matrix[success_mask] - T_m_k_uplink_c_matrix_t[success_mask]
    
    # Mask to identify service failures
    failures_mask = (T_m_k_uplink_c_matrix_t > T_requirement_matrix) & mask
    utility_matrix[failures_mask] = -T_p * weight.flatten()[failures_mask]
    
    # Penalize the dropped services (channel == 0)
    utility_matrix[~mask] = -T_p * weight.flatten()[~mask]
    
    return utility_matrix

def analyze_and_penalize(T_m_k_uplink_c_matrix_t, T_requirement_matrix, channel, T_p, weight):
    """
    Penalizes service failures and dropped services in a network simulation.

    Parameters:
    - T_m_k_uplink_c_matrix_t: Array representing the service times.
    - T_requirement_matrix: Matrix of service time requirements.
    - channel_opt: Array indicating the channel option for users; 0 indicates dropped service.
    - t: Time index for which the requirement matrix is considered.
    - T_p: Penalty value to be applied.

    Returns:
    - Updated T_m_k_uplink_c_matrix_t after applying penalties.
    """
    # Mask to identify non-zero entries in channel (service not dropped)
    mask = channel != 0
    
    # Initialize utility matrix
    utility_matrix = np.zeros_like(T_m_k_uplink_c_matrix_t, dtype=float)

    # Initialize the status matrix with all 1s
    status_matrix = np.ones_like(T_m_k_uplink_c_matrix_t, dtype=int)
    
    # Mask to identify services that met the requirements
    success_mask = (T_m_k_uplink_c_matrix_t <= T_requirement_matrix) & mask
    utility_matrix[success_mask] = T_requirement_matrix[success_mask] - T_m_k_uplink_c_matrix_t[success_mask]
    
    # Mask to identify service failures
    failures_mask = (T_m_k_uplink_c_matrix_t > T_requirement_matrix) & mask
    utility_matrix[failures_mask] = -T_p * weight.flatten()[failures_mask]
    
    # Set service but failed entries to 0
    status_matrix[failures_mask] = 0  #服务但失败

    # Penalize the dropped services (channel == 0)
    utility_matrix[~mask] = -T_p * weight.flatten()[~mask]
    
    num_service_but_failure = np.sum(failures_mask)
    # Count the number of dropped services (channel_opt == 0)
    num_not_service = np.sum(~mask)  # ~mask gives the inverse of the mask, identifying zeros in channel_opt
    # Sum of not meeting requirements (either not serviced or service failed to meet requirement)
    num_not_requirement = num_service_but_failure + num_not_service
    return utility_matrix, num_service_but_failure, num_not_service, num_not_requirement, status_matrix

def optimal_compute_resource_allocation(m, cpu_matrix, task_matrix, cpu_cycles, cell_mapping,
                                        users_channel):
    """
    compute allocation

    :param m: 小区数量
    :param cpu_matrix: 任务计算强度矩阵
    :param task_matrix: 任务矩阵
    :param cpu_cycles: MEC计算资源
    :param cell_mapping: 用户小区映射矩阵
    :param users_channel: 用户所选信道编号
    :return: T_m_k_c_matrix: 每个用户计算所用时间 x_m_k: 每个用户分配计算资源比例
    """

    t_m_k_c_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户在最优的分配比例下的计算时间，矩阵大小为（用户数）
    x_m_k_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户的计算资源分配比例

    for target_cell in range(m):
        # 筛选出当前帧、当前小区、信道不为0的用户
        selected_users = (cell_mapping == target_cell) & (users_channel != 0)

        X_m_k = cpu_matrix[selected_users]  # 筛选出的用户的每bit数据所需的CPU轮数大小
        L_m_k = task_matrix[selected_users]  # 筛选出的用户的任务的bit大小

        if len(X_m_k) > 0:  # 如果选中的用户不为空
            x_m_k = compute_allocation(X_m_k, L_m_k)  # 采用闭式解进行计算资源划分
            T_m_k_c = (L_m_k * X_m_k) / (x_m_k * cpu_cycles)  # 计算用户的计算时间

            t_m_k_c_matrix[selected_users] = T_m_k_c  # 存储计算时间至原始位置
            x_m_k_matrix[selected_users] = x_m_k  # 存储计算资源分配比例至原始位置

    return t_m_k_c_matrix, x_m_k_matrix


def mean_compute_resource_allocation(m, cpu_matrix, task_matrix, cpu_cycles, cell_mapping,
                                     users_channel):
    """
    compute allocation (mean)

    :param m: 小区数量
    :param cpu_matrix: 任务计算强度矩阵
    :param task_matrix: 任务矩阵
    :param cpu_cycles: MEC计算资源
    :param cell_mapping: 用户小区映射矩阵
    :param users_channel: 用户所选信道编号
    :return: T_m_k_c_matrix: 每个用户计算所用时间 x_m_k: 每个用户分配计算资源比例
    """

    t_m_k_c_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户在最优的分配比例下的计算时间，矩阵大小为（用户数）
    x_m_k_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户的计算资源分配比例

    for target_cell in range(m):
        # 筛选出当前帧、当前小区、信道不为0的用户
        selected_users = (cell_mapping == target_cell) & (users_channel != 0)

        X_m_k = cpu_matrix[selected_users]  # 筛选出的用户的每bit数据所需的CPU轮数大小
        L_m_k = task_matrix[selected_users]  # 筛选出的用户的任务的bit大小

        if len(X_m_k) > 0:  # 如果选中的用户不为空
            x_m_k_mean = np.full(len(L_m_k), 1.0 / len(L_m_k))  # 采用均分方式进行计算资源划分
            T_m_k_c_mean = (L_m_k * X_m_k) / (x_m_k_mean * cpu_cycles)

            t_m_k_c_matrix[selected_users] = T_m_k_c_mean  # 存储计算时间至原始位置
            x_m_k_matrix[selected_users] = x_m_k_mean  # 存储计算资源分配比例至原始位置

    return t_m_k_c_matrix, x_m_k_matrix


def update_weights(channel_history, beta):
    """
    Update priority weights based on the history of channel selections, with weight formula adjusted per frame.

    Parameters:
    - channel_history: A list of 2D numpy arrays where each array represents the channel selections for a specific frame.
                       Values of 0 indicate the user was denied service.
    - beta: A hyperparameter that influences the sensitivity of the weight increase as the denial count increases.

    Returns:
    - weights: A numpy array containing the updated weights for each user, calculated from the most recent frame.
    """
    # Convert list of channel histories into a 3D numpy array
    channel_array = np.array(channel_history)
    
    # Calculate total frames (i)
    total_frames = len(channel_history)
    
    # Count denials across all frames (where channel == 0)
    denials = (channel_array == 0).sum(axis=0)
    
    # Initialize weights array
    weights = np.ones_like(denials, dtype=float)
    
    # For indices where there have been denials, calculate weights using the exponential function
    denied_indices = denials > 0
    if total_frames > 0:  # Avoid division by zero if there are no frames
        weights[denied_indices] = np.exp(beta * denials[denied_indices] / total_frames)

    return weights.reshape(-1,1)

def generate_user_order(u, num_samples):
    # 初始化一个 (num_samples, u) 形状的矩阵
    user_order_matrix = np.zeros((u, num_samples), dtype=int)
    
    # 生成用户ID序列
    user_ids = np.arange(u)
    
    # 为每个时隙生成随机用户顺序
    for i in range(num_samples):
        np.random.shuffle(user_ids)
        user_order_matrix[:, i] = user_ids
    
    return user_order_matrix

def queue_computing_delay(m, cpu_matrix, task_matrix,cpu_cycles, cell_mapping,users_channel,users_uplink_delay,queue_number):
    
    t_m_k_c_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户在最优的分配比例下的计算时间，矩阵大小为（用户数）
    t_m_k_queue_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户在最优的分配比例下的排队时间，矩阵大小为（用户数）

    for target_cell in range(m):
        # 1. 筛选出当前帧、当前小区、信道不为0的用户
        selected_users = (cell_mapping == target_cell) & (users_channel != 0)
        user_indices = np.where(selected_users)[0]

        # 2. 基于上传延迟进行FIFO排队
        selected_users_uplink_delays = users_uplink_delay[selected_users]
        sorted_indices = user_indices[np.argsort(selected_users_uplink_delays)] # 按延迟升序排列的用户索引

        # 当前小区的处理开始时间（初始为0）
        current_process_time = 0.0  # 记录 MEC 服务器处理完前一批次的时间，确保后续批次不会提前开始

        # 4. 按queue_number分批处理用户（模拟并行处理队列，类似于MEC可以同时处理queue_number个任务）
        for i in range(0, len(sorted_indices), queue_number):    # 每次取出queue_number个用户执行下面的操作

            batch_indices = sorted_indices[i:i+queue_number]

            # 当前批次每个任务的到达时间（上传完成时间）
            arrival_times = users_uplink_delay[batch_indices]
            
            # 计算批次开始时间：当前处理时间 vs 最早到达时间，取较大值
            batch_start_time = max(current_process_time, np.min(arrival_times)) # 正确处理任务到达时间不一致的情况。
            
            batch_size = len(batch_indices)
            
            if batch_size == 0:
                continue  # 空批次跳过
            
            # 5. 获取当前批次用户的计算资源需求和任务大小
            X_m_k = cpu_matrix[batch_indices]
            L_m_k = task_matrix[batch_indices]

            # 6. 计算资源分配和计算时间
            x_m_k = compute_allocation(X_m_k, L_m_k)  # 计算资源划分
            T_m_k_c = (L_m_k * X_m_k) / (x_m_k * cpu_cycles)  # 计算用户的计算时间
            
            # 7. 存储计算时间至结果矩阵的对应位置
            t_m_k_c_matrix[batch_indices] = T_m_k_c

            # 计算每个任务的排队延迟：批次开始时间 - 到达时间
            queue_delays = np.maximum(batch_start_time - arrival_times, 0)  #若任务到达时MEC空闲则排队时延为0，否则为等待时间
            
            # 8. 计算排队时间：当前批次开始时的累计排队时间
            t_m_k_queue_matrix[batch_indices] = queue_delays  

            # 更新当前处理时间：批次开始时间 + 最大计算延迟
            current_process_time = batch_start_time + np.max(T_m_k_c) # 同一批次内的任务被视为并行处理，总耗时为最长任务的计算时间

    return t_m_k_c_matrix, t_m_k_queue_matrix


# 回复审稿意见
def compute_computing_delay(m, cpu_matrix, task_matrix, cpu_cycles, cell_mapping,
                                        users_channel,users_computing_ra):
    """
    compute allocation

    :param m: 小区数量
    :param cpu_matrix: 任务计算强度矩阵
    :param task_matrix: 任务矩阵
    :param cpu_cycles: MEC计算资源
    :param cell_mapping: 用户小区映射矩阵
    :param users_channel: 用户所选信道编号
    :param users_computing_ra: 每个用户的计算资源比例
    :return: T_m_k_c_matrix: 每个用户计算所用时间 x_m_k: 每个用户的计算资源比例
    """

    t_m_k_c_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户在最优的分配比例下的计算时间，矩阵大小为（用户数）
    x_m_k_matrix = np.zeros(cpu_matrix.shape[0])  # 存储每个用户的计算资源分配比例

    for target_cell in range(m):
        # 筛选出当前帧、当前小区、信道不为0的用户
        selected_users = (cell_mapping == target_cell) & (users_channel != 0)

        X_m_k = cpu_matrix[selected_users]  # 筛选出的用户的每bit数据所需的CPU轮数大小
        L_m_k = task_matrix[selected_users]  # 筛选出的用户的任务的bit大小
        x_m_k_temp = users_computing_ra[selected_users]  #筛选出当前小区被选用户的计算资源比例(临时的)
        
        if len(X_m_k) > 0:  # 如果选中的用户不为空
            s = x_m_k_temp.sum()
            if s <= 0:
                # 全为0：均分
                x_m_k = np.full(x_m_k_temp.shape, 1.0 / x_m_k_temp.size, dtype=float)
            else:
                # 归一化：和为1
                x_m_k = x_m_k_temp / s
                
            # x_m_k = compute_allocation(X_m_k, L_m_k)  # 采用闭式解进行计算资源划分
            T_m_k_c = (L_m_k * X_m_k) / (x_m_k * cpu_cycles)  # 计算用户的计算时间

            t_m_k_c_matrix[selected_users] = T_m_k_c  # 存储计算时间至原始位置
            x_m_k_matrix[selected_users] = x_m_k  # 存储计算资源分配比例至原始位置

    return t_m_k_c_matrix, x_m_k_matrix