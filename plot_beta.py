import matplotlib.pyplot as plt
import numpy as np
import os
import project_backend as pb
# Nature Inspired Colors
colors = [
    "#E64B35FF",# Red
    "#66C2A5FF",  # Teal
    "#3C5488FF",  # Indigo 
    "#FAA9A3FF",  # Pink
    "#999999FF",
    "#F0A45BFF",
    
    "#E69F00FF",
    "#4DBBD5FF",  # Blue
    "#91D1C2FF",  # Aquamarine
    "#7E7E7EFF",
    "#8491B4FF",  # Periwinkle
    "#00A087FF",
]
# Hollow markers
markers = ['o', 's', '^', 'd', 'h', '*', '+']
# Set the font to Times New Roman for all text in the plot

# 设置全局字体大小
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
# We will use the provided 'data_result' function template to simulate loading of data for different 'u_value'
# The function 'load_data' within 'data_result' should handle the data retrieval, we will simulate it here.
    # Simulate the loading of data
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping...")
        return None
    return np.load(file_path, allow_pickle=True)

def data_result_ceao(**kwargs):
    # Extract the required parameters from kwargs
    params = {k: kwargs.get(k) for k in
              ['m_value', 'u_value', 'n_value', 'output_dir', 'B', 'P_max', 'cpu_cycles', 'T_p', 'NIND', 'z_n', 'NOMA', 'drop','beta']}
    json_file = f'M{params["m_value"]}_U{params["u_value"]}_N{params["n_value"]}_Task_u15000_Ucpu_e1000_Tdemand_u15_fix'
    output_filename = f"{json_file}_B{params['B'] / 1e6}_Pmax{params['P_max']}_Cpu{params['cpu_cycles'] / 1e9}_Tp{params['T_p']}_z_n{params['z_n']}_NIND{params['NIND']}_NOMA{params['NOMA']}_drop{params['drop']}_beta{params['beta']}.npz"

    # For CEAO data
    output_file_ceao = os.path.join(params['output_dir'], output_filename)
    data_ceao = load_data(output_file_ceao)
    if data_ceao is not None:
        T_m_k_uplink_c_matrix_t = data_ceao['T_m_k_uplink_matrix'] + data_ceao['T_m_k_c_matrix']
        T_requirement_matrix = data_ceao['T_requirement_matrix']
        channel_opt = data_ceao['user_channel_opt_all']
        num_sum_not_requirement = 0  # 用于运行过程中显示截至第t个时隙的失败结果的中间变量
        Utility_m_k_uplink_c_matrix_copy = 0  # 
        num_sum_not_service = 0  # 用于运行过程中显示截至第t个时隙的未服务用户数的中间变量
        num_sum_service_but_failure = 0  # 用于运行过程中显示截至第t个时隙的服务但是失败的用户数中间变量
        weighted_initial = np.ones((params["u_value"],1))  # 初始化用户权重为1
        for t in range(len(channel_opt)):
            if t == 0:
                weighted = weighted_initial
            else:
                weighted = weighted_previous
            Utility_m_k_uplink_c_matrix_t, num_service_but_failure, num_not_service, num_not_requirement,_ = pb.analyze_and_penalize(
            T_m_k_uplink_c_matrix_t[:,t], T_requirement_matrix[:, t], channel_opt[t], params['T_p'], weighted)
            num_sum_service_but_failure += num_service_but_failure
            num_sum_not_service += num_not_service
            num_sum_not_requirement += num_not_requirement
            Utility_m_k_uplink_c_matrix_copy += np.mean(Utility_m_k_uplink_c_matrix_t)
            weighted_previous = pb.update_weights(channel_opt[:t],params['beta'])
    # Simulate removing the second element and normalizing by the number of users
    service_success_rate= 1- (num_sum_not_requirement/len(channel_opt)/params['u_value'])
    avg_utility_per_user = 1000 * Utility_m_k_uplink_c_matrix_copy/len(channel_opt)  # 变成ms

    return avg_utility_per_user,service_success_rate,channel_opt

def data_result_ejo(**kwargs):
    # Extract the required parameters from kwargs
    params = {k: kwargs.get(k) for k in
              ['m_value', 'u_value', 'n_value', 'output_dir', 'B', 'P_max', 'cpu_cycles', 'T_p', 'NIND', 'z_n', 'NOMA', 'drop','beta']}
    json_file = f'M{params["m_value"]}_U{params["u_value"]}_N{params["n_value"]}_Task_u15000_Ucpu_e1000_Tdemand_u15_fix'
    output_filename = f"{json_file}_B{params['B'] / 1e6}_Pmax{params['P_max']}_Cpu{params['cpu_cycles'] / 1e9}_Tp{params['T_p']}_z_n{params['z_n']}_NIND{params['NIND']}_NOMA{params['NOMA']}_drop{params['drop']}_beta{params['beta']}.npz"

    # For ejo data
    output_file_ejo = os.path.join(params['output_dir'], output_filename)
    data_ejo = load_data(output_file_ejo)
    if data_ejo is not None:
        T_m_k_uplink_c_matrix_t = data_ejo['T_m_k_uplink_matrix'] + data_ejo['T_m_k_c_matrix']
        T_requirement_matrix = data_ejo['T_requirement_matrix']
        channel_opt = data_ejo['user_channel_opt_all']
        num_sum_not_requirement = 0  # 用于运行过程中显示截至第t个时隙的失败结果的中间变量
        Utility_m_k_uplink_c_matrix_copy = 0  # 
        num_sum_not_service = 0  # 用于运行过程中显示截至第t个时隙的未服务用户数的中间变量
        num_sum_service_but_failure = 0  # 用于运行过程中显示截至第t个时隙的服务但是失败的用户数中间变量
        weighted_initial = np.ones((params["u_value"],1))  # 初始化用户权重为1
        for t in range(len(channel_opt)):
            if t == 0:
                weighted = weighted_initial
            else:
                weighted = weighted_previous
            Utility_m_k_uplink_c_matrix_t, num_service_but_failure, num_not_service, num_not_requirement,_ = pb.analyze_and_penalize(
            T_m_k_uplink_c_matrix_t[:,t], T_requirement_matrix[:, t], channel_opt[t], params['T_p'],weighted)
            num_sum_service_but_failure += num_service_but_failure
            num_sum_not_service += num_not_service
            num_sum_not_requirement += num_not_requirement
            Utility_m_k_uplink_c_matrix_copy += np.mean(Utility_m_k_uplink_c_matrix_t)
            weighted_previous = pb.update_weights(channel_opt[:t],params['beta'])
    # Simulate removing the second element and normalizing by the number of users
    service_success_rate= 1- (num_sum_not_requirement/len(channel_opt)/params['u_value'])
    avg_utility_per_user = 1000 * Utility_m_k_uplink_c_matrix_copy/len(channel_opt)  # 变成ms

    return avg_utility_per_user,service_success_rate,channel_opt

params_template = {
    'u_value': 42,
    'm_value': 3,
    'n_value': 5,
    'output_dir': "ceao_output",  # General output directory
    'P_max': 23,  # Typical P_max value
    'cpu_cycles': 15e9,
    'T_p': 0.01,
    'NIND': 500,
    'z_n': 2,
    'NOMA': True,
    'drop': True,
    'B':20e6
}

beta_values = [0, 1, 2]
results = {'CEAO-BE': {}, 'CEAO-OMA': {}, 'CEAO': {}, 'EJO':{}, 'FURA':{}, 'LURA':{}}
service_rates = {'CEAO-BE': {}, 'CEAO-OMA': {}, 'CEAO': {}, 'EJO':{}, 'FURA':{}, 'LURA':{}}
jain_indices = []


def jain_index(service_counts):
    """
    计算 Jain's Fairness Index
    :param service_counts: 用户被服务次数的数组
    :return: Jain's Fairness Index
    """
    n = len(service_counts)
    sum_of_squares = np.sum(service_counts ** 2)
    square_of_sum = np.sum(service_counts) ** 2
    if sum_of_squares == 0:
        return 0  # 避免除以零
    return square_of_sum / (n * sum_of_squares)


import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 设置字体大小增加8个字号
default_font_size = 12  # 您可以根据需要调整默认字体大小
font_size_increment = 6
new_font_size = default_font_size + font_size_increment

# 更新 rcParams 以直接设置字体大小
rcParams.update({
    'font.size': new_font_size,
    'axes.titlesize': new_font_size,
    'axes.labelsize': new_font_size,
    'xtick.labelsize': new_font_size,
    'ytick.labelsize': new_font_size,
    'legend.fontsize': new_font_size,
    'figure.titlesize': new_font_size
})

# 绘制CDF曲线
plt.figure(figsize=(8, 6))
for beta in beta_values:
    params = params_template.copy()
    params.update({'beta': beta})

    # # CEAO-BE scenario: drop is False
    # params.update({'drop': False})
    # avg_utility, service_rate = data_result_ceao(**params)
    # results['CEAO-BE'][beta] = avg_utility
    # service_rates['CEAO-BE'][beta] = service_rate

    # # CEAO-OMA: NOMA is False, drop reset to True for safety
    # params.update({'NOMA': False, 'drop': True})
    # avg_utility, service_rate = data_result_ceao(**params)
    # results['CEAO-OMA'][beta] = avg_utility
    # service_rates['CEAO-OMA'][beta] = service_rate

    # Standard CEAO: NOMA is True, ensuring 'drop' is True
    params.update({'NOMA': True, 'drop': True})
    avg_utility, service_rate, channel_opt = data_result_ceao(**params)
    results['CEAO'][beta] = avg_utility
    service_rates['CEAO'][beta] = service_rate
    # 统计每个用户在100个时隙被服务的次数
    user_service_counts = np.sum(channel_opt != 0, axis=0)

    # 计算CDF
    sorted_counts = np.sort(user_service_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)


    

    jain_index_value = jain_index(user_service_counts)
    jain_indices.append((beta, jain_index_value))
    plt.plot(sorted_counts, cdf, marker='o',label=f'$\mathrm{{\\beta}}$={beta}, Jain Index={jain_index_value:.2f}')
    # # EJO : NOMA is True, ensuring 'drop' is True
    # params.update({'output_dir': "ejo_output"})
    # avg_utility, service_rate = data_result_ejo(**params)
    # results['EJO'][beta] = avg_utility
    # service_rates['EJO'][beta] = service_rate
    
    # # fura : NOMA is True, ensuring 'drop' is False
    # params.update({'output_dir': "fura_output",'drop': False})
    # avg_utility, service_rate = data_result_ejo(**params)
    # results['FURA'][beta] = avg_utility
    # service_rates['FURA'][beta] = service_rate
    
    # # lura : NOMA is True, ensuring 'drop' is False
    # params.update({'output_dir': "lura_output"})
    # avg_utility, service_rate = data_result_ejo(**params)
    # results['LURA'][beta] = avg_utility
    # service_rates['LURA'][beta] = service_rate
    
plt.xlabel('Number of Times Users are Served')
plt.ylabel('CDF')
# plt.title('CDF of Times Users are Served in 100 Slots')
plt.legend()
plt.grid(False)   
# 保存图表
save_dir = './fig'  # 确保这个目录存在或者用 os.makedirs 创建它
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "beta1.eps"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=500)
print(os.path.join(save_dir, filename))
plt.show()
print(jain_indices)






colors = [ 'red', 'green','blue']
plt.figure(figsize=(8, 6))
# 保存所有用户在不同beta下的服务次数
all_user_service_counts = []
for idx, beta in enumerate(beta_values):
    params = params_template.copy()
    params.update({'beta': beta})

    # 设置NOMA和drop参数
    params.update({'NOMA': True, 'drop': True})
    avg_utility, service_rate, channel_opt = data_result_ceao(**params)
    
    # 统计每个用户在100个时隙被服务的次数
    user_service_counts = np.sum(channel_opt != 0, axis=0)
    all_user_service_counts.append(user_service_counts)
    # 计算Jain's公平性指数
    jain_index_value = jain_index(user_service_counts)

    # 绘制散点图
    plt.scatter(range(len(user_service_counts)), user_service_counts, label=f'β={beta}, Jain Index={jain_index_value:.2f}', marker=markers[idx], facecolors='none', edgecolors=colors[idx])

# 转置以便按用户聚合数据
all_user_service_counts = np.array(all_user_service_counts).T

# 圈出三种beta值下服务次数均小于55的用户
for i, counts in enumerate(all_user_service_counts):
    if all(count < 55 for count in counts):
        # plt.plot([i, i, i], counts, 'o-', color='none', markersize=10, alpha=0.5)
        plt.plot([i]*len(counts), counts, 'o--', color='gray', markeredgecolor='gray', markerfacecolor='none', markersize=10, markeredgewidth=0.8, alpha=0.5)
plt.xlabel('User Index')
plt.ylabel('Number of Times Served')
plt.legend()
plt.grid(False)
# 保存图表
save_dir = './fig'  # 确保这个目录存在或者用 os.makedirs 创建它
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "beta2.eps"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=500)
print(os.path.join(save_dir, filename))
plt.show()

# 遍历不同的beta值
for beta in beta_values:
    # 获取CEAO的数据
    ceao_utility = results['CEAO'][beta]
    ceao_success_rate = service_rates['CEAO'][beta]

    print(f"At beta {beta}:")
    print(f"CEAO LA-AveUSD: {ceao_utility}")
    print(f"CEAO Success Rate: {ceao_success_rate}")
    print("------------------------------")
        

