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
    json_file = f'M{params["m_value"]}_U{params["u_value"]}_N{params["n_value"]}_Task_u15000_Ucpu_e1000_Tdemand_u15'
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

    return avg_utility_per_user,service_success_rate

def data_result_ejo(**kwargs):
    # Extract the required parameters from kwargs
    params = {k: kwargs.get(k) for k in
              ['m_value', 'u_value', 'n_value', 'output_dir', 'B', 'P_max', 'cpu_cycles', 'T_p', 'NIND', 'z_n', 'NOMA', 'drop','beta']}
    json_file = f'M{params["m_value"]}_U{params["u_value"]}_N{params["n_value"]}_Task_u15000_Ucpu_e1000_Tdemand_u15'
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

    return avg_utility_per_user,service_success_rate

params_template = {
    'u_value': 42,
    'm_value': 7,
    'n_value': 5,
    'output_dir': "ceao_output",  # General output directory
    'P_max': 23,  # Typical P_max value
    'cpu_cycles': 15e9,
    'T_p': 0.01,
    'NIND': 500,
    'z_n': 2,
    'NOMA': True,
    'drop': True,
    'beta':1
}

B_values = [20e6, 40e6, 60e6, 80e6, 100e6]
results = {'NURA-CEAO': {}, 'JUSRA-CEAO-OMA': {}, 'JUSRA-CEAO': {}, 'EJO':{}, 'FURA':{}, 'LURA':{}}
service_rates = {'NURA-CEAO': {}, 'JUSRA-CEAO-OMA': {}, 'JUSRA-CEAO': {}, 'EJO':{}, 'FURA':{}, 'LURA':{}}

for B in B_values:
    params = params_template.copy()
    params.update({'B': B})

    # NURA-CEAO scenario: drop is False
    params.update({'drop': False})
    avg_utility, service_rate = data_result_ceao(**params)
    results['NURA-CEAO'][B] = avg_utility
    service_rates['NURA-CEAO'][B] = service_rate

    # JUSRA-CEAO-OMA: NOMA is False, drop reset to True for safety
    params.update({'NOMA': False, 'drop': True, 'z_n':1})
    avg_utility, service_rate = data_result_ceao(**params)
    results['JUSRA-CEAO-OMA'][B] = avg_utility
    service_rates['JUSRA-CEAO-OMA'][B] = service_rate

    # Standard CEAO: NOMA is True, ensuring 'drop' is True
    params.update({'NOMA': True, 'drop': True, 'z_n':2})
    avg_utility, service_rate = data_result_ceao(**params)
    results['JUSRA-CEAO'][B] = avg_utility
    service_rates['JUSRA-CEAO'][B] = service_rate
    
    # EJO : NOMA is True, ensuring 'drop' is True
    params.update({'output_dir': "ejo_output"})
    avg_utility, service_rate = data_result_ejo(**params)
    results['EJO'][B] = avg_utility
    service_rates['EJO'][B] = service_rate
    
    # fura : NOMA is True, ensuring 'drop' is False
    params.update({'output_dir': "fura_output",'drop': False})
    avg_utility, service_rate = data_result_ejo(**params)
    results['FURA'][B] = avg_utility
    service_rates['FURA'][B] = service_rate
    
    # lura : NOMA is True, ensuring 'drop' is False
    params.update({'output_dir': "lura_output"})
    avg_utility, service_rate = data_result_ejo(**params)
    results['LURA'][B] = avg_utility
    service_rates['LURA'][B] = service_rate
    
    
# 定义算法名称
algorithms = ['JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO','FURA','LURA']  # 不包括CEAO本身

# 遍历不同的带宽值
for B in B_values:
    # 获取CEAO的数据
    ceao_utility = results['JUSRA-CEAO'][B]
    ceao_success_rate = service_rates['JUSRA-CEAO'][B]

    print(f"At Bandwidth {B/1e6} MHz:")

    # 逐一与其他算法比较
    for alg in algorithms:
        other_utility = results[alg][B]
        other_success_rate = service_rates[alg][B]

        # 计算平均效用的百分比降低
        if other_utility != 0:
            utility_reduction = (ceao_utility - other_utility) / other_utility * 100
        else:
            utility_reduction = 0  # 防止除以零

        # 计算服务成功率的百分比提升
        if other_success_rate != 0:
            success_rate_increase = (ceao_success_rate - other_success_rate) / other_success_rate * 100
        else:
            success_rate_increase = 0  # 防止除以零

        # 打印结果
        print(f"  Compared to {alg}:")
        # print(ceao_utility)
        # print(other_utility)
        print(f"    Average Utility increase: {utility_reduction:.2f}%")
        print(f"    Success Rate Increase: {success_rate_increase:.2f}%")
        
# line_styles = ['-', '--', '-.',':']  # Line styles
line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]  # Line styles


# Plotting Average Utility per User
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO','FURA','LURA']):
    avg_utility_values = [results[alg][B] for B in B_values]
    plt.plot([B/1e6 for B in B_values], avg_utility_values, label=alg, linestyle=line_styles[i], marker=markers[i], color=colors[i], fillstyle='none')

plt.xlabel('Bandwidth (MHz)', fontsize=16)
plt.ylabel('LA-AveUSD (ms)', fontsize=16)
plt.xticks([B/1e6 for B in B_values])  # Setting x-ticks to show values in MHz
plt.legend()
plt.grid(False)
plt.tight_layout()
# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')  # Adjusted for the current environment
# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "B_Value_Utility.svg"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution
# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()

# Plotting Service Success Rate
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO','FURA','LURA']):
    service_rate_values = [service_rates[alg][B] for B in B_values]
    plt.plot([B/1e6 for B in B_values], service_rate_values, label=alg, linestyle=line_styles[i], marker=markers[i], color=colors[i], fillstyle='none')

plt.xlabel('Bandwidth (MHz)', fontsize=16)
plt.ylabel('Service Success Rate', fontsize=16)
plt.xticks([B/1e6 for B in B_values])  # Setting x-ticks to show values in MHz
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.legend()
plt.grid(False)
plt.tight_layout()
filename = "B_Value_Success.svg"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()


###############################################################################去掉EJO##################################
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

# 绘制每个用户的平均效用，跳过 EJO
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO', 'FURA', 'LURA']):  # 包含 EJO 但在循环中跳过
    if alg == 'EJO':  # 跳过 EJO
        continue
    avg_utility_values = [results[alg][B] for B in B_values]
    plt.plot([B/1e6 for B in B_values], avg_utility_values, label=alg,
             linestyle=line_styles[i], marker=markers[i], color=colors[i], fillstyle='none')

plt.xlabel('Bandwidth (MHz)')
plt.ylabel('DA-AveUSD')
plt.xticks([B/1e6 for B in B_values])  # 设置 x 轴刻度显示为 MHz
plt.yticks(np.arange(11))  # y轴从0到10
plt.legend()
plt.grid(False)
plt.tight_layout()

# 设置保存图片的目录
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "B_Value_Utility_No_EJO.eps"  # 以 SVG 格式保存
# 保存图像
plt.savefig(os.path.join(save_dir, filename), dpi=500, bbox_inches='tight')
# 返回保存的图像路径
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()

# 绘制服务成功率，跳过 EJO
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO', 'FURA', 'LURA']):  # 包含 EJO 但在循环中跳过
    if alg == 'EJO':  # 跳过 EJO
        continue
    service_rate_values = [service_rates[alg][B] for B in B_values]
    plt.plot([B/1e6 for B in B_values], service_rate_values, label=alg,
             linestyle=line_styles[i], marker=markers[i], color=colors[i], fillstyle='none')

plt.xlabel('Bandwidth (MHz)')
plt.ylabel('Service Success Rate')
plt.xticks([B/1e6 for B in B_values])  # 设置 x 轴刻度显示为 MHz
plt.legend()
plt.grid(False)
plt.tight_layout()
filename = "B_Value_Success_No_EJO.eps"  # 以 SVG 格式保存
# 保存图像
plt.savefig(os.path.join(save_dir, filename), dpi=500, bbox_inches='tight')
# 返回保存的图像路径
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()