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

# Parameters for the different scenarios
params_template = {
    'm_value': 7,
    'n_value': 5,
    'output_dir': "ceao_output",  # This is a placeholder directory
    'B': 20e6,
    'P_max': 23,
    'cpu_cycles': 15e9,
    'T_p': 0.01,
    'NIND': 500,
    'z_n': 2,
    'NOMA': True,
    'drop': True,
    'beta': 1
}


avg_utility_per_user_all = []
service_success_rate_all = []
# Iterating over different u_values
for i, u_value in enumerate([14, 28, 42, 56, 70]):
    params = params_template.copy()
    params['u_value'] = u_value   # Update the u_value for each scenario

    # Get the average utility per user for ceao
    avg_utility_per_user_ceao,service_success_rate_ceao = data_result_ceao(**params)
    avg_utility_per_user_all.append(avg_utility_per_user_ceao)
    service_success_rate_all.append(service_success_rate_ceao)
    
    params = params_template.copy()
    params['u_value'] = u_value   # Update the u_value for each scenario
    params['NOMA'] = False 
    params['z_n'] = 1
    # Get the average utility per user for ceao-OMA
    avg_utility_per_user_ceao,service_success_rate_ceao = data_result_ceao(**params)
    avg_utility_per_user_all.append(avg_utility_per_user_ceao)
    service_success_rate_all.append(service_success_rate_ceao)
    
    params = params_template.copy()
    params['u_value'] = u_value 
    params['drop'] = False 
    
    # Get the average utility per user for ceao-BE
    avg_utility_per_user_ceao_nodrop,service_success_rate_ceao_nodrop = data_result_ceao(**params)
    avg_utility_per_user_all.append(avg_utility_per_user_ceao_nodrop)
    service_success_rate_all.append(service_success_rate_ceao_nodrop)

    params = params_template.copy()
    params['u_value'] = u_value 
    params['output_dir'] = 'ejo_output'  # Update the u_value for each scenario

    # Get the average utility per user for ejo
    avg_utility_per_user_ejo,service_success_rate_ejo = data_result_ejo(**params)
    avg_utility_per_user_all.append(avg_utility_per_user_ejo)
    service_success_rate_all.append(service_success_rate_ejo)
    
    params = params_template.copy()
    params['u_value'] = u_value 
    params['output_dir'] = 'fura_output'  # Update the u_value for each scenario
    params['drop'] = False 
    # Get the average utility per user for fura
    avg_utility_per_user_fura,service_success_rate_fura = data_result_ejo(**params)
    avg_utility_per_user_all.append(avg_utility_per_user_fura)
    service_success_rate_all.append(service_success_rate_fura)
    
    params = params_template.copy()
    params['u_value'] = u_value 
    params['output_dir'] = 'lura_output'  # Update the u_value for each scenario
    params['drop'] = False 
    # Get the average utility per user for lura
    avg_utility_per_user_lura,service_success_rate_lura = data_result_ejo(**params)
    avg_utility_per_user_all.append(avg_utility_per_user_lura)
    service_success_rate_all.append(service_success_rate_lura)

u_values = ['U=14','U=28', 'U=42', 'U=56', 'U=70']
# 计算负载率百分比
load_rates_percentage = [u_value / params_template['n_value'] / params_template['m_value'] * 100 for u_value in [14, 28, 42, 56, 70]]

# 重新定义标签，将负载率百分比添加到标签中
u_values_with_load = [f"{load_rates_percentage[i]:.0f}%\n({u_values[i]})"
                      for i in range(len(u_values))]

algorithms = ['JUSRA-CEAO','JUSRA-CEAO-OMA','NURA-CEAO', 'EJO','FURA','LURA']

# 循环处理每个 U 值
for u_index in range(len(u_values)):
    base_index = u_index * len(algorithms)
    ceao_utility = avg_utility_per_user_all[base_index]
    ceao_success_rate = service_success_rate_all[base_index]

    print(f"\nComparisons for {u_values[u_index]}:")

    # 比较 CEAO 与其他每种算法
    for alg_index, alg_name in enumerate(algorithms[1:], 1):  # 从1开始，跳过 CEAO
        other_utility = avg_utility_per_user_all[base_index + alg_index]
        other_success_rate = service_success_rate_all[base_index + alg_index]

        # 计算差异百分比
        if other_utility != 0:
            utility_Increase = (ceao_utility - other_utility) / other_utility * 100
        else:
            utility_Increase = 0

        if other_success_rate != 0:
            success_rate_increase = (ceao_success_rate - other_success_rate) / other_success_rate * 100
        else:
            success_rate_increase = 0

        # 打印结果
        print(f"  Compared to {alg_name}:")
        print(f"    Average Utility Increase: {utility_Increase:.2f}%")
        print(f"    Success Rate Increase: {success_rate_increase:.2f}%")




line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]  # Line styles
x = np.arange(len(u_values_with_load))  # the label locations
width = 0.12  # the width of the bars

# Create plot for average utility per user
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    alg_positions = x - (len(algorithms) * width / 2) + i * width + width / 2
    data = avg_utility_per_user_all[i::len(algorithms)]
    plt.bar(alg_positions, data, width=width, color=colors[i],label=alg,align='center')

plt.xlabel('Load',fontsize=16)
plt.ylabel('LA-AveUSD',fontsize=16)
# plt.title('Average Utility per User by M Value and Algorithm')
plt.xticks(x, u_values_with_load)
plt.legend()
plt.tight_layout()
# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')  # Adjusted for the current environment
# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "U_Value_Utility.svg"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)

plt.show()
# Create plot for service success rate
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    alg_positions = x + i * width - width
    alg_positions = x - (len(algorithms) * width / 2) + i * width + width / 2
    data = service_success_rate_all[i::len(algorithms)]
    plt.bar(alg_positions, data, width=width, color=colors[i],label=alg,align='center')

plt.xlabel('Load',fontsize=16)
plt.ylabel('Service Success Rate',fontsize=16)
# plt.title('Service Success Rate by M Value and Algorithm')
plt.xticks(x, u_values_with_load)
plt.legend()
plt.tight_layout()
filename = "U_Value_Success.svg"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()


#############################################折线图################################
# 创建平均效用值的折线图
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    data = avg_utility_per_user_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i],  marker=markers[i],  color=colors[i],label=alg, fillstyle='none')

plt.xlabel('Load', fontsize=16)
plt.ylabel('LA-AveUSD', fontsize=16)
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存平均效用值的折线图
save_dir = './fig'  # 确保这个目录存在或者用 os.makedirs 创建它
filename = "U_Value_Utility_Line.svg"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
print(os.path.join(save_dir, filename))

plt.show()

# 创建服务成功率的折线图
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    data = service_success_rate_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i],  marker=markers[i],  color=colors[i],label=alg, fillstyle='none')

plt.xlabel('Load', fontsize=16)
plt.ylabel('Service Success Rate', fontsize=16)
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存服务成功率的折线图
filename = "U_Value_Success_Line.svg"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
print(os.path.join(save_dir, filename))

plt.show()


#############################################总效用折线图################################

total_utility_all = []
U_values = [14, 28, 42, 56, 70]
# 计算总效用
for i, u_value in enumerate(U_values):
    for j in range(len(algorithms)):
        index = i * len(algorithms) + j
        total_utility = avg_utility_per_user_all[index] * u_value
        total_utility_all.append(total_utility)

x = np.arange(len(U_values))  # 标签位置

# 创建总效用的折线图
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    data = total_utility_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i],  marker=markers[i],  color=colors[i], label=alg, fillstyle='none')

plt.xlabel('Load', fontsize=16)
plt.ylabel('LA-USD', fontsize=16)
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存图表
save_dir = './fig'  # 确保这个目录存在或者用 os.makedirs 创建它
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "U_Value_Total_Utility.svg"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
print(os.path.join(save_dir, filename))

plt.show()

##################################################################################除去EJO#########################################################################
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 设置字体大小增加8个字号
default_font_size = 12  # 您可以根据需要调整默认字体大小
font_size_increment = 6
new_font_size = default_font_size + font_size_increment

# 更新rcParams以直接设置字体大小
rcParams.update({
    'font.size': new_font_size,
    'axes.titlesize': new_font_size,
    'axes.labelsize': new_font_size,
    'xtick.labelsize': new_font_size,
    'ytick.labelsize': new_font_size,
    'legend.fontsize': new_font_size,
    'figure.titlesize': new_font_size
})

# 创建平均效用值的折线图，不包括 EJO
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    if alg == 'EJO':  # 跳过 EJO
        continue
    data = avg_utility_per_user_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=alg, fillstyle='none')

plt.xlabel('Load')
plt.ylabel('DA-AveUSD')
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存平均效用值的折线图
save_dir = './fig'  # 确保这个目录存在或者用 os.makedirs 创建它
filename = "U_Value_Utility_Line_No_EJO.eps"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=500)
print(os.path.join(save_dir, filename))

plt.show()

# 创建服务成功率的折线图，不包括 EJO
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    if alg == 'EJO':  # 跳过 EJO
        continue
    data = service_success_rate_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=alg, fillstyle='none')

plt.xlabel('Load')
plt.ylabel('Service Success Rate')
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存服务成功率的折线图
filename = "U_Value_Success_Line_No_EJO.eps"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=500)
print(os.path.join(save_dir, filename))

plt.show()

# 创建总效用的折线图，不包括 EJO
plt.figure(figsize=(8, 6))
for i, alg in enumerate(algorithms):
    if alg == 'EJO':  # 跳过 EJO
        continue
    data = total_utility_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=alg, fillstyle='none')

plt.xlabel('Load')
plt.ylabel('DA-USD')
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存总效用折线图
filename = "U_Value_Total_Utility_No_EJO.eps"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=500)
print(os.path.join(save_dir, filename))

plt.show()
