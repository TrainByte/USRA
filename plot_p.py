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
    'u_value': 42,
    'm_value': 7,
    'n_value': 5,
    'output_dir': "ceao_output",  # This is a placeholder directory
    'B': 20e6,
    'cpu_cycles': 15e9,
    'T_p': 0.01,
    'NIND': 500,
    'z_n': 2,
    'NOMA': True,
    'drop': True,
    'beta': 1
}


# Assuming data_result_ceao() and a similar function for CEAO-OMA are defined elsewhere
P_maxs = [5, 15, 25, 35, 45]
results = {'NURA-CEAO': {}, 'JUSRA-CEAO-OMA': {}, 'JUSRA-CEAO': {}, 'EJO':{}, 'FURA':{}, 'LURA':{}}
service_rates = {'NURA-CEAO': {}, 'JUSRA-CEAO-OMA': {}, 'JUSRA-CEAO': {}, 'EJO':{}, 'FURA':{}, 'LURA':{}}

for P_max in P_maxs:
    params = params_template.copy()
    params.update({'P_max': P_max})

    # CEAO-BE scenario: drop is False
    params.update({'drop': False})
    avg_utility, service_rate = data_result_ceao(**params)
    results['NURA-CEAO'][P_max] = avg_utility
    service_rates['NURA-CEAO'][P_max] = service_rate

    # CEAO-OMA: NOMA is False, drop reset to True for safety
    params.update({'NOMA': False, 'drop': True, 'z_n':1})
    avg_utility, service_rate = data_result_ceao(**params)  # If CEAO-OMA uses a different function, adjust accordingly
    results['JUSRA-CEAO-OMA'][P_max] = avg_utility
    service_rates['JUSRA-CEAO-OMA'][P_max] = service_rate

    # Standard CEAO: NOMA is True, ensuring 'drop' is True
    params.update({'NOMA': True, 'drop': True, 'z_n':2})
    avg_utility, service_rate = data_result_ceao(**params)
    results['JUSRA-CEAO'][P_max] = avg_utility
    service_rates['JUSRA-CEAO'][P_max] = service_rate
    
    # EJO : NOMA is True, ensuring 'drop' is True
    params.update({'NOMA': True, 'drop': True})
    params.update({'output_dir': "ejo_output"})
    avg_utility, service_rate = data_result_ejo(**params)
    results['EJO'][P_max] = avg_utility
    service_rates['EJO'][P_max] = service_rate
    
    # fura : NOMA is True, ensuring 'drop' is False
    params.update({'output_dir': "fura_output",'drop': False})
    avg_utility, service_rate = data_result_ejo(**params)
    results['FURA'][P_max] = avg_utility
    service_rates['FURA'][P_max] = service_rate
    
    # lura : NOMA is True, ensuring 'drop' is False
    params.update({'output_dir': "lura_output"})
    avg_utility, service_rate = data_result_ejo(**params)
    results['LURA'][P_max] = avg_utility
    service_rates['LURA'][P_max] = service_rate
    
# line_styles = ['-', '--', '-.',':']  # Line styles
line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]  # Line styles
# Plotting Average Utility per User
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO' ,'FURA','LURA']):
    avg_utility_values = [results[alg][P] for P in P_maxs]
    plt.plot(P_maxs, avg_utility_values, label=alg, color=colors[i], marker=markers[i], linestyle=line_styles[i], fillstyle='none')

plt.xlabel('Maximum Power (dBm)', fontsize=16)
plt.ylabel('LA-AveUSD', fontsize=16)
plt.xticks(P_maxs)  # Use P_maxs as is, since they're already in dBm
plt.legend()
plt.grid(False)
plt.tight_layout()
# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')  # Adjusted for the current environment
# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "Pmax_Value_Utility.svg"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution
# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()

# Plotting Service Success Rate
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO','FURA','LURA' ]):
    service_rate_values = [service_rates[alg][P] for P in P_maxs]
    plt.plot(P_maxs, service_rate_values, label=alg, color=colors[i], marker=markers[i], linestyle=line_styles[i], fillstyle='none')

plt.xlabel('Maximum Power (dBm)', fontsize=16)
plt.ylabel('Service Success Rate', fontsize=16)
plt.xticks(P_maxs)  # Use P_maxs as is, since they're already in dBm
# plt.title('Service Success Rate for Different Maximum Power Levels', fontsize=16)
plt.legend()
plt.grid(False)
plt.tight_layout()
filename = "Pmax_Value_Success.svg"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()


############################################################################################EJO#####################
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

# Plotting Average Utility per User, skipping EJO
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO', 'FURA', 'LURA']):  # 包含EJO, 但跳过绘制
    if alg == 'EJO':  # 跳过 EJO
        continue
    avg_utility_values = [results[alg][P] for P in P_maxs]
    plt.plot(P_maxs, avg_utility_values, label=alg, color=colors[i], marker=markers[i], linestyle=line_styles[i], fillstyle='none')

plt.xlabel('Maximum Power (dBm)')
plt.ylabel('DA-AveUSD')
plt.xticks(P_maxs)  # Use P_maxs as is, since they're already in dBm
plt.legend()
plt.grid(False)
plt.tight_layout()

# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')  # Adjusted for the current environment
# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "Pmax_Value_Utility_No_EJO.eps"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), dpi=500, bbox_inches='tight')  # Save as SVG with high resolution
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()

# Plotting Service Success Rate, skipping EJO
plt.figure(figsize=(8, 6))
for i, alg in enumerate(['JUSRA-CEAO', 'JUSRA-CEAO-OMA', 'NURA-CEAO', 'EJO', 'FURA', 'LURA']):  # 包含EJO, 但跳过绘制
    if alg == 'EJO':  # 跳过 EJO
        continue
    service_rate_values = [service_rates[alg][P] for P in P_maxs]
    plt.plot(P_maxs, service_rate_values, label=alg, color=colors[i], marker=markers[i], linestyle=line_styles[i], fillstyle='none')

plt.xlabel('Maximum Power (dBm)')
plt.ylabel('Service Success Rate')
plt.xticks(P_maxs)  # Use P_maxs as is, since they're already in dBm
plt.legend()
plt.grid(False)
plt.tight_layout()

filename = "Pmax_Value_Success_No_EJO.svg"  # Saving as SVG format
# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution
file_path = os.path.join(save_dir, filename)
print(file_path)
plt.show()

