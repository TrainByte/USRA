"patent"

import matplotlib.pyplot as plt
import numpy as np
import os
import project_backend as pb
# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# Nature Inspired Colors
colors = [
    "#000000"	
    # "#E64B35FF",# Red
    # "#66C2A5FF",  # Teal
    # "#3C5488FF",  # Indigo 
    # "#FAA9A3FF",  # Pink
    # "#999999FF",
    # "#F0A45BFF",
    
    # "#E69F00FF",
    # "#4DBBD5FF",  # Blue
    # "#91D1C2FF",  # Aquamarine
    # "#7E7E7EFF",
    # "#8491B4FF",  # Periwinkle
    # "#00A087FF",
]
# Hollow markers
markers = ['o', 's', '^', 'd', 'h', '*', '+']
# Set the font to Times New Roman for all text in the plot

# 设置全局字体大小
plt.rcParams.update({'font.size': 12, 'font.family': 'SimHei'})
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
    json_file = f'M{params["m_value"]}_U{params["u_value"]}_N{params["n_value"]}_Task_u15000_Ucpu_e1000_Tdemand_u15_slot1'
    output_filename = f"{json_file}_B{params['B'] / 1e6}_Pmax{params['P_max']}_Cpu{params['cpu_cycles'] / 1e9}_Tp{params['T_p']}_z_n{params['z_n']}_NIND{params['NIND']}_NOMA{params['NOMA']}_drop{params['drop']}_beta{params['beta']}.npz"

    # For CEAO data
    print("Current Working Directory:", os.getcwd())
    output_file_ceao = os.path.join(params['output_dir'], output_filename)
    data_ceao = load_data(output_file_ceao)
    if data_ceao is not None:
        sum_utility_list = data_ceao['sum_utility_list']
    # Simulate removing the second element and normalizing by the number of users
    T_requirement_matrix = data_ceao['T_requirement_matrix']
    print(sum_utility_list)
    sum_utility_list = np.delete(sum_utility_list, 1)
    avg_utility_per_user = 1000 * sum_utility_list / params['u_value']  # 变成ms

    return avg_utility_per_user,T_requirement_matrix

def data_result_exa(T_requirement_matrix,weighted_initial,**kwargs):
    # Extract the required parameters from kwargs
    params = {k: kwargs.get(k) for k in
              ['m_value', 'u_value', 'n_value', 'output_dir', 'B', 'P_max', 'cpu_cycles', 'T_p', 'NIND', 'z_n', 'NOMA', 'drop','beta']}
    json_file = f'M{params["m_value"]}_U{params["u_value"]}_N{params["n_value"]}_Task_u15000_Ucpu_e1000_Tdemand_u15_slot1'
    output_filename = f"{json_file}_B{params['B'] / 1e6}_Pmax{params['P_max']}_Cpu{params['cpu_cycles'] / 1e9}_Tp{params['T_p']}_z_n{params['z_n']}_NIND{params['NIND']}_NOMA{params['NOMA']}_drop{params['drop']}_beta{params['beta']}.npz"
    output_file_exa = os.path.join(params['output_dir'], output_filename)
    data_exa = load_data(output_file_exa)
    # T_requirement_matrix = data_exa["T_requirement_matrix"]
    sum_utility_list,_,_,_,_ = pb.analyze_and_penalize((data_exa['T_m_k_uplink_matrix'][:,0]+data_exa['T_m_k_c_matrix'][:,0]),T_requirement_matrix[:,0], data_exa['user_channel_opt_all'][0], params['T_p'],weighted_initial)
    avg_utility_per_user = 1000 * np.mean(sum_utility_list)# 变成ms
    return avg_utility_per_user
# Parameters for the different scenarios
params_template = {
    'm_value': 3,
    'n_value': 2,
    'output_dir': "ceao_output",  # This is a placeholder directory
    'B': 20e6,
    'P_max': 23,
    'cpu_cycles': 15e9,
    'T_p': 0.01,
    'NIND': 500,
    'z_n': 2,
    'NOMA': True,
    'drop': True,
    'beta':1
}


#############################################################去掉中间PA更新的数据 且 加圈和箭头########################
from matplotlib.patches import Ellipse
from matplotlib import rcParams
# Set all font sizes to be 8 points larger than default
default_font_size = 12  # You can adjust this as needed
font_size_increment = 6
new_font_size = default_font_size + font_size_increment

# Update the rcParams to set font sizes directly
rcParams.update({
    'font.size': new_font_size,
    'axes.titlesize': new_font_size,
    'axes.labelsize': new_font_size,
    'xtick.labelsize': new_font_size,
    'ytick.labelsize': new_font_size,
    'legend.fontsize': new_font_size,
    'figure.titlesize': new_font_size
})

# Plotting setup
plt.figure(figsize=(8, 6))
# Iterating over different u_values
for i, u_value in enumerate([9, 12, 15]):
    params = params_template.copy()
    params['u_value'] = u_value  # Update the u_value for each scenario
    weighted_initial = np.ones((u_value, 1))  # Initialize user weights to 1
    # Get the average utility per user
    avg_utility_per_user_ceao, T_requirement_matrix = data_result_ceao(**params)
    print(avg_utility_per_user_ceao)
    # Selecting color and marker
    color = colors[i % len(colors)]
    # Correcting the x-axis to start from 1
    x_axis = np.arange(1, len(avg_utility_per_user_ceao) + 1)
    # Modify avg_utility_per_user_ceao to plot every other data point
    avg_utility_per_user_ceao_reduced = avg_utility_per_user_ceao[::2]
    x_axis_reduced = np.arange(1, len(avg_utility_per_user_ceao_reduced) + 1)
    
    # EXA data
    params = params_template.copy()
    params['u_value'] = u_value  # Update the u_value for each scenario
    params['output_dir'] = 'exhaustive_search_output'
    # Get the average utility per user
    avg_utility_per_user_exa = data_result_exa(T_requirement_matrix, weighted_initial, **params)
    print(avg_utility_per_user_exa)
    
    ### 等跑完去除这句话
    # if i == 2: avg_utility_per_user_exa=7.1
    
    color_exa = colors[(i + 2) % len(colors)]  # Selecting a distinct color for EXA data
    
    # 绘制 ESA 数据（去掉 U）
    plt.plot(
        np.arange(1, len(avg_utility_per_user_ceao_reduced) + 1),
        np.repeat(avg_utility_per_user_exa, len(avg_utility_per_user_ceao_reduced)),
        label='JUSRA-ESA' if i == 2 else "",
        color=color_exa,
        marker="o",
        linestyle='--',
        fillstyle='none'
    )
    
    # 绘制 CEAO 数据（去掉 U）
    plt.plot(
        x_axis_reduced,
        avg_utility_per_user_ceao_reduced,
        label='JUSRA-CEAO' if i == 2 else "",  # 仅在第一次循环中添加图例，以避免重复
        color=color,
        marker="s",
        linestyle='-',
        fillstyle='none'
    )

    # 示例数据
    annotation_index = len(avg_utility_per_user_ceao_reduced) // 2
    x_center = x_axis_reduced[annotation_index]
    y_center = avg_utility_per_user_ceao_reduced[annotation_index]

    # 创建椭圆
    ellipse = Ellipse((x_center, y_center), width=0.2, height=1.2, edgecolor='black', fill=False)
    plt.gca().add_patch(ellipse)

    # 文字位置（右上方）
    text_x, text_y = x_center + 0.3, y_center - 1.6

    # 箭头从椭圆右上方边缘开始
    arrow_start_x = x_center + 0.05  # 手动偏移椭圆右下方
    arrow_start_y = y_center - 0.5  # 手动偏移椭圆右下方

    # 使用 plt.annotate 绘制箭头
    plt.annotate(
        f'U={u_value}', 
        xy=(arrow_start_x, arrow_start_y),  # 箭头起始点
        xytext=(text_x, text_y),  # 文字的位置
        arrowprops=dict(facecolor='black', arrowstyle='<-')  # 使用单向箭头
    )
    
    # Calculating gap percentage
    ceao_final = avg_utility_per_user_ceao[-1]
    exa_final = avg_utility_per_user_exa
    gap_percentage = abs(exa_final - ceao_final) / exa_final * 100
    print(f'For U={u_value}, the gap percentage of CEAO from EXA is {gap_percentage:.2f}%, '
          f'{"within acceptable range" if gap_percentage < 1 else "not within acceptable range"}.')

# Adjust x-axis ticks to show integers
plt.xticks(np.arange(min(x_axis_reduced), max(x_axis_reduced) + 1, 1.0))
# Adding labels
plt.xlabel('迭代轮数')
plt.ylabel('DA-AveUSD')

# Adding a legend to distinguish between CEAO and EXA data
plt.legend(bbox_to_anchor=(0.78, 0.06), loc='lower center', borderaxespad=0.)

# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_patent')
# filename = "convergence_annotate.svg"  # Saving as SVG format
filename = "patent_convergence_annotate.png"  # 以EPS格式保存
# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the figure in the specified directory
plt.savefig(os.path.join(save_dir, filename), dpi=500, bbox_inches='tight')

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)

# Showing the plot
plt.show()


"======================================================================================================================================================="
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
##################################################################################除去EJO#########################################################################
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
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
u_values_with_load = [f"({u_values[i]})"
                      for i in range(len(u_values))]

algorithms = ['JUSRA-CEAO','JUSRA-CEAO-OMA','NURA-CEAO', 'EJO','FURA','LURA']

# 设置字体大小增加8个字号
default_font_size = 12  # 您可以根据需要调整默认字体大小
font_size_increment = 6
new_font_size = default_font_size + font_size_increment
line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]  # Line styles
x = np.arange(len(u_values_with_load))  # the label locations
width = 0.12  # the width of the bars


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
    if alg == 'JUSRA-CEAO-OMA':  # 跳过 EJO
        continue
    if alg == 'NURA-CEAO':  # 跳过 EJO
        continue
    data = avg_utility_per_user_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=alg, fillstyle='none')

plt.xlabel('用户数')
plt.ylabel('DA-AveUSD')
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存平均效用值的折线图
save_dir = './fig_patent'  # 确保这个目录存在或者用 os.makedirs 创建它
filename = "patent_U_Value_Utility_Line_No_EJO.png"
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
    if alg == 'JUSRA-CEAO-OMA':  # 跳过 EJO
        continue
    if alg == 'NURA-CEAO':  # 跳过 EJO
        continue
    data = service_success_rate_all[i::len(algorithms)]
    plt.plot(u_values, data, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=alg, fillstyle='none')

plt.xlabel('用户数')
plt.ylabel('服务成功率（即被选用户数/总用户数）')
plt.xticks(x, u_values_with_load)
plt.legend()
plt.grid(False)
plt.tight_layout()

# 保存服务成功率的折线图
filename = "patent_U_Value_Success_Line_No_EJO.png"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=500)
print(os.path.join(save_dir, filename))

plt.show()



'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
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
plt.rcParams.update({'font.size': 12, 'font.family': 'SimHei'})
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
for i,beta in enumerate(beta_values):
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
    plt.plot(sorted_counts, cdf, marker=markers[i], markerfacecolor='none',label=f'$\mathrm{{\\beta}}$={beta}, Jain Index={jain_index_value:.2f}')
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
    
plt.xlabel('用户累计被服务次数')
plt.ylabel('CDF')
# plt.title('CDF of Times Users are Served in 100 Slots')
plt.legend()
plt.grid(False)   
# 保存图表
save_dir = './fig_patent'  # 确保这个目录存在或者用 os.makedirs 创建它
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename = "patent_beta1.png"
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=500)
print(os.path.join(save_dir, filename))
plt.show()
print(jain_indices)
        

