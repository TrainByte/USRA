import matplotlib.pyplot as plt
import numpy as np
import os
import project_backend as pb
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

# Plotting setup
plt.figure(figsize=(8, 6))

# Iterating over different u_values
for i, u_value in enumerate([12,15]):
    params = params_template.copy()
    params['u_value'] = u_value  # Update the u_value for each scenario
    weighted_initial = np.ones((u_value,1))  # 初始化用户权重为1
    # Get the average utility per user
    avg_utility_per_user_ceao,T_requirement_matrix = data_result_ceao(**params)
    print(avg_utility_per_user_ceao)
    # Selecting color and marker
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    # Correcting the x-axis to start from 1
    x_axis = np.arange(1, len(avg_utility_per_user_ceao) + 1)
    # Plotting the average utility per user against the iteration number
    plt.plot(x_axis, avg_utility_per_user_ceao, label=f'CEAO M={params_template["m_value"]} U={u_value} N={params_template["n_value"]}', color=color, marker=marker, linestyle='-', fillstyle='none')
    # exa_data
    params = params_template.copy()
    params['u_value'] = u_value  # Update the u_value for each scenario
    params['output_dir'] = 'exhaustive_search_output'
    # Get the average utility per user
    avg_utility_per_user_exa = data_result_exa(T_requirement_matrix,weighted_initial,**params)
    print(avg_utility_per_user_exa)
    color_exa = colors[(i+2) % len(colors)]  # Selecting a distinct color for EXA data
    marker_exa = markers[(i+2) % len(markers)]  # Selecting a distinct marker for EXA data
    plt.plot(np.arange(1, len(avg_utility_per_user_ceao)+1),np.repeat(avg_utility_per_user_exa, len(avg_utility_per_user_ceao)), label=f' ESA   M={params_template["m_value"]} U={u_value} N={params_template["n_value"]}', color=color_exa, marker=marker_exa, linestyle='--', fillstyle='none')

    # 计算差距百分比
    ceao_final = avg_utility_per_user_ceao[-1]
    exa_final = avg_utility_per_user_exa
    gap_percentage = abs(exa_final - ceao_final) / exa_final * 100
    print(f'For U={u_value}, the gap percentage of CEAO from EXA is {gap_percentage:.2f}%, {"within acceptable range" if gap_percentage < 1 else "not within acceptable range"}.')

# Adjust x-axis ticks to show integers
plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
# Adding title and labels
# plt.title('Comparison of CEAO (All u_values) and EXA (u_value=12) Data (Nature Inspired Colors with Hollow Markers)')
plt.xlabel('Iteration Number',fontsize=16)
plt.ylabel('DA-AveUSD (ms)',fontsize=16)

# Adding a legend to distinguish between CEAO and EXA data
plt.legend()


# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')  # Adjusted for the current environment
filename = "convergence.svg"  # Saving as SVG format

# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the figure in the specified directory with a high resolution
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')  # Save as SVG with high resolution

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)


# Showing the plot
plt.show()

##################################################################去掉中间PA更新的数据################
import matplotlib.pyplot as plt
import os
import numpy as np
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
    marker = markers[i % len(markers)]
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
    color_exa = colors[(i + 2) % len(colors)]  # Selecting a distinct color for EXA data
    marker_exa = markers[(i + 2) % len(markers)]  # Selecting a distinct marker for EXA data
    plt.plot(
        np.arange(1, len(avg_utility_per_user_ceao_reduced) + 1),
        np.repeat(avg_utility_per_user_exa, len(avg_utility_per_user_ceao_reduced)),
        label=f'ESA U={u_value}',
        color=color_exa,
        marker=marker_exa,
        linestyle='--',
        fillstyle='none'
    )
    
    
    # Plotting the average utility per user against the iteration number
    plt.plot(
        x_axis_reduced,
        avg_utility_per_user_ceao_reduced,
        label=f'CEAO U={u_value}',
        color=color,
        marker=marker,
        linestyle='-',
        fillstyle='none'
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
plt.xlabel('Iteration Number')
plt.ylabel('DA-AveUSD')

# Adding a legend to distinguish between CEAO and EXA data
# 调整图例位置，稍微向上移动
plt.legend(bbox_to_anchor=(0.71, 0.56), loc='lower center', borderaxespad=0.)


# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
filename = "convergence.svg"  # Saving as SVG format

# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the figure in the specified directory
plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)

# Showing the plot
plt.show()


#############################################################去掉中间PA更新的数据 且 加圈和箭头########################
from matplotlib.patches import Ellipse

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

    # EJO data
    params = params_template.copy()
    params['u_value'] = u_value  # Update the u_value for each scenario
    params['output_dir'] = 'ejo_output'
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
plt.xlabel('Iteration Number')
plt.ylabel('DA-AveUSD')

# Adding a legend to distinguish between CEAO and EXA data
plt.legend(bbox_to_anchor=(0.78, 0.06), loc='lower center', borderaxespad=0.)

# Set the directory where you want to save the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
# filename = "convergence_annotate.svg"  # Saving as SVG format
filename = "convergence_annotate.eps"  # 以EPS格式保存
# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the figure in the specified directory
# plt.savefig(os.path.join(save_dir, filename), dpi=500, bbox_inches='tight')

# Return the path of the saved figure
file_path = os.path.join(save_dir, filename)
print(file_path)

# Showing the plot
plt.show()
