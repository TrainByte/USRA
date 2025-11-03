import json
import os

# Define the function to generate JSON with varying task size
def generate_json_varying_task_size(m_value, u_value, n_value, avg_size, avg_size_cpu, output_dir):
    min_val = avg_size - 5000  # Set the min_val based on the avg_size
    max_val = avg_size + 5000  # Set the max_val based on the avg_size
    
    data = {
        "simulation": {
            "num_simulations": 1,
            "simulation_index_start": 0,
            "M": m_value,
            "U": u_value,
            "N": n_value,
            "R_defined": 400,
            "min_dist": 35,
            "dcor": 10.0,
            "shadowing_dev": 10.0,
            "T": 0.01,
            "total_samples": 100,
            "uplink_sinr": 2.0,
            "train_sequence_uplink": 20,
            "noise": -174,
            "isTrain": False,
            "equal_number_for_BS": True,
        },
        "train_episodes": {
            "T_train": 1,
            "T_sleep": 0,
            "cell_passing_training": True,
            "cell_passing_sleeping": True,
            "T_register": 50
        },
        "mobility_params": {
            "v_c": 300000000.0,
            "f_c": 2000000000.0,
            "v_max": 0.0,
            "a_max": 0.5,
            "alpha_angle_rad": 0.175,
            "T_mobility": 50,
            "max_doppler": "independent"
        },
        "compute_params": {
            "task": {
                "distribution": "uniform",
                "min_val": min_val,
                "max_val": max_val,
                "avg_size": avg_size
            },
            "cpu": {
                "distribution": "exponential",
                "min_val": 500,
                "max_val": 1500,
                "avg_size": avg_size_cpu
            },
            "T_requirement": {
                "distribution": "uniform",
                "min_val": 0.01,
                "max_val": 0.02,
                "avg_latency": 0.015
            },
        }
    }

    filename = f"M{m_value}_U{u_value}_N{n_value}_Task_u{avg_size}_Ucpu_e{avg_size_cpu}_Tdemand_u15.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

# 创建 "config/deployment" 文件夹
output_directory = os.path.join(os.getcwd(), "config", "deployment")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

m_value = 7
u_value = 42
n_value = 5
avg_size_cpu = 1000
avg_sizes = [10000, 15000, 20000, 25000, 30000]

for avg_size in avg_sizes:
    generate_json_varying_task_size(m_value, u_value, n_value, avg_size, avg_size_cpu, output_directory)

"JSON files with varying task sizes have been generated and saved to the specified folder."
avg_size_cpus = [500, 1000, 1500, 2000, 2500]
avg_size = 15000
for avg_size_cpu in avg_size_cpus:
    generate_json_varying_task_size(m_value, u_value, n_value, avg_size, avg_size_cpu, output_directory)