import json
import os


def generate_json(m_value, u_value, n_value, output_dir):
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
                "min_val": 10000,
                "max_val": 20000,
                "avg_size": 15000
            },
            "cpu": {
                "distribution": "exponential",
                "min_val": 500,
                "max_val": 1500,
                "avg_size": 1000
            },
            "T_requirement": {
                "distribution": "uniform",
                "min_val": 0.01,
                "max_val": 0.02,
                "avg_latency": 0.015
            },
        }
    }

    filename = f"M{m_value}_U{u_value}_N{n_value}_Task_u15000_Ucpu_e1000_Tdemand_u15.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


# 创建 "config/deployment" 文件夹
output_directory = os.path.join(os.getcwd(), "config", "deployment")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

######################################################################################################
m_value = 7
u_values = [14, 28, 42, 56, 70]
n_value = 5
for i, u_value in enumerate(u_values):
    generate_json(m_value, u_value, n_value, output_directory)

print("JSON files generated and saved to the 'config/deployment' folder.")
# ######################################################################################################
# u_value = 42
# n_values = [4, 6, 8, 10, 12]
# for i, n_value in enumerate(n_values):
#     generate_json(m_value, u_value, n_value, output_directory)

# print("JSON files generated and saved to the 'config/deployment' folder.")

# ######################################################################################################
# n_value = 5
# u_value = 42
# m_values = [4, 8]
# for i, m_value in enumerate(m_values):
#     generate_json(m_value, u_value, n_value, output_directory)

# print("JSON files generated and saved to the 'config/deployment' folder.")
