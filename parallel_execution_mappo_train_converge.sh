#!/bin/bash


M="7"
U="14"
N="5"
T_p="0.01"
z_n="2"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
gamma="0.1"
data_cpu="1000"
# 首先把random deployment 全都部署完毕
# Random Deployment and Execution ags -N
# for U in 28 42; do
#   echo "Random Deployment with U=${U}"
#   json_file="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e${data_cpu}_Tdemand_u15_converge"
#   # Random Deployment
#   python ./random_deployment.py --json-file "${json_file}" &
#   json_file1="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e${data_cpu}_Tdemand_u15_converge1"
#   python ./random_deployment_madrl_train.py --json-file "${json_file1}" &
#   wait
# done


# ###############################################train_mappo-users######################################################
# Execution ags - users
# 根据上面选出来gamma=0.1
mappo_output_dir="mappo_output"
madqn_output_dir="madaqn_output1"
M="7"
U="14"
N="5"
T_p="0.01"
z_n="2"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
gamma="0.1"

# 循环error值
for U in 28 42; do
  json_file="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_converge"
  echo "Running with U=${U}"
  # Policy Proposed with NOMA and drop
  # python ./main_joint_solve_mappo_train.py --json-file "${json_file}" --output_dir "${mappo_output_dir}" --gamma "${gamma}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NOMA --drop &
  # python ./main_joint_solve_madqn_train.py --json-file "${json_file}" --output_dir "${madqn_output_dir}" --gamma "${gamma}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NOMA --drop &
  
  # 任务也全相同针对每个episode中的每个time slot下用户位置和任务与下一个episode对应的time slot下用户位置和任务相同,但是每个time slot的任务是不同的.
  json_file1="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_converge1"
  # python ./main_joint_solve_mappo_train.py --json-file "${json_file1}" --output_dir "${mappo_output_dir}" --gamma "${gamma}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NOMA --drop &
  python ./main_joint_solve_madqn_train.py --json-file "${json_file1}" --output_dir "${madqn_output_dir}" --gamma "${gamma}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NOMA --drop &
  
done