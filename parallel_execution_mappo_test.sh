#!/bin/bash


# # ###############################################train_mappo-user=42,different gamma######################################################
# # Execution ags - users
# mappo_output_dir="mappo_output"
# M="7"
# U="42"
# N="5"
# T_p="0.01"
# z_n="2"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"

# # 循环error值
# for gamma in 0 0.3 0.5 0.7 0.99; do
#   json_file_trained="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   echo "Running with gamma=${gamma}"
#   # Policy Proposed with NOMA and drop
#   python ./main_joint_solve_mappo_test.py --json-file-trained "${json_file_trained}" --json-file "${json_file}" --output_dir "${mappo_output_dir}" --gamma "${gamma}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NOMA --drop &
#   wait
# done


# ###############################################train_mappo-user=42,different U######################################################
# Execution ags - users
mappo_output_dir="mappo_output"
M="7"
U="42"
N="5"
T_p="0.01"
z_n="2"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
gamma="0.1"
# 循环error值
for U in 70; do
  json_file_trained="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
  json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
  echo "Running with U=${U}"
  # Policy Proposed with NOMA and drop
  python ./main_joint_solve_mappo_test.py --json-file-trained "${json_file_trained}" --json-file "${json_file}" --output_dir "${mappo_output_dir}" --gamma "${gamma}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NOMA --drop &
  wait
done