#!/bin/bash
# M=7
# N=5
# # Random Deployment -U
# for U in {14..70..14}
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_time"
#   echo "Random Deployment with U=${U}"
#   python ./random_deployment.py --json-file "${json_file}" &
# done
# wait
# ###############################################train_mappo-user=42,different U######################################################
# Execution ags - users
mappo_output_dir="mappo_output"
ceao_output_dir="ceao_output"
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
# # 循环U值
# for U in 14 28 42 56 70; do
#   json_file_trained="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_time"
#   echo "Running with U=${U}"
#   # Policy Proposed with NOMA and drop
#   python ./main_joint_solve_mappo_test.py --json-file-trained "${json_file_trained}" --json-file "${json_file}" --output_dir "${mappo_output_dir}" --gamma "${gamma}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NOMA --drop
#   wait
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop
#   wait
# done


# 循环NIND值
for NIND in 50 100 200 300 400; do
  json_file_trained="train_M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
  json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_time"
  echo "Running with U=${U}"
  # Policy Proposed with NOMA and drop
  python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop
  wait
done