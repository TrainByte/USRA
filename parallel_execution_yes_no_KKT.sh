#!/bin/bash
M=7
N=5
# Random Deployment -U
for U in {42..42..14}
do
  json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_YNKKT"
  echo "Random Deployment with U=${U}"
  python ./random_deployment.py --json-file "${json_file}" &
done
wait
# ###############################################train_mappo-user=42,different U######################################################
# Execution ags - users
ceao_withoutKKT_output_dir="ceao_NoKKT_output"
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
# 循环U值
# for U in 14 28 42 56 70; do
for U in 42; do
  json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_YNKKT"
  echo "Running with U=${U}"
  # Policy without KKT
  python ./main_joint_solve_NoKKT.py --json-file "${json_file}" --output_dir "${ceao_withoutKKT_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
  # Policy with KKT
  python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop

done
