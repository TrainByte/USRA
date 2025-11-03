#!/bin/bash
# 首先把random deployment 全都部署完毕
# Random Deployment -M
# for M in 4 8
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   echo "Random Deployment with U=${M}"
#   python ./random_deployment.py --json-file "${json_file}" &
# done
# wait

# M=7
# N=5
# # Random Deployment -U
# for U in {14..70..14}
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   echo "Random Deployment with U=${U}"
#   python ./random_deployment.py --json-file "${json_file}" &
# done
# wait

# # Random Deployment and Execution ags -N
# U=42
# M=7
# exclude_N="5"
# # 循环N值
# for N in {4..12..2}
# do
#   # 跳过不需要执行的N值
#   if [[ ! $exclude_N =~ $N ]]; then
#     json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#     echo "Random Deployment with N=${N}"
#     # Random Deployment
#     python ./random_deployment.py --json-file "${json_file}" &
#   fi
# done
# wait

######################################################不同beta############################################################
ceao_output_dir="ceao_output"
# fura_output_dir="fura_output"
# lura_output_dir="lura_output"
# baselines_output_dir="baselines_output"
# ejo_output_dir="ejo_output"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
T_p="0.01"
z_n="2"
M=3
U=42
N=5
# execution ags -beta "0" "1" "2"
for beta in "0.5" "1.5" "2.5" "0.25" "0.75"
do
  json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_fix"
  echo "Running with beta=${beta}"

  # Policy Proposed with NOMA and drop
  python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --beta "${beta}" --NOMA --drop &
  
  # sleep 1m
  wait
done




