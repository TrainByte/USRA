#!/bin/bash

ceao_output_dir="ceao_output"
baselines_output_dir="baselines_output"
ejo_output_dir="ejo_output"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
T_p="0.01"
z_n="2"
M=7
N=5
U=42
# # 首先把random deployment 全都部署完毕
# # Random Deployment -M
# for M in 4 8
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   echo "Random Deployment with U=${M}"
#   python ./random_deployment.py --json-file "${json_file}" &
# done
# wait

# M=6
# # Random Deployment -U
# for U in {12..84..12}
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   echo "Random Deployment with U=${U}"
#   python ./random_deployment.py --json-file "${json_file}" &
# done
# wait

# # Random Deployment and Execution ags -N
# U=48
# M=6
# exclude_N="8"
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
######################################################不同z_n，不同N############################################################
# ceao_output_dir="ceao_output"
# baselines_output_dir="baselines_output"
# ejo_output_dir="ejo_output"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# T_p="0.02"
# z_n="2"
# M=7
# U=42
#N = "8"

# # 循环N值
# for N in {4..12..2}
# do
#   for z_n in 3 4  # z_n的值分别为 3, 4
#   do
#     json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#     echo "Running with N=${N} and z_n=${z_n}"

#     # Policy Proposed with NOMA and drop
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Policy Proposed with NOMA only
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#     # Policy Proposed with drop only
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

#     # main_joint_solve_ga_sp with NOMA and drop
#     # python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Baselines AG with NOMA and drop
#     # python ./baselines_ag.py --json-file "${json_file}" --output_dir "${baselines_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     sleep 1  # 等待
#   done
# done
# #######################################################z_n和U###########################################################
# ceao_output_dir="ceao_output"
# baselines_output_dir="baselines_output"
# ejo_output_dir="ejo_output"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# T_p="0.02"
# z_n="2"
# M=6
# U=48
# N=8
# # 循环N值
# for z_n in 3 4 5 6 # z_n的值分别为 2, 3, 4, 5, 6
# do
#   for U in 24 36 48  # z_n的值分别为 24, 36, 48
#   do
#     json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#     echo "Running with N=${N} and z_n=${z_n}"

#     # Policy Proposed with NOMA and drop
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Policy Proposed with NOMA only
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#     # Policy Proposed with drop only
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

#     # main_joint_solve_ga_sp with NOMA and drop
#     # python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Baselines AG with NOMA and drop
#     # python ./baselines_ag.py --json-file "${json_file}" --output_dir "${baselines_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     sleep 1  # 等待
#   done
# done

#######################################################P_max###########################################################
ceao_output_dir="ceao_output"
baselines_output_dir="baselines_output"
ejo_output_dir="ejo_output"
fura_output_dir="fura_output"
lura_output_dir="lura_output"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
T_p="0.01"
z_n="2"
z_n_oma="1"
M=7
U=42
N=5
# Execution ags -P_max
json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
# 循环P_max值
for P_max in {5..45..10}
do
  echo "Running with P_max=${P_max}"
  # # Policy Proposed with NOMA and drop
  # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

  # # Policy Proposed with NOMA only
  # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

  # # Policy Proposed with drop only without SIC
  # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &
  
  # Policy Proposed with drop only OMA
  python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n_oma}" --NIND "${NIND}" --drop &

  # # main_joint_solve_ga_sp with NOMA and drop
  # python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
 
  # # Policy Proposed with NOMA only
  # python ./main_lura_solve.py --json-file "${json_file}" --output_dir "${lura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &
  # # Policy Proposed with NOMA only
  # python ./main_fura_solve.py --json-file "${json_file}" --output_dir "${fura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

  # Baselines AG with NOMA and drop
  # python ./baselines_ag.py --json-file "${json_file}" --output_dir "${baselines_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
  # wait
done
wait

#######################################################不同U###########################################################
# ceao_output_dir="ceao_output"
# baselines_output_dir="baselines_output"
# ejo_output_dir="ejo_output"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# T_p="0.02"
# z_n="2"
# M=7
# N=5
# # Execution ags -U
# for U in {12..84..12}
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#   echo "Running with N=${N}"
#   # Policy Proposed with NOMA and drop
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#   # Policy Proposed with NOMA only
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#   # Policy Proposed with drop only
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

#   # main_joint_solve_ga_sp with NOMA and drop
#   python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#   # Baselines AG with NOMA and drop
#   python ./baselines_ag.py --json-file "${json_file}" --output_dir "${baselines_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

# done