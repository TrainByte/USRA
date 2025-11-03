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
# ######################################################不同B############################################################
# ceao_output_dir="ceao_output"
# fura_output_dir="fura_output"
# lura_output_dir="lura_output"
# baselines_output_dir="baselines_output"
# ejo_output_dir="ejo_output"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# T_p="0.01"
# z_n="2"
# z_n_oma="1"
# M=7
# N=5
# #N = "8"
# # 循环B值
# for B in "10e7" "8e7" "6e7" "4e7"
# do
#   echo "Running simulations with B=${B}"
#   # 循环U值
#   for U in {42..42..14}
#   do
#     json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
#     echo "Running with U=${U} and Bandwidth=${B}"

#     # Policy Proposed with NOMA and drop
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Policy Proposed with NOMA only
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#     # Policy Proposed with drop only Without SIC
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

#     # Policy Proposed with drop only OMA
#     # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n_oma}" --NIND "${NIND}" --drop &

#     # main_joint_solve_ga_sp with NOMA and drop
#     # python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Policy Proposed with NOMA only, user selection是先到先服务
#     # python ./main_fura_solve.py --json-file "${json_file}" --output_dir "${fura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#     # Policy Proposed with NOMA only, user selection是先截至的先服务
#     # python ./main_lura_solve.py --json-file "${json_file}" --output_dir "${lura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#     # sleep 1m
#     # wait
#   done
# done

# ######################################################不同U############################################################
ceao_output_dir="ceao_output"
fura_output_dir="fura_output"
lura_output_dir="lura_output"
baselines_output_dir="baselines_output"
ejo_output_dir="ejo_output"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
T_p="0.01"
z_n="2"
z_n_oma="1"
M=7
N=5
#N = "8"
# 循环B值
for B in "2e7"
do
  echo "Running simulations with B=${B}"
  # 循环U值
  for U in {14..70..14}
  do
    json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
    echo "Running with U=${U} and Bandwidth=${B}"

    # Policy Proposed with NOMA and drop
    python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

    # # Policy Proposed with NOMA only
    # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

    # # Policy Proposed with drop only
    # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &
    
    # # Policy Proposed with drop only
    # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n_oma}" --NIND "${NIND}" --drop &

    # # main_joint_solve_ga_sp with NOMA and drop
    # python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

    # # Policy Proposed with NOMA only, user selection是先到先服务
    # python ./main_fura_solve.py --json-file "${json_file}" --output_dir "${fura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

    # # Policy Proposed with NOMA only, user selection是先截至的先服务
    # python ./main_lura_solve.py --json-file "${json_file}" --output_dir "${lura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

    # sleep 1m
    # wait
  done
done

######################################################不同beta############################################################
# ceao_output_dir="ceao_output"
# # fura_output_dir="fura_output"
# # lura_output_dir="lura_output"
# # baselines_output_dir="baselines_output"
# # ejo_output_dir="ejo_output"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# T_p="0.01"
# z_n="2"
# M=3
# U=42
# N=5
# # execution ags -M
# for beta in "0" "1" "2"
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_fix"
#   echo "Running with beta=${beta}"

#   # Policy Proposed with NOMA and drop
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --beta "${beta}" --NOMA --drop &
  
#   sleep 1m
# done



