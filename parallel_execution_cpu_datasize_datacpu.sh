#!/bin/bash

# ceao_output_dir="ceao_output"
# baselines_output_dir="baselines_output"
# ejo_output_dir="ejo_output"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# T_p="0.01"
# z_n="2"
# M="7"
# U="42"
# N="5"
# # 首先把random deployment 全都部署完毕
# # Random Deployment -avge_data_size
# exclude_avge_data_size="15000"
# for avge_data_size in {10000..30000..5000}
# do
#   # 跳过不需要执行的N值
#   if [[ ! $exclude_avge_data_size =~ $avge_data_size ]]; then
#     json_file="M${M}_U${U}_N${N}_Task_u${avge_data_size}_Ucpu_e1000_Tdemand_u15"
#     echo "Random Deployment with U=${U}"
#     python ./random_deployment.py --json-file "${json_file}" &
#   fi
# done
# wait

# # Random Deployment and Execution ags -N
# exclude_data_cpu="1000"
# # 循环N值
# for data_cpu in {500..2500..500}
# do
#   # 跳过不需要执行的N值
#   if [[ ! $exclude_data_cpu =~ $data_cpu ]]; then
#     json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e${data_cpu}_Tdemand_u15"
#     echo "Random Deployment with N=${N}"
#     # Random Deployment
#     python ./random_deployment.py --json-file "${json_file}" &
#   fi
# done
# wait

###############################################MEC_cpu_cycles######################################################
# Execution ags - MEC_cpu_cycles
ceao_output_dir="ceao_output"
fura_output_dir="fura_output"
lura_output_dir="lura_output"
ejo_output_dir="ejo_output"
M="7"
U="42"
N="5"
T_p="0.01"
z_n="2"
z_n_oma="1"
P_max="23"
B="2e7"
NIND="500"
exclude_cpu_cycles=$(python -c "print(1.5e10)") # 将排除值转换为浮点数
json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15"
# 循环cpu_cycles值
for cpu_multiplier in {1..5}
do
  cpu_cycles=$(python -c "print(${cpu_multiplier}*0.5e10)") # 使用Python进行计算
  if [[ $cpu_cycles != $exclude_cpu_cycles ]]; then

    echo "Running with cpu_cycles=${cpu_cycles}"
    # Policy Proposed with NOMA and drop
    python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

    # Policy Proposed with NOMA only
    python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

    # Policy Proposed with drop only -Without SIC
    python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

    # Policy Proposed with drop only -OMA
    python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n_oma}" --NIND "${NIND}" --drop &

    # main_joint_solve_ga_sp with NOMA and drop
    python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

    # Policy Proposed with NOMA only lura
    python ./main_lura_solve.py --json-file "${json_file}" --output_dir "${lura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

    # Policy Proposed with NOMA only fura
    python ./main_fura_solve.py --json-file "${json_file}" --output_dir "${fura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

  wait
  fi
done

# ###############################################data_size######################################################
# # Execution ags -data_size
# M="7"
# U="42"
# N="5"
# z_n="2"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# exclude_avge_data_size=$(python -c "print(15000)") # 将排除值转换为浮点数
# # 循环T_p值
# for avge_data_size in {10000..30000..5000}
# do
#   json_file="M${M}_U${U}_N${N}_Task_u${avge_data_size}_Ucpu_e1000_Tdemand_u15"
#   if [[ $avge_data_size != $exclude_avge_data_size ]]; then
#     echo "Running with avge_data_size=${avge_data_size}"
#     # Policy Proposed with NOMA and drop
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Policy Proposed with NOMA only
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#     # Policy Proposed with drop only
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

#     # main_joint_solve_ga_sp with NOMA and drop
#     python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Baselines AG with NOMA and drop
#     python ./baselines_ag.py --json-file "${json_file}" --output_dir "${baselines_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
#   fi
# done
# wait

# ###############################################data_cpu######################################################
# # Execution ags -data_cpu
# M="7"
# U="42"
# N="5"
# z_n="2"
# P_max="23"
# B="2e7"
# NIND="500"
# cpu_cycles="1.5e10"
# exclude_data_cpu=$(python -c "print(1000)") # 将排除值转换为浮点数
# # 循环data_cpu值
# for data_cpu in {500..2500..500}
# do
#   json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e${data_cpu}_Tdemand_u15"
#   if [[ $data_cpu != $exclude_data_cpu ]]; then
#     echo "Running with data_cpu=${data_cpu}"
#     # Policy Proposed with NOMA and drop
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Policy Proposed with NOMA only
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#     # Policy Proposed with drop only
#     python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

#     # main_joint_solve_ga_sp with NOMA and drop
#     python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#     # Baselines AG with NOMA and drop
#     python ./baselines_ag.py --json-file "${json_file}" --output_dir "${baselines_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
#   fi
# done
# wait
