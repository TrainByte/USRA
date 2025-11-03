#!/bin/bash

ceao_output_dir="ceao_output"
# baselines_output_dir="baselines_output"
# ejo_output_dir="ejo_output"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
T_p="0.01"
z_n="2"
M="3"
U="15"
N="5"
data_cpu="1000"
# 首先把random deployment 全都部署完毕
# Random Deployment and Execution ags -N
for U in 24; do
    # 循环error值
    for error in 0 01 02 03; do
    json_file_homo="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e${data_cpu}_Tdemand_u15_error${error}_homo"
    json_file_hetero="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e${data_cpu}_Tdemand_u15_error${error}_hetero"
    echo "Random Deployment with error=${error}"
    # Random Deployment
    python ./random_deployment.py --json-file "${json_file_homo}" &
    python ./random_deployment.py --json-file "${json_file_hetero}"
    done
    wait
done
# ###############################################errors######################################################
# Execution ags - errors

ceao_output_dir="ceao_error_homohetero_output"
M="3"
U="15"
N="5"
T_p="0.01"
z_n="2"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"

# 生成所有需要执行的命令
commands=()
# 循环error值
for U in 24; do
    for error in 0 01 02 03; do
    json_file_homo="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_error${error}_homo"
    json_file_hetero="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_error${error}_hetero"
    commands+=("python ./main_joint_solve_error_homohetero.py --json-file \"${json_file_homo}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --cpu_cycles \"${cpu_cycles}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA --drop")
    commands+=("python ./main_joint_solve_error_homohetero.py --json-file \"${json_file_hetero}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --cpu_cycles \"${cpu_cycles}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA --drop")
    # echo "Running with error=${error} homa"
    # # Policy Proposed with NOMA and drop
    # python ./main_joint_solve_error_homohetero.py --json-file "${json_file_homo}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop
    
    # echo "Running with error=${error} hetero"
    # # Policy Proposed with NOMA and drop
    # python ./main_joint_solve_error_homohetero.py --json-file "${json_file_hetero}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

    done
done

# 使用 xargs 和 parallel 并行执行命令，最多同时执行 4 个任务
printf "%s\n" "${commands[@]}" | xargs -P 4 -I {} sh -c "echo 'Running: {}'; {}"