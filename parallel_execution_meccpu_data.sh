#!/bin/bash

P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
T_p="0.01"
z_n="2"
M="7"
U="42"
N="5"
data_size="15000"
# 首先把random deployment 全都部署完毕

for U in {42..14..42}
do
  json_file="M${M}_U${U}_N${N}_Task_u${data_size}_Ucpu_e1000_Tdemand_u15_rate"
  echo "Random Deployment with U=${U}"
  python ./random_deployment.py --json-file "${json_file}" &
done
wait

###############################################MEC_cpu_cycles######################################################
# Execution ags - MEC_cpu_cycles
ceao_output_dir="ceao_output"
fura_output_dir="fura_output"
lura_output_dir="lura_output"
# ejo_output_dir="ejo_output"
M="7"
U="42"
N="5"
T_p="0.01"
z_n="2"
z_n_oma="1"
P_max="23"
B="2e7"
NIND="500"
# exclude_cpu_cycles=$(python -c "print(1.5e10)") # 将排除值转换为浮点数
json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_rate"

# 生成cpu_cycles值
cpu_cycles_values=()
for cpu_multiplier in {2..14}
do
    cpu_cycles=$(python -c "print(${cpu_multiplier}*0.25e10)")
    cpu_cycles_values+=("$cpu_cycles")
done

# 定义要执行的命令模板
commands=()
for cpu_cycles in "${cpu_cycles_values[@]}"
do
    echo "Running with cpu_cycles=${cpu_cycles}"
    # # Policy Proposed with NOMA and drop
    # commands+=("python ./main_joint_solve.py --json-file \"${json_file}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --cpu_cycles \"${cpu_cycles}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA --drop")
    # Policy Proposed with NOMA only
    commands+=("python ./main_joint_solve.py --json-file \"${json_file}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --cpu_cycles \"${cpu_cycles}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA")
    # # Policy Proposed with drop only -OMA
    # commands+=("python ./main_joint_solve.py --json-file \"${json_file}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --cpu_cycles \"${cpu_cycles}\" --T_p \"${T_p}\" --z_n \"${z_n_oma}\" --NIND \"${NIND}\" --drop")
    # # Policy Proposed with NOMA only lura
    # commands+=("python ./main_lura_solve.py --json-file \"${json_file}\" --output_dir \"${lura_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --cpu_cycles \"${cpu_cycles}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA")
    # # Policy Proposed with NOMA only fura
    # commands+=("python ./main_fura_solve.py --json-file \"${json_file}\" --output_dir \"${fura_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --cpu_cycles \"${cpu_cycles}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA")

done

# 使用xargs控制并发进程数量
printf "%s\n" "${commands[@]}" | xargs -P 18 -I {} bash -c "{}"

# # 循环cpu_cycles值
# for cpu_multiplier in {1..5}
# do
#   cpu_cycles=$(python -c "print(${cpu_multiplier}*0.5e10)") # 使用Python进行计算

#   echo "Running with cpu_cycles=${cpu_cycles}"
#   # Policy Proposed with NOMA and drop
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#   # Policy Proposed with NOMA only
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#   # # Policy Proposed with drop only -Without SIC
#   # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

#   # Policy Proposed with drop only -OMA
#   python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n_oma}" --NIND "${NIND}" --drop &

#   # # main_joint_solve_ga_sp with NOMA and drop
#   # python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

#   # Policy Proposed with NOMA only lura
#   python ./main_lura_solve.py --json-file "${json_file}" --output_dir "${lura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

#   # Policy Proposed with NOMA only fura
#   python ./main_fura_solve.py --json-file "${json_file}" --output_dir "${fura_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

# wait
# done
