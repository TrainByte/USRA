#!/bin/bash 2025/5/18
# 排队的脚本程序，不同的queue_number

# 首先把random deployment 部署完毕,
# 无需部署，采用基准的环境即可M7U42K6...等

# 实验一下均匀分布/单小区用户很多的情况，所以再部署下random deployment
M="7"
U="42"
N="5"
data_size="15000"
for U in {42..14..42}
do
  json_file="M${M}_U${U}_N${N}_Task_u${data_size}_Ucpu_e1000_Tdemand_u15_queue_uniform"
  json_file1="M3_U30_N${N}_Task_u${data_size}_Ucpu_e1000_Tdemand_u15_queue"
  json_file2="M3_U30_N${N}_Task_u${data_size}_Ucpu_e1000_Tdemand_u15_queue_uniform"
  echo "Random Deployment with U=${U}"
  python ./random_deployment.py --json-file "${json_file}" &
  python ./random_deployment.py --json-file "${json_file1}" &
  python ./random_deployment.py --json-file "${json_file2}" &
done
wait
sleep 15
###############################################MEC_cpu_cycles######################################################
# Execution ags - MEC_cpu_cycles
ceao_output_dir="ceao_output"
M="7"
U="42"
N="5"
T_p="0.01"
z_n="2"
z_n_oma="1"
P_max="23"
B="2e7"
NIND="500"
data_size="15000"
json_file="M${M}_U${U}_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_queue_uniform"
json_file1="M3_U30_N${N}_Task_u15000_Ucpu_e1000_Tdemand_u15_queue"
json_file2="M3_U30_N${N}_Task_u${data_size}_Ucpu_e1000_Tdemand_u15_queue_uniform"
# 生成cpu_cycles值
cpu_cycles_values=()
for cpu_multiplier in {1..5}
do
    cpu_cycles=$(python -c "print(${cpu_multiplier}*0.5e10)")
    cpu_cycles_values+=("$cpu_cycles")
done

# 手动指定queue_number值
queue_numbers=(1 6 3)


# 定义要执行的命令模板
commands=()
for queue_number in "${queue_numbers[@]}"
do
  for cpu_cycles in "${cpu_cycles_values[@]}"
  do
    echo "Running with cpu_cycles=${cpu_cycles}, queue_number=${queue_number}"
    # Policy Proposed with NOMA and drop
    commands+=("python ./main_joint_solve_queue.py --json-file \"${json_file}\" --queue_number \"${queue_number}\" --cpu_cycles \"${cpu_cycles}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA --drop")
    commands+=("python ./main_joint_solve_queue.py --json-file \"${json_file1}\" --queue_number \"${queue_number}\" --cpu_cycles \"${cpu_cycles}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA --drop")
    commands+=("python ./main_joint_solve_queue.py --json-file \"${json_file2}\" --queue_number \"${queue_number}\" --cpu_cycles \"${cpu_cycles}\" --output_dir \"${ceao_output_dir}\" --B \"${B}\" --P_max \"${P_max}\" --T_p \"${T_p}\" --z_n \"${z_n}\" --NIND \"${NIND}\" --NOMA --drop")
  done
done

# 使用xargs控制并发进程数量
printf "%s\n" "${commands[@]}" | xargs -P 18 -I {} bash -c "{}"