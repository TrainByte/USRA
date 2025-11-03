#!/bin/bash

ceao_output_dir="ceao_output"
baselines_output_dir="baselines_output"
ejo_output_dir="ejo_output"
esa_output_dir="exhaustive_search_output"
P_max="23"
B="2e7"
NIND="500"
cpu_cycles="1.5e10"
T_p="0.01"
z_n="2"

# # Random Deployment -U
# for U in {9..15..3}
# do
#   json_file="M3_U${U}_N2_Task_u15000_Ucpu_e1000_Tdemand_u15_slot1"
#   echo "Random Deployment with U=${U}"
#   python ./random_deployment.py --json-file "${json_file}" &
# done
# wait

# execution ags -U
for U in {9..15..3}
do
  json_file="M3_U${U}_N2_Task_u15000_Ucpu_e1000_Tdemand_u15_slot1"
  echo "Running with U=${U}"

  # Policy Proposed with NOMA and drop
  # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

  # # Policy Proposed with NOMA only
  # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA &

  # # Policy Proposed with drop only
  # python ./main_joint_solve.py --json-file "${json_file}" --output_dir "${ceao_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --drop &

  # # main_joint_solve_ga_sp with NOMA and drop
  python ./main_joint_solve_ga_sp.py --json-file "${json_file}" --output_dir "${ejo_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &

  # # Baselines AG with NOMA and drop
  # python ./baselines_ag.py --json-file "${json_file}" --output_dir "${baselines_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
  wait
done
wait

# execution ags -U
for U in {9..15..3}
do
  json_file="M3_U${U}_N2_Task_u15000_Ucpu_e1000_Tdemand_u15_slot1"
  # 并行执行
  # python ./main_parallel_exhaustive_search.py --json-file "${json_file}" --output_dir "${esa_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
  # 串行执行
  #python ./main_exhaustive_search.py --json-file "${json_file}" --output_dir "${esa_output_dir}" --B "${B}" --P_max "${P_max}" --cpu_cycles "${cpu_cycles}" --T_p "${T_p}" --z_n "${z_n}" --NIND "${NIND}" --NOMA --drop &
done
wait

