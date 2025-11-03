"""
@author: Ning zhe Shi
"""
import cvxpy as cp
import numpy as np
import project_backend as pb
import time

def power_allocation_cvx(max_iter, p_temp, user_channel_temp, H_gain_data, u, n, m, cell_mapping, P_max, epsilon,
                         sum_utility_old, T_m_k_c_matrix, T_requirement_matrix, T_p, task_matrix, W, NOMA, dropped,weight):
    task_matrix_cp = task_matrix.reshape(-1, 1)
    # 处理总时间约束
    t_constraint = ((T_requirement_matrix - T_m_k_c_matrix) * W / task_matrix).reshape(-1, 1)
    # sr_constraint = se_r_matrix.reshape(-1, 1)
    # sr_c = np.mean(sr_constraint) * u

    H_large_t = np.zeros((u, u))
    for i in range(u):
        H_large_t[i, :] = H_gain_data[i, :,
                          user_channel_temp[i] - 1]  # 取出第i个用户在对应信道上与其余所有用户所在小区基站的信道增益矩阵,如果被丢弃即信道号为0则随机取即可，不影响，所以这里直接取-1
    H_gain = H_large_t.T  # 转置对应上求解方式中的信道增益矩阵

    alp_temp = pb.user_channel_mapping_matrix(user_channel_temp, u, n)
    # 选择信道为0的不对别人产生干扰，自身也不受干扰
    alp_temp[:, 0] = 0
    # 计算干扰矩阵
    OMA_temp = np.multiply(alp_temp @ alp_temp.T, 1 - np.eye(u))
    OMA_temp[OMA_temp >= 1] = 1

    if NOMA is True:
        A_temp = pb.NOMA_handle(user_channel_temp, OMA_temp, m, cell_mapping, H_gain, p_temp)
    else:
        A_temp = OMA_temp

    iter_power = 0
    power_allocation_index = np.zeros(max_iter)  # 存储大循环中第iter轮有解的次数
    power_allocation_index_infeasible = np.zeros(max_iter)  # 存储大循环中的第iter轮无解的次数
    power_no_sloved = 0  # 指示器，指示整个大循环迭代过程中是否有功率优化
    
    power_time_iter = 0  # 用于记录功率分配多次迭代求解的总求解时间
    while iter_power < max_iter:
        iter_power += 1
        #print(f"#######################power allocation第{iter_power}轮####################")

        # Step 1
        y_star = np.sqrt(np.diag(H_gain).reshape(-1, 1) * p_temp) / (H_gain * A_temp @ p_temp + 1)  # 公式（33）
        kk = np.diag(H_gain).reshape(-1, 1)
        v_star = 1 / np.sqrt(2 * np.multiply(np.log2(
            1 + 2 * np.multiply(y_star, np.sqrt(np.multiply(kk, p_temp))) - np.multiply(
                np.square(y_star),
                (H_gain * A_temp @ p_temp + 1))), W))
        # Step 2
        p = cp.Variable((u, 1))
        # 复杂的表达式
        complex_expression = cp.multiply(task_matrix_cp,
                                         (np.square(v_star) + cp.square(cp.inv_pos(cp.multiply(cp.multiply(cp.log(
                                             1 + 2 * cp.multiply(y_star, cp.sqrt(cp.multiply(kk, p))) - cp.multiply(
                                                 cp.square(y_star),
                                                 (H_gain * A_temp @ p + 1))) / np.log(2), W),
                                                                                               np.multiply(2,
                                                                                                           v_star))))))

        # 创建包含条件逻辑的表达式数组
        user_latency_expressions = [
            cp.multiply(user_channel_temp[i] == 0, weight.flatten()[i]*T_p) + cp.multiply(user_channel_temp[i] != 0,
                                                                      complex_expression[i])
            for i in range(user_channel_temp.shape[0])
        ]

        # 将表达式数组垂直堆叠成一个向量
        user_latency = cp.vstack(user_latency_expressions)

        # for i in range(user_channel_temp.shape[0]):
        #     if user_channel_temp[i]==0:
        #         user_latency[i] = T_max
        #     else:
        #         user_latency[i] = cp.multiply(task_matrix_cp, (cp.square(cp.inv_pos((cp.log(
        #     1 + 2 * cp.multiply(y_star, cp.sqrt(cp.multiply(kk, p))) - cp.multiply(cp.square(y_star), (
        #             H_gain * A_temp @ p + 1))))))))
        # product = 2 * cp.multiply(y_star, cp.sqrt(cp.multiply(kk, p))) - cp.multiply(cp.square(y_star), (
        #         H_gain * A_temp @ p + 1))
        # z = cp.Variable(u, boolean=True)  # 对每个用户引入一个辅助变量，这里假设是布尔变量

        # 原始的目标函数，不包括违反约束的惩罚
        # original_objective = cp.sum(complex_expression)  # 假设complex_expression已经定义

        objective = cp.Minimize(cp.sum(user_latency) )  # 公式（32）
        # objective = cp.Minimize(cp.sum(user_latency) + T_p * cp.sum(z))  # 公式（32）
        # constraints = [p >= 0, p <= P_max, (cp.inv_pos(cp.log(
        #     1 + 2 * cp.multiply(y_star, cp.sqrt(cp.multiply(kk, p))) - cp.multiply(cp.square(y_star), (
        #             H_gain * A_temp @ p + 1)))/ np.log(2))) <= t_constraint]
        constraints = [p >= 0, p <= P_max]
        # for i in range(u):
        #     # 如果用户未被丢弃，则添加时延约束
        #     if user_channel_temp[i] != 0:
        #         constraints.append(
        #             cp.inv_pos(cp.log(
        #                 1 + 2 * cp.multiply(y_star[i], cp.sqrt(cp.multiply(kk[i], p[i])))
        #                 - cp.multiply(cp.square(y_star[i]), (H_gain * A_temp @ p + 1)[i])
        #             ) / np.log(2)) <= t_constraint[i]
        #         )
        for i in range(u):
            if user_channel_temp[i] != 0:
                # 添加修改后的约束，包括辅助变量z[i]
                constraint_expr = cp.inv_pos(cp.log(
                    1 + 2 * cp.multiply(y_star[i], cp.sqrt(cp.multiply(kk[i], p[i])))
                    - cp.multiply(cp.square(y_star[i]), (H_gain * A_temp @ p + 1)[i])
                ) / np.log(2)) - t_constraint[i]

                # 使用辅助变量z[i]来指示是否违反约束
                constraints.append(constraint_expr <= 0)
                # constraints.append(z[i] >= cp.pos(constraint_expr))  # 如果违反约束，则z[i]应为1，否则为0
        # objective = cp.Minimize((cp.quad_over_lin(cp.inv_pos(cp.log1p((cp.multiply(2*y_star, cp.sqrt(cp.multiply(kk,p))) - cp.multiply(cp.square(y_star), (H_gain*A_temp @ p + 1))))),1)))
        # constraints = [product >= 1e-6, p >= 0, p <= P_max]
        # , cp.sum((cp.log(
        #     1 + 2 * cp.multiply(y_star, cp.sqrt(cp.multiply(kk, p))) - cp.multiply(cp.square(y_star), (
        #             H_gain * A_temp @ p + 1))))) >= sr_c]
        # , cp.log(
        #     1 + 2 * cp.multiply(y_star, cp.sqrt(cp.multiply(kk, p))) - cp.multiply(cp.square(y_star), (
        #             H_gain * A_temp @ p + 1))) >= sr_constraint]
        
        problem = cp.Problem(objective, constraints)

        #print("开始求解")
        start_time = time.time()
        try:
            problem.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME': 60.0})# 设置最大求解时间为60秒
            #print("MOSEK求解完成")
        except cp.SolverError as e:
            #print("求解器错误:", str(e))
            pass
        end_time = time.time()
        power_time_iter += end_time - start_time

        if problem.status == 'infeasible' or problem.value == None or np.any(p.value == None):
            power_allocation_index_infeasible[iter_power - 1] += 1
            #print('功率无解退出')
            break
        else:
            power_allocation_index[iter_power - 1] += 1
            power_no_sloved = 1  # 如果整个大循环power_allocation都无解则该值一直为0，在大循环结束后进行提示
        # TODO MOSEK不可解，分析原因，并非infeasible. 换成ECOS求解器可以求解
        if not np.any(p.value == None):
            # 根据输出功率计算时延
            user_uplink_time,_ = pb.user_uplink_time(H_gain_data, user_channel_temp, p.value, task_matrix, u,
                                    W, m, cell_mapping, T_p, NOMA, dropped,weight)
            user_uplink_plus_computing_time = user_uplink_time.flatten() + T_m_k_c_matrix
            sum_utility_new = np.sum(pb.penalize_service_failures_and_drops(user_uplink_plus_computing_time, T_requirement_matrix, user_channel_temp,
                                                       T_p,weight))
        else:
            user_uplink_time,_ = pb.user_uplink_time(H_gain_data, user_channel_temp, p_temp, task_matrix, u,W, m, cell_mapping, T_p, NOMA, dropped,weight)
            user_uplink_plus_computing_time = user_uplink_time.flatten() + T_m_k_c_matrix
            sum_utility_new = np.sum(pb.penalize_service_failures_and_drops(user_uplink_plus_computing_time, T_requirement_matrix, user_channel_temp,
                                                       T_p,weight))
        if abs(sum_utility_new - sum_utility_old) / sum_utility_old < epsilon or sum_utility_new < sum_utility_old:
            #print('结束：', user_uplink_plus_computing_time)
            break
        else:
            #print('未结束', user_uplink_plus_computing_time)
            #print("效用值:",sum_utility_new)
            #print("上一次循环的效用值:", sum_utility_old)
            #print(abs(sum_utility_new - sum_utility_old) / sum_utility_old)
            #print(epsilon)
            # if user_uplink_plus_computing_time < sum_delay_old:
            sum_utility_old = sum_utility_new
            p_temp = p.value

    if problem.value != None and not np.isnan(problem.value) and not np.isinf(problem.value):
        return sum_utility_new, p.value, power_no_sloved,power_time_iter
    else:
        return sum_utility_old, p_temp, power_no_sloved,power_time_iter
