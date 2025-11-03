"""
@author: Ning zhe Shi
"""
import geatpy as ea
import numpy as np
import project_backend as pb


class MyProblem_Drop(ea.Problem):  # 继承geatpy中的Problem父类

    def __init__(self, H_data, P_data, u, n, m, cpu_cycles, cell_mapping, T_requirement, T_p, Cpu_matrix, Task_matrix, W, NOMA, dropped,
                 z_n, weighted, queue_number=None, channel_updated=None):
        name = 'z'
        self.U = u
        self.N = n
        self.m = m  # 采用小写避免与Problem类中的self.M相撞
        self.W = W
        self.cpu_cycles = cpu_cycles
        self.cell_mapping = cell_mapping
        self.NOMA = NOMA
        self.H_all_2 = H_data
        self.p_temp = P_data
        self.cpu_matrix = Cpu_matrix
        self.task_matrix = Task_matrix
        self.T_p = T_p
        self.T_requirement = T_requirement
        self.dropped = dropped
        self.z_n = z_n
        self.weighted = weighted
        self.fixchannel = channel_updated
        self.queue_number = queue_number
        # self.se_r = SE_r.reshape(-1, 1)
        M = 1  # 目标维数
        maxormins = [-1]  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
        Dim = self.U  # 决策变量维数
        varTypes = [1] * Dim  # 决策变量的类型列表，0：实数；1：整数
        if self.dropped:
            lb = [0] * Dim  # 决策变量下界
        else:
            lb = [1] * Dim  # 未开启drop时所选信道号至少为1
        ub = [self.N] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 定义目标函数（含约束）
        Vars_rows, Vars_columns = Vars.shape
        objective = np.zeros((Vars_rows, 1))
        # 计算违反约束程度
        CV = np.zeros((Vars_rows, self.U + self.m))
        for k in range(Vars_rows):
            vars_temp = Vars[k, :]  # 第k个种群的决策变量，为U*1的矩阵，里面存储的为U个用户决策的信道号
            
            # 如果fixchannel存在，则更新vars_temp
            if self.fixchannel is not None:
                vars_temp[self.fixchannel == 0] = 0
            if self.queue_number is not None:
                # 基于功率和信道分配计算上传延迟
                uplink_latency,_ = pb.user_uplink_time(self.H_all_2, vars_temp, self.p_temp, self.task_matrix, self.U, self.W,
                                                            self.m, self.cell_mapping, self.T_p, self.NOMA,
                                                            self.dropped,self.weighted)
                # 基于FIFO和queue_number规则，并根据上传延迟计算排队延迟和计算延迟
                computing_latency_inital, queue_latency_inital = pb.queue_computing_delay(self.m, self.cpu_matrix, self.task_matrix,
                                                                                self.cpu_cycles, self.cell_mapping,
                                                                                vars_temp,uplink_latency.flatten(),self.queue_number)
                T_m_k_c_matrix = queue_latency_inital + computing_latency_inital  # 总MEC的时延
            else:
                # 根据分配信道,利用KKT计算最佳计算资源分配,并计算时延约束
                T_m_k_c_matrix,_ = pb.optimal_compute_resource_allocation(self.m, self.cpu_matrix, self.task_matrix,
                                                                            self.cpu_cycles, self.cell_mapping,
                                                                            vars_temp)  # 最优分配
            
            t_constraint = ((self.T_requirement - T_m_k_c_matrix) * self.W / self.task_matrix).reshape(-1, 1)

            H_large_t = np.zeros((self.U, self.U))
            for i in range(self.U):
                H_large_t[i, :] = self.H_all_2[i, :,
                                  vars_temp[i] - 1]  # 取出第i个用户在对应信道上与其余所有用户所在小区基站的信道增益矩阵,如果被丢弃即信道号为0则随机取即可，不影响，所以这里直接取-1
            H_gain_data = H_large_t.T  # 转置对应上SEGA中求解方式中的信道增益矩阵
            # 构建干扰矩阵
            interference_matrix = np.zeros((self.U, self.U))
            # 根据用户选择的子信道号设置干扰矩阵
            for i in range(self.U):
                for j in range(self.U):
                    if vars_temp[i] == vars_temp[j] and vars_temp[j] != 0:
                        interference_matrix[i, j] = 1

            OMA_temp = np.multiply(interference_matrix, 1 - np.eye(self.U))
            OMA_temp[OMA_temp >= 1] = 1

            if self.NOMA is True:
                A_temp = pb.NOMA_handle(vars_temp, OMA_temp, self.m, self.cell_mapping, H_gain_data, self.p_temp)
            else:
                A_temp = OMA_temp

            y_star = np.sqrt(np.diag(H_gain_data).reshape(-1, 1) * self.p_temp) / (
                    H_gain_data * A_temp @ self.p_temp + 1)  # 公式（33）

            # 识别被丢弃的用户（信道号为0）
            discarded_users = vars_temp.reshape(-1, 1) == 0
            # 计算所有用户违反约束的情况值
            all_users_cv_value = (1.0 / np.log2(1 + 2 * np.multiply(y_star, np.sqrt(
                np.multiply(np.diag(H_gain_data).reshape(-1, 1), self.p_temp))) - np.multiply(np.square(y_star), (
                    H_gain_data * A_temp @ self.p_temp + 1)))) - t_constraint  # 所有U个用户违反的情况值
            
            all_users_com_delay = (self.task_matrix.reshape(-1, 1) / (self.W * np.log2(1 + 2 * np.multiply(y_star, np.sqrt(
                np.multiply(np.diag(H_gain_data).reshape(-1, 1), self.p_temp))) - np.multiply(np.square(y_star), (
                    H_gain_data * A_temp @ self.p_temp + 1)))))  # 所有U个用户的上传时延
            

            # # 计算目标函数，对被丢弃的用户应用惩罚项
            # objective_components_1 = np.where(
            #     discarded_users | (all_users_cv_value > 0), # 被丢弃的用户或违反约束的用户
            #     -self.weighted*self.T_p,
            #     np.multiply(
            #         self.task_matrix.reshape(-1, 1),
            #             1.0 / (self.W * (np.log2(
            #                 1 + 2 * np.multiply(y_star,
            #                                     np.sqrt(np.multiply(np.diag(H_gain_data).reshape(-1, 1), self.p_temp)))
            #                 - np.multiply(np.square(y_star), (H_gain_data * A_temp @ self.p_temp + 1))
            #             )))
            #     )
            # )
            all_users_gain = (self.T_requirement - T_m_k_c_matrix).reshape(-1,1) - all_users_com_delay
                        # 计算目标函数，对被丢弃的用户应用惩罚项
            objective_components = np.where(
                discarded_users | (all_users_cv_value > 0), # 被丢弃的用户或违反约束的用户
                -self.weighted*self.T_p,
                all_users_gain
            )
            # 计算最终的目标值
            objective[k] = np.sum(objective_components)
            # objective[k] = np.sum(np.multiply(self.task_matrix.reshape(-1, 1), np.square(1.0 / ((np.log(
            #     1 + 2 * np.multiply(y_star, np.sqrt(
            #         np.multiply(np.diag(H_gain_data).reshape(-1, 1), self.p_temp))) - np.multiply(
            #         np.square(y_star), (
            #                 H_gain_data * A_temp @ self.p_temp + 1))))))))  # 公式（32）

            # for i in range(self.U):
            #     CV[k][i] = np.sum(alp_temp, axis=1)[i] - 1
            # 避免循环赋值
            # CV[k][:self.U] = np.sum(alp_temp, axis=1) - 1  # 一定满足
            # for j in range(self.U):
            #     CV[k][j + self.U] = (1.0 / ((np.log2(
            #         1 + 2 * np.multiply(y_star, np.sqrt(
            #             np.multiply(np.diag(H_gain_data).reshape(-1, 1), self.p_temp))) - np.multiply(
            #             np.square(y_star), (
            #                     H_gain_data * A_temp @ self.p_temp + 1))))))[j] - self.t_constraint[j]
            # 对 CV[k] 的后续 self.U 个元素进行赋值，避免上述循环赋值
            CV[k][: self.U] = np.where(discarded_users, 0, all_users_cv_value).flatten()  # 如果被丢弃则无违反情况赋值0
            # CV[k][2 * self.U] = np.mean(self.se_r) - np.mean(np.log(
            #     1 + 2 * np.multiply(y_star, np.sqrt(
            #         np.multiply(np.diag(H_gain_data).reshape(-1, 1), self.p_temp))) - np.multiply(
            #         np.square(y_star), (
            #                 H_gain_data * A_temp @ self.p_temp + 1))))
            icell_allsubchanel_maxusers_count = pb.subchannel_users_count(vars_temp, self.cell_mapping,
                                                                          self.m)  # 各个小区内各个子信道上的用户最大数量
            # for icell in range(self.m):
            #     # 小区内noma上所有子信道上用户数最多为2
            #     CV[k][2 * self.U + icell] = icell_allsubchanel_maxusers_count[icell] - 2
            # 避免循环赋值
            CV[k][self.U:self.U + self.m] = icell_allsubchanel_maxusers_count.reshape(-1) - self.z_n  # z_n为每个子信道上最大的用户数
        return objective, CV
