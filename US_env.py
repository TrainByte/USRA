# -*- coding: utf-8 -*-
"""
@author: ningzhe
"""

import numpy as np

class userselection_subchannelscheduling:
    def __init__(self, options,options_policy,u,m,n,noise_var,seed=None):
        self.total_samples = options['simulation']['total_samples']
        self.train_episodes = options['train_episodes']
        R_defined = options['simulation']['R_defined']
        self.R = (2.0/np.sqrt(3))*R_defined
        self.N = n  # 子信道数量
        self.M = m  # 小区数
        self.U = u  # 总用户数
        self.noise_var = noise_var
        self.seed = seed
        
        scale_R_inner = options_policy['scale_R_inner']
        scale_R_interf = options_policy['scale_R_interf']
        

        scale_g_dB_R = scale_R_inner*self.R
        rb = 200.0
        if(scale_g_dB_R < rb):
            scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_R))
        else:
            scale_g_dB = - (128.1 + 37.6* np.log10(scale_g_dB_R/rb) + 37.6* np.log10(0.001*rb)) 
        self.scale_gain = np.power(10.0,scale_g_dB/10.0)
        self.input_placer = np.log10(self.noise_var/self.scale_gain)
        scale_g_dB_inter_R = scale_R_interf * self.R
        if(scale_g_dB_R < rb):
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_inter_R))
        else:
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(scale_g_dB_inter_R/rb) + 37.6* np.log10(0.001*rb))
        self.scale_gain_interf = np.power(10.0,scale_g_dB_interf/10.0)
        
        
        self.policynum_input = (7+n)
        self.policynum_actions = (n+1) # Kumber of actions  每个用户有N+1个选择
    
        self.previous_state = np.zeros((u,self.policynum_input))
        self.previous_action = np.ones(u) * 1
        self.previous_action_logprob = np.ones(u) * 1
        self.previous_reward = np.ones(u)

       
        
    def local_state(self, agent, H_all_2,cpu_matrix, task_matrix, cell_mapping,T_requirement_matrix,weighted,US_list,reward_list,status_matrix_list):
        global_state = np.zeros(self.policynum_input)
        local_state_input = self.policynum_input // self.N
        # for n in range(self.N):
            # global_state[n*local_state_input:(n+1)*local_state_input] = self.state_concatenate(agent,n,H_all_2,cpu_matrix, task_matrix, cell_mapping,T_requirement_matrix,weighted)
        global_state = self.state_concatenate(agent,H_all_2,cpu_matrix, task_matrix, cell_mapping,T_requirement_matrix,weighted,US_list,reward_list,status_matrix_list)
        return global_state
    
    def state_concatenate(self,agent,H_all_2,cpu_matrix, task_matrix, cell_mapping,T_requirement_matrix,weighted,US_list,reward_list,status_matrix_list):
        state = np.zeros(self.policynum_input)
        cursor = 0

        state[cursor:cursor+self.N] = np.log10(H_all_2[agent,agent,:]/self.scale_gain)
        cursor += self.N

        state[cursor] = cpu_matrix[agent] / 1000
        cursor += 1

        state[cursor] = task_matrix[agent] / 10000
        cursor += 1

        state[cursor] = T_requirement_matrix[agent] * 1000
        cursor += 1

        state[cursor] = weighted[agent]
        cursor += 1
        
        state[cursor] = US_list[agent]
        cursor += 1

        state[cursor] = reward_list[agent]
        cursor += 1

        state[cursor] = status_matrix_list[agent]
        cursor += 1
        
        return state
