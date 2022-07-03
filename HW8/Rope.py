import os
import numpy as np 
from utils import *


class Mass(object):
    def __init__(self, pos, mass, pinned):
        '''
        @param
            `pos` --type=np.array --shape=(2,) --description="2D position"
            `mass` --type=float
            `pinned` --type=boolean 
        '''
        self.start_pos = pos.copy() # 初始位置
        self.pos = pos.copy()       # 当前位置
        self.last_pos = pos.copy()  # 上一时刻的位置

        self.v = np.zeros(2, dtype=np.float32) # 速度
        self.F = np.zeros(2, dtype=np.float32) # 受作用的力

        self.mass = mass 
        self.pinned = pinned


class Spring(object):
    def __init__(self, m1, m2, k):
        '''
        @param
            `m1, m2` --type=Mass 
            `k` --阻尼系数
        '''
        self.k = k
        self.m1 = m1 
        self.m2 = m2 
        self.l = magnitude_a(m1.pos-m2.pos) # rest of length


class Rope(object):
    def __init__(self, start, end, n_nodes, node_mass, k, pinned_nodes):
        '''
        @param
            `start, end` --type=np.array --shape=(2,) # the start and end positions
            `n_nodes` --type=int --description="节点数量"
            `node_mass` --type=float --description="每个节点的质量"
            `k` --type=float --description="阻尼系数"
            `pinned_nodes` --type=list<int> --description="哪些节点要被钉住"
        '''
        if n_nodes <= 1:
            return 
        self.masses = [] 
        self.springs= [] 

        curr_pos = start.copy()
        m = Mass(start, node_mass, False)
        self.masses.append(m)
        for i in range(1, n_nodes):
            if i == n_nodes-1:
                curr_pos = end.copy()
            else:
                curr_pos = start + i * (end-start)/(n_nodes-1)
            prev_m = self.masses[-1]
            curr_m = Mass(curr_pos, node_mass, False)
            s = Spring(prev_m, curr_m, k)
            self.masses.append(curr_m)
            self.springs.append(s)

        for i in pinned_nodes:
            self.masses[i].pinned = True 

    
    def simulate_Euler(self, delta_t, gravity):
        ## 1. 计算每个质点所受到的弹力, 并加和到所受的合力中
        for s in self.springs:
            '''
            ├a ---- b
            '''
            ab = s.m2.pos - s.m1.pos  
            f_ab = s.k * (ab/(magnitude_a(ab)+1e-6)) * (magnitude_a(ab)-s.l)
            s.m1.F += f_ab 
            s.m2.F += -f_ab
        ## 2. 
        i = 0
        for m in self.masses:
            if m.pinned:
                m.F = np.zeros(2, dtype=np.float32) # reset
                continue
            # 2.1 计算每个质点所受到的重力
            m.F += gravity*m.mass 

            # 2.2 计算每个质点所受到的空气阻力
            k_d_global  = .01
            m.F += -k_d_global*m.v 

            # 2.3 更新瞬时速度和位置
            a = m.F/m.mass 
            # 2.3.1 explicit method
            # m.pos += m.v*delta_t
            # m.v += a*delta_t
            # 2.3.2 semi-implicit method
            m.v += a*delta_t
            m.pos += m.v*delta_t
            m.F = np.zeros(2, dtype=np.float32) # reset


    def simulate_Verlet(self, delta_t, gravity):
        ## 1. 计算每个质点所受到的弹力, 并加和到所受的合力中
        for s in self.springs:
            '''
            ├a ---- b
            '''
            ab = s.m2.pos - s.m1.pos  
            f_ab = s.k * (ab/(magnitude_a(ab)+1e-6)) * (magnitude_a(ab)-s.l)
            s.m1.F += f_ab 
            s.m2.F += -f_ab
        ## 2. 
        for m in self.masses:
            if m.pinned:
                m.F = np.zeros(2, dtype=np.float32) # reset
                continue
            # 2.1 计算每个质点所受到的重力
            m.F += gravity*self.m.mass 

            # 2.3 更新瞬时速度和位置
            a = m.F/m.mass 
            
            last_pos = m.pos.copy() 
            damp_factor = .00005
            m.pos = m.pos + (1-damp_factor) * (m.pos-m.last_pos) + a * delta_t * delta_t
            m.last_pos = last_pos

            m.F = np.zeros(2, dtype=np.float32) # reset