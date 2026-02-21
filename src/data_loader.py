"""
CardiffDataLoader - extracted from MILP.py for reuse across B&P modules.
"""

import numpy as np
import sys


class CardiffDataLoader:
    def __init__(self, file_path):
        # 读取文件
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"错误：找不到文件{file_path}")
            sys.exit(1)

        # 参数读取
        self.n = int(lines[0])         # 客户数量
        self.k_v = int(lines[1])       # 卡车数量
        self.k_d = int(lines[2])       # 总无人机数量
        self.Gamma = int(lines[3])     # 每辆车最多能带的无人机数量

        self.Q = float(lines[4])       # 每辆车的载重
        self.q1_d = float(lines[5])    # 无人机+配套设备重量
        self.q2_d = float(lines[6])    # 无人机本身重量
        self.t_0 = int(lines[7])       # 装载+充电时间
        self.vel_v = float(lines[8])   # 卡车速度
        self.vel_d = float(lines[9])   # 无人机速度
        self.B_c = float(lines[10])    # 电池容量
        self.Para = float(lines[11])   # 功耗的系数

        matrix_start = 12
        self.dist_matrix = []
        for row_index in range(self.n + 1):
            line = lines[matrix_start + row_index]
            parts = line.split()
            current_row = []
            for part in parts:
                distance = float(part)
                current_row.append(distance)
            self.dist_matrix.append(current_row)
        self.dist_matrix = np.array(self.dist_matrix)

        self.t_v = self.dist_matrix / self.vel_v
        self.t_d = self.dist_matrix / self.vel_d

        self.demand = {}
        self.l = {}
        self.ser_v = {}
        self.ser_d = {}
        self.Z_d = []
        self.f_d = {}

        cust_info_start = matrix_start + self.n + 1

        for line in lines[cust_info_start:]:
            parts = line.split()
            idx = int(parts[0])
            self.demand[idx] = float(parts[3])
            self.l[idx] = float(parts[5])
            self.ser_v[idx] = float(parts[6])
            self.ser_d[idx] = float(parts[7])
            if idx > 0 and self.demand[idx] <= 10 and parts[2] == '1':
                self.f_d[idx] = 1
                self.Z_d.append(idx)
            else:
                self.f_d[idx] = 0

        # 集合定义
        self.V = list(range(self.n + 1))
        self.Z = list(range(1, self.n + 1))
        self.K_v = list(range(self.k_v))
        self.K_d = list(range(self.k_d))

        # 大M参数
        self.M = len(self.Z_d)
        E = np.zeros((self.n + 1, self.n + 1))

        # 能量参数
        for i in self.Z:
            for j in self.Z_d:
                if i == j:
                    continue
                P_load = (self.q2_d + self.demand[j]) ** 1.5 * self.Para
                P_empty = self.q2_d ** 1.5 * self.Para
                energy_consumed = P_load * self.t_d[i, j] + P_empty * self.t_d[j, i]
                if energy_consumed <= self.B_c:
                    E[i, j] = 1
                else:
                    E[i, j] = 0
        self.E = E
