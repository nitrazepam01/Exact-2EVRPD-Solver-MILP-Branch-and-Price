import cplex
import numpy as np
import math
import sys

class CardiffDataLoader:
    def __init__(self,file_path):
        #读取文件
        try:
            with open(file_path,'r') as f:
                lines=[line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"错误：找不到文件{file_path}")
            sys.exit(1) 
        
        #参数读取
        self.n=int(lines[0])    #客户数量
        self.k_v=int(lines[1])      #卡车数量
        self.k_d=int(lines[2])      #总无人机数量
        self.Gamma=int(lines[3])     #每辆车最多能带的无人机数量

        self.Q=float(lines[4])        #每辆车的载重
        self.q1_d=float(lines[5])         #无人机+配套设备
        self.q2_d=float(lines[6])      #无人机本身
        self.t_0=int(lines[7])   #装载+充电时间
        self.vel_v=float(lines[8])  #卡车速度
        self.vel_d=float(lines[9])  #无人机速度
        self.B_c=float(lines[10])   #电池容量
        self.Para=float(lines[11])   #功耗的系数

        matrix_start=12            
        self.dist_matrix=[]
        for row_index in range(self.n+1):
             line=lines[matrix_start+row_index]
             parts=line.split()
             current_row =[]
             for part in parts:
                 distance=float(part)
                 current_row.append(distance)
            
             self.dist_matrix.append(current_row)
        self.dist_matrix=np.array(self.dist_matrix)  #距离矩阵读取
                 
        self.t_v=self.dist_matrix / self.vel_v
        self.t_d=self.dist_matrix / self.vel_d       #时间矩阵预处理 

        self.demand={}   #客户需求
        self.l={}       #右时间窗
        self.ser_v={}   #车辆服务时间
        self.ser_d={}   #无人机服务时间
        self.Z_d = []   #可由无人机服务的客户
        self.f_d ={}    #无人机服务参数

        cust_info_start = matrix_start + self.n +1

        for line in lines[cust_info_start:]:
            parts=line.split()
            idx=int(parts[0])
            self.demand[idx]=float(parts[3])
            self.l[idx]=float(parts[5])
            self.ser_v[idx]=float(parts[6])
            self.ser_d[idx]=float(parts[7])
            if idx>0 and self.demand[idx]<=10 and parts[2]=='1':
                self.f_d[idx]=1
                self.Z_d.append(idx)
            else :
                self.f_d[idx]=0  
        #集合定义        
        self.V=list(range(self.n+1))        
        self.Z=list(range(1,self.n+1))
        self.K_v=list(range(self.k_v))
        self.K_d=list(range(self.k_d))

        #大M参数
        self.M = len(self.Z_d) 
        E = np.zeros((self.n + 1, self.n + 1)) 
        
        #能量参数
        for i in self.Z:
            for j in self.Z_d:
                if i==j:
                   continue
                P_load=(self.q2_d+self.demand[j])**1.5*self.Para
                P_empty=self.q2_d**1.5*self.Para
                energy_consumued=P_load*self.t_d[i,j]+P_empty*self.t_d[j,i]
                if energy_consumued <= self.B_c :
                    E[i,j]=1
                else :
                    E[i,j]=0 
        self.E=E

class MIPSolver:
    def __init__(self,prob):
        self.prob =prob
        self.pcpx=cplex.Cplex()
        self.formulate()
    def formulate(self):
        prob=self.prob
        pcpx=self.pcpx
        pcpx.objective.set_sense(pcpx.objective.sense.minimize)

        #关闭流输出
        # pcpx.set_log_stream(None)
        # pcpx.set_error_stream(None)
        # pcpx.set_warning_stream(None)
        # pcpx.set_results_stream(None)
        pcpx.set_log_stream(sys.stdout)
        pcpx.set_error_stream(sys.stdout)
        pcpx.set_warning_stream(sys.stdout)
        pcpx.set_results_stream(sys.stdout)
        #x_{i,j,k} 
        x_vars={}
        for k in prob.K_v:
            for i in prob.V:
                for j in prob.V:
                    if i != j:
                        x_vars[f"x_{i}_{j}_{k}"]=prob.t_v[i,j]
        pcpx.variables.add(names=list(x_vars.keys()),types=['B']*len(x_vars),obj=list(x_vars.values())) 
        
        #y_{d_k}
        y_vars=[f"y_{d}_{k}" for k in prob.K_v for d in prob.K_d]
        pcpx.variables.add(names=y_vars,types=['B']*len(y_vars))

        #h_{i_k}
        h_vars=[f"h_{i}_{k}" for i in prob.Z for k in prob.K_v]
        pcpx.variables.add(names=h_vars,types=['B']*len(h_vars))

        #u_{i,j}
        u_vars=[f"u_{i}_{j}" for i in prob.Z for j in prob.Z_d]
        pcpx.variables.add(names=u_vars,types=['B']*len(u_vars))

        #z_{i,j}
        z_vars=[f"z_{i}_{j}" for i in prob.Z for j in prob.Z_d]
        pcpx.variables.add(names=z_vars,types=['B']*len(z_vars))

        #omega_{i,k}
        omega_vars=[f"omega_{i}_{k}" for i in prob.Z for k in prob.K_v]
        pcpx.variables.add(
            names=omega_vars,
            types=['C']*len(omega_vars),
            lb=[0]*len(omega_vars),
            ub=[prob.Q]*len(omega_vars)
        )

        #a_i
        a_names=[f"a_{i}" for i in prob.V]
        a_ubs=[prob.l[i] for i in prob.V]
        pcpx.variables.add(
            names=a_names,
            types=['C']*len(a_names),
            ub=a_ubs
        )

        #phi_i
        pcpx.variables.add(
            names=[f"phi_{i}" for i in prob.Z],
            types=['C']*len(prob.Z),
            obj=[1.0]*len(prob.Z)
        )


        #约束 2
        for j in prob.Z:
            ind =[]
            for k in prob.K_v:
                for i in prob.V:
                    if i!=j:
                        ind.append(f"x_{i}_{j}_{k}")
            if j in prob.Z_d:
                for i in prob.Z:
                    if i!=j:
                        ind.append(f"u_{i}_{j}")
            if len(ind)>0:
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,[1.0]*len(ind))],
                    rhs=[1.0],
                    senses=['E']
                )            
        #约束3          
        for k in prob.K_v:
            ind =[f"x_{0}_{j}_{k}" for j in prob.Z]
            if ind:
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,[1.0]*len(ind))],
                    rhs=[1.0],
                    senses=['L']
                )     
        #约束4
        for k in prob.K_v:
            for j in prob.V:
                ind=[]
                val=[]
                for i in prob.V:
                    if i!=j:
                        ind.append(f"x_{i}_{j}_{k}")
                        val.append(1.0)
                        ind.append(f"x_{j}_{i}_{k}")
                        val.append(-1.0)
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[0.0],
                    senses=['E']
                )       
        #约束5
        for i in prob.Z:
            for j in prob.Z_d:
                if i==j:
                    continue
                ind=[f"u_{i}_{j}"]+[f"h_{i}_{k}" for k in prob.K_v]
                val=[1.0] +[-1.0]*len(prob.K_v)
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[0.0],
                    senses=['L']
                )
        #约束6            
        for i in prob.Z:
            for k in prob.K_v:
                ind =[f"h_{i}_{k}"]+[f"y_{d}_{k}" for d in prob.K_d]
                val=[1.0]+[-1.0]*len(prob.K_d)
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[0.0],
                    senses=['L']
                )
        #约束7
        for i in prob.Z:
            for k in prob.K_v:
                ind=[f"h_{i}_{k}"]+[f"x_{j}_{i}_{k}" for j in prob.V if j!=i]
                val=[1.0]+[-1.0]*(len(prob.V)-1)
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[0.0],
                    senses=['L']
                )            
        #约束8       
        for k in prob.K_v:
            ind =[f"y_{d}_{k}" for d in prob.K_d]
            pcpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind,[1.0]*len(ind))],
                rhs=[prob.Gamma],
                senses=['L']
            )
        #约束9
        for i in prob.Z:
            for j in prob.Z_d:
                if i!=j and prob.E[i,j]==0:
                        pcpx.variables.set_upper_bounds(f"u_{i}_{j}",0.0)
        #约束10
        for j in prob.Z_d:
            ind_in=[f"u_{i}_{j}" for i in prob.Z if i!=j]
            ind_out=[f"z_{i}_{j}" for i in prob.Z if i!=j]
            ind=ind_in+ind_out
            val=[1.0]*len(ind_in)+[-1.0]*len(ind_out)
            if len(ind)>0:
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[0.0],
                    senses=['E']
                )
        #约束11
        for i in prob.Z_d:
            for  j in prob.Z_d:
                if i== j:
                    continue
                ind =[f"z_{i}_{j}",f"u_{i}_{j}"]+[f"h_{i}_{k}" for k in prob.K_v]
                val=[1.0,-1.0]+[1.0]*len(prob.K_v)
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[1.0],
                    senses=['L']
                )
        #约束12
        for i in prob.Z:
            if i in prob.Z_d:
                continue
            for j in prob.Z_d:
                if i==j :
                    continue
                ind=[f"z_{i}_{j}"]+[f"u_{i}_{j}"]
                val=[1.0]+[-1.0]
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[0.0],
                    senses=['L']
                )
        #约束13
        for l in prob.Z:
            for i in prob.Z_d:
                if i==l:
                    continue
                for j in prob.Z_d:
                    if j==i or j==l:
                        continue
                    ind=[
                        f"u_{l}_{i}",
                        f"u_{l}_{j}",
                        f"z_{i}_{j}"
                    ]+[f"h_{i}_{k}" for k in prob.K_v]
                    val=[1.0,-1.0,1.0]+[-1.0]*len(prob.K_v)
                    pcpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind,val)],
                        rhs=[1.0],
                        senses=['L']
                    )

        #约束14
        for i in prob.Z_d:
            for j in prob.Z_d:
                if i==j:
                    continue
                ind=[f"z_{i}_{j}"]
                val=[1.0]
                for l in prob.Z:
                    if l!=i:
                        ind.append(f"u_{l}_{i}")
                        val.append(-1.0)
                for k in prob.K_v:
                    ind.append(f"h_{i}_{k}")
                    val.append(-1.0)
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[0.0],
                    senses=['L']
                )    
        #约束15
        for i in prob.Z:
            for k in prob.K_v:
                ind=[]
                val=[]

                for j in prob.Z_d:
                    if i!=j:
                        ind.append(f"z_{i}_{j}")
                        val.append(1.0)

                for d in prob.K_d:
                    ind.append(f"y_{d}_{k}")
                    val.append(-1.0)            

                ind.append(f"h_{i}_{k}")
                val.append(prob.M)

                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[prob.M],
                    senses=['L']
                )
        #约束16
        M_prime=len(prob.Z)-1
        for i in prob.Z:
            ind=[]
            val=[]
            for j in prob.Z_d:
                if i!=j:
                    ind.append(f"z_{i}_{j}")
                    val.append(1.0)
            
            for k in prob.K_v:
                ind.append(f"h_{i}_{k}")
                val.append(-M_prime)

            pcpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind,val)],
                rhs=[1.0],
                senses=['L']
            )    
        #约束17
        for j in prob.Z:
            M_j=prob.demand[j]+sum(prob.demand[i] for i in prob.Z_d)
            for  k in prob.K_v:
                ind=[]
                val=[]

                for i in prob.Z_d:
                    if i!=j:
                        ind.append(f"u_{j}_{i}")
                        val.append(prob.demand[i])
                ind.append(f"omega_{j}_{k}")
                val.append(-1.0)

                for i in prob.V:
                    if i!=j:
                        ind.append(f"x_{i}_{j}_{k}")
                        val.append(M_j)

                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[M_j-prob.demand[j]],
                    senses=['L']
                )
        #约束18
        for k in prob.K_v:
            ind =[]
            val =[]
            
            for j in prob.Z:
                ind.append(f"omega_{j}_{k}")
                val.append(1.0)

            for d in prob.K_d:
                ind.append(f"y_{d}_{k}")
                val.append(prob.q1_d)

            pcpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind,val)],
                rhs=[prob.Q],
                senses=['L']
            )    
                    
        #约束19
        for j in prob.Z:
            M_prime_j=prob.t_v[0,j]
            ind=[]
            val=[]
            ind.append(f"a_{j}")
            val.append(-1.0)
            for k in prob.K_v:
                ind.append(f"x_{0}_{j}_{k}")
                val.append(M_prime_j)
            pcpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind,val)],
                rhs=[0],
                senses=['L']
            )                
        #约束20
        for i in prob.Z:
            for j in prob.V:
                if i== j:
                    continue
                M_ij=prob.l[0]+prob.t_v[i,j]
                ind=[]
                val=[]
                ind.extend([f"a_{i}",f"phi_{i}",f"a_{j}"])
                val.extend([1.0,1.0,-1.0])

                for k in prob.K_v:
                    ind.append(f"x_{i}_{j}_{k}")
                    val.append(M_ij)

                rhs=M_ij-prob.t_v[i,j]
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[rhs],
                    senses=['L']
                )   
        #约束21
        for i in prob.Z:
            for j in prob.Z_d:
                if i==j:
                    continue
                M_prime_ij=prob.l[i]+prob.t_0+prob.t_d[i,j]
                ind=[]
                val=[]
                
                ind.extend([f"a_{i}",f"a_{j}"])
                val.extend([1.0,-1.0])

                ind.extend([f"z_{i}_{j}",f"u_{i}_{j}"])
                val.extend([M_prime_ij,M_prime_ij])

                rhs=2*M_prime_ij-prob.t_0-prob.t_d[i,j]

                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[rhs],
                    senses=['L']
                )
        #约束22
        for l in prob.Z:
            for i in prob.Z_d:
                if i==l:
                    continue
                for j in prob.Z_d:            
                    if i==j or j==l:
                        continue
                    M_ijl=prob.l[0]+prob.ser_d[i]+prob.t_d[i,l]+prob.t_0+prob.t_d[l,j]
                    ind=[]
                    val=[]
                    ind.extend([f"a_{i}",f"a_{j}"])
                    val.extend([1.0,-1.0])

                    ind.extend([f"z_{i}_{j}",f"u_{l}_{i}"])
                    val.extend([M_ijl,M_ijl])

                    rhs=2*M_ijl-prob.ser_d[i]-prob.t_d[i,l]-prob.t_0-prob.t_d[l,j]

                    pcpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind,val)],
                        rhs=[rhs],
                        senses=['L']
                        
                    )
        #约束23
        for i in prob.Z:
            ind=[]
            val=[]

            ind.append(f"phi_{i}")
            val.append(1.0)

            for j in prob.V:
                if j==i:
                    continue
                for k in prob.K_v:
                    ind.append(f"x_{j}_{i}_{k}")
                    val.append(-prob.ser_v[i])
            pcpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind,val)],
                rhs=[0.0],
                senses=['G']
            )
        #约束24
        for i in prob.Z:
            for j in prob.Z_d:
                if i==j :
                    continue
                M_double_ij=prob.l[0]+prob.ser_d[j]+prob.t_d[j,i]
                ind=[]
                val=[]
                ind.extend([f"a_{j}",f"a_{i}",f"phi_{i}"])
                val.extend([1.0,-1.0,-1.0])

                ind.append(f"u_{i}_{j}")
                val.append(M_double_ij)

                rhs=M_double_ij-prob.ser_d[j]-prob.t_d[j,i]
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind,val)],
                    rhs=[rhs],
                    senses=['L']
                )
        
        #破除对称性
        # 1. 车辆使用顺序约束：
        # 如果车辆 k 未被使用（没离开过仓库），则车辆 k+1 也不能被使用。
        # 即：sum(x_{0,j,k}) >= sum(x_{0,j,k+1})
        for k in range(prob.k_v - 1):
            ind = []
            val = []
            # 车辆 k 离开仓库的边
            for j in prob.Z:
                ind.append(f"x_{0}_{j}_{k}")
                val.append(1.0)
            # 车辆 k+1 离开仓库的边
            for j in prob.Z:
                ind.append(f"x_{0}_{j}_{k+1}")
                val.append(-1.0)
            
            if ind:
                pcpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind, val)],
                    rhs=[0.0],
                    senses=['G'] # >= 0
                )

        # 2. 车辆载重优先约束 (可选，进一步加强)：
        # 编号小的车辆携带的无人机数量 >= 编号大的车辆
        # sum(y_{d,k}) >= sum(y_{d,k+1})
        for k in range(prob.k_v - 1):
            ind = []
            val = []
            for d in prob.K_d:
                ind.append(f"y_{d}_{k}")
                val.append(1.0)
                ind.append(f"y_{d}_{k+1}")
                val.append(-1.0)
            
            pcpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind, val)],
                rhs=[0.0],
                senses=['G']
            )
        # 3. 强力剪枝：闲置车辆不带无人机
        # 如果车辆 k 没有离开仓库 (sum(x_{0,j,k}) == 0)，则携带的无人机数量必须为 0
        # 实现方式：sum(y_{d,k}) <= Gamma * sum(x_{0,j,k})
        for k in prob.K_v:
            ind = []
            val = []
            # 车辆 k 携带的所有无人机
            for d in prob.K_d:
                ind.append(f"y_{d}_{k}")
                val.append(1.0)
            
            # 车辆 k 离开仓库的边 (作为开关)
            for j in prob.Z:
                ind.append(f"x_{0}_{j}_{k}")
                val.append(-prob.Gamma) # 移项到左边：sum(y) - Gamma*sum(x) <= 0
            
            pcpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind, val)],
                rhs=[0.0],
                senses=['L']
            )
    def solve(self):

        print("="*60)
        print("开始求解")
        print(f"客户: {self.prob.n}, 车辆: {self.prob.k_v}, 无人机: {self.prob.k_d}, 容量: {self.prob.Q}")
        print("=" * 60)
        print(f"可行无人机航线数量: {np.sum(self.prob.E)}")
        print(f"Z_d (无人机客户) 数量: {len(self.prob.Z_d)}")

        try:
            self.pcpx.parameters.timelimit.set(3600)
            self.pcpx.parameters.mip.tolerances.mipgap.set(0.01)
            self.pcpx.parameters.mip.tolerances.integrality.set(1e-5)
            self.pcpx.solve()
            status=self.pcpx.solution.get_status()
            
            if self.pcpx.solution.is_primal_feasible():
                objective = self.pcpx.solution.get_objective_value()
                print(f"目标函数值 (总等待时间): {objective:.2f}")
                
                # 尝试打印 Gap 信息（如果有的话）
                try:
                    gap = self.pcpx.solution.MIP.get_mip_relative_gap()
                    print(f"当前 Gap: {gap * 100:.2f}%")
                except:
                    pass

                self._extract_solution()
                return True
            else:
                 print("未找到可行解或求解失败")
                 return False
        except cplex.CplexSolverError as e:
            print(f"CPLEX求解错误: {e}")
            return False      

    def _extract_solution(self): 
        prob = self.prob
        pcpx = self.pcpx

        var_names = pcpx.variables.get_names()
        var_values = pcpx.solution.get_values()
        solution = dict(zip(var_names, var_values))   

        print("\n" + "="*60)
        print("求解结果详情")
        print("="*60)

        vehicle_routes = {k: [] for k in prob.K_v} 
        for k in prob.K_v:
            route_edges = []
            for i in prob.V:
                for j in prob.V:
                    if i != j:
                        var_name = f"x_{i}_{j}_{k}"
                        if solution.get(var_name, 0) > 0.5:  # 二元变量，>0.5视为1
                            route_edges.append((i, j))
            if route_edges:
                vehicle_routes[k] = self._reconstruct_route(route_edges)
                print(f"\n车辆 {k+1} 路径: {' -> '.join(map(str, vehicle_routes[k]))}")
                travel_time = sum(prob.t_v[i,j] for i,j in zip(vehicle_routes[k][:-1], vehicle_routes[k][1:]))
                print(f"  行驶时间: {travel_time:.2f}")
        drone_assignment = {k: [] for k in prob.K_v}
        for k in prob.K_v:
            drones = []
            for d in prob.K_d:
                var_name = f"y_{d}_{k}"
                if solution.get(var_name, 0) > 0.5:
                    drones.append(d)
            drone_assignment[k] = drones
            if drones:
                print(f"\n车辆 {k+1} 携带无人机: {len(drones)} 架 (编号: {drones})")
        drone_services = {}  # (发射点, 无人机客户) -> 1
        for i in prob.Z:
            for j in prob.Z_d:
                var_name = f"u_{i}_{j}"
                if solution.get(var_name, 0) > 0.5:
                    drone_services[(i, j)] = 1
                    print(f"  从客户 {i} 派出无人机服务客户 {j}")        
        print("无人机路径详情")
        print("="*60)
        
        for l in prob.Z:  # 对每个车辆停靠点（发射点）
            # 检查是否有无人机从这里出发
            has_drones = any(solution.get(f"u_{l}_{j}", 0) > 0.5 for j in prob.Z_d)
            if not has_drones:
                continue
            
            print(f"\n【车辆节点 {l}】的无人机任务：")
            
            # 找出所有从 l 出发的"第一个客户"（每个无人机一个）
            # z_{l,j}=1 表示 j 是某个无人机的第一个任务
            first_customers = []
            for j in prob.Z_d:
                if solution.get(f"z_{l}_{j}", 0) > 0.5:
                    first_customers.append(j)
            
            # 为每个首客户重构完整的无人机任务链
            for drone_id, start in enumerate(first_customers, 1):
                route = [l, start]  # 起点：车辆节点 -> 第一个客户
                current = start
                
                # 追踪后续任务（连续访问的客户）
                while True:
                    next_customer = None
                    for j in prob.Z_d:
                        if j != current and solution.get(f"z_{current}_{j}", 0) > 0.5:
                            next_customer = j
                            break
                    
                    if next_customer:
                        # 完成任务 current 后返回 l，然后去 next_customer
                        route.extend([l, next_customer])
                        current = next_customer
                    else:
                        # 没有后续任务，返回车辆节点结束
                        route.append(l)
                        break
                
                # 格式化打印路径
                path_str = " -> ".join(map(str, route))
                print(f"  无人机 {drone_id}: {path_str}")
                
                # 显示时间信息
                arrival_l = solution.get(f"a_{l}", 0)
                arrival_first = solution.get(f"a_{start}", 0)
                print(f"    车辆到达 {l} 时间: {arrival_l:.2f}, "
                      f"无人机首次出发到达 {start} 时间: {arrival_first:.2f}")
                
                # 计算该无人机的总飞行时间（所有往返）
                total_flight = 0
                for idx in range(1, len(route)-1, 2):  # 每次 l->customer
                    if idx+1 < len(route):
                        cust = route[idx]
                        total_flight += prob.t_d[l, cust] + prob.t_d[cust, l]
                print(f"    预估总飞行时间: {total_flight:.2f}") 
        print("\n" + "-"*60)
        print("时间信息:")
        for i in prob.V:
            arrival_time = solution.get(f"a_{i}", 0)
            print(f"  节点 {i}: 到达时间 = {arrival_time:.2f}, 截止时间 = {prob.l[i]}")
        
        for i in prob.Z:
            waiting_time = solution.get(f"phi_{i}", 0)
            if waiting_time > 0.01:  # 只显示有意义的等待时间
                print(f"  客户 {i}: 等待时间 = {waiting_time:.2f}")
        
        # 6. 验证总时间（目标函数验证）
        total_waiting = sum(solution.get(f"phi_{i}", 0) for i in prob.Z)
        total_travel = sum(solution.get(f"x_{i}_{j}_{k}", 0) * prob.t_v[i,j] 
                          for k in prob.K_v for i in prob.V for j in prob.V if i!=j)
        print(f"\n总等待时间: {total_waiting:.2f}")
        print(f"总行驶时间: {total_travel:.2f}")
        print(f"总路线时长 (目标函数): {total_waiting + total_travel:.2f}")
    def _reconstruct_route(self, edges):
        adj = {}
        for i, j in edges:
            adj[i] = j  # 假设每个节点只有一个出边（车辆路径是简单路径）
        
        # 从仓库0开始追踪
        route = [0]
        current = 0
        visited = set([0])
        
        while current in adj and adj[current] not in visited:
            next_node = adj[current]
            route.append(next_node)
            visited.add(next_node)
            current = next_node
            
            # 安全检查，防止无限循环
            if len(route) > len(edges) + 2:
                break
        
        # 如果最后不是回到0，添加回到0（如果是闭环）
        if route[-1] != 0 and 0 in adj.values():
            route.append(0)
            
        return route
    def export_solution(self, filename="solution.txt"):
        """将解导出到文件"""
        try:
            with open(filename, 'w') as f:
                f.write("2E-VRP-D 求解结果\n")
                f.write("="*60 + "\n")
                f.write(f"目标值: {self.pcpx.solution.get_objective_value():.2f}\n")
                f.write(f"求解时间: {self.pcpx.get_time():.2f} 秒\n")
                f.write(f"最佳界限: {self.pcpx.solution.MIP.get_best_objective():.2f}\n")
                f.write(f"间隙: {self.pcpx.solution.MIP.get_mip_relative_gap()*100:.2f}%\n")
        except Exception as e:
            print(f"导出失败: {e}")               
if __name__ == "__main__":


    # 检查命令行参数或直接使用默认文件
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "Cardiff10_01.txt"  # 默认测试文件
    
    # 加载数据
    try:
        prob = CardiffDataLoader(data_file)
        print(f"成功加载数据文件: {data_file}")
        
        # 创建并求解模型
        solver = MIPSolver(prob)
        solver.solve()
        
        # 可选：导出结果
        # solver.export_solution("result.txt")
        
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()        