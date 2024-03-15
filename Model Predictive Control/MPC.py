import numpy as np
from pyomo.environ import *
from pyomo.dae import *

N = 9 # 前向预测步数
s_dim = 6  # 状态: 1: x, 2: y, 3: ψ, 4: u, 5: v, 6: r

class MPC(object):
    def __init__(self, s_0, s_r):

        model = ConcreteModel() # 创建模型
        
        model.state_range = RangeSet(0, N-1) # N组状态
        model.u_range = RangeSet(0, N-2) # N-1组输入
        
        # 时间间隔
        dt = 0.1

        # 动力学参数

        M_11 = 25.8
        M_22 = 33.8
        M_23 = 1.19

        M_inv_11 = 0.03875969
        M_inv_22 = 0.03004183
        M_inv_23 =-0.01295282
        M_inv_32 =-0.01295282
        M_inv_33 = 0.36790357
        
        X_u = -0.72
        X_uu = -1.33
        X_uuu = -5.8664

        Y_v = -0.86
        Y_r = 0.11
        Y_vv = -36.28
        Y_vr = -0.85
        Y_rv = -0.81
        Y_rr = -3.45

        N_v = 0.11
        N_r = -1.90
        N_vv = 5.04
        N_vr = 0.08
        N_rv = 0.13
        N_rr = -0.75
        
        # 受控船舶初始状态
        self.s_0 = s_0
        
        # 领航船舶参考轨迹
        self.s_r = s_r

        # 决策变量 
        model.s = Var(model.state_range,RangeSet(0, s_dim-1))
        model.tau_1 = Var(model.u_range, bounds=(-10.0, 10.0))
        model.tau_2 = Var(model.u_range, bounds=(-10.0, 10.0))
        model.tau_3 = Var(model.u_range, bounds=( -3.0, 3.0))
        
        # 0: x, 1: y, 2: ψ, 3: u, 4: v, 5: r
        # 约束

        # 初始状态约束
        model.s0_update = Constraint(RangeSet(0, s_dim-1), rule = lambda m, i: m.s[0,i] == self.s_0[i])

        # x约束
        model.x_update = Constraint(model.state_range, rule=lambda m, k:
                                    m.s[k+1,0] == m.s[k,0] + (m.s[k,3]*cos(m.s[k,2])-m.s[k,4]*sin(m.s[k,2]))*dt
                                    if k<N-1 else Constraint.Skip)
        
        # y约束
        model.y_update = Constraint(model.state_range, rule=lambda m, k: 
                                    m.s[k+1,1] == m.s[k,1] + (m.s[k,3]*sin(m.s[k,2])+m.s[k,4]*cos(m.s[k,2]))*dt
                                    if k<N-1 else Constraint.Skip)
        
        # ψ约束
        model.ψ_update = Constraint(model.state_range, rule=lambda m, k: 
                                    m.s[k+1,2] == m.s[k,2] + m.s[k,5]*dt
                                    if k<N-1 else Constraint.Skip)
        
        # u约束
        model.u_update = Constraint(model.state_range, rule=lambda m, k:
                                    m.s[k+1,3] == m.s[k,3] + 
                                    M_inv_11*(m.tau_1[k]-((-X_u-X_uu*abs(m.s[k,3])-X_uuu*pow(m.s[k,3],2))*m.s[k,3]+(-M_22*m.s[k,4]-M_23*m.s[k,5])*m.s[k,5]))*dt
                                    if k<N-1 else Constraint.Skip)
        
        # v约束
        model.v_update = Constraint(model.state_range, rule=lambda m, k: 
                                    m.s[k+1,4] == m.s[k,4] +
                                    (M_inv_22*(m.tau_2[k]-((-Y_v-Y_vv*abs(m.s[k,4])-Y_rv*abs(m.s[k,5]))*m.s[k,4]+(M_11*m.s[k,3]-Y_r-Y_vr*abs(m.s[k,4])-Y_rr*abs(m.s[k,5]))*m.s[k,5])) +
                                     M_inv_23*(m.tau_3[k]-((M_22*m.s[k,4]+M_23*m.s[k,5])*m.s[k,3]+(-M_11*m.s[k,3]-N_v-N_vv*abs(m.s[k,4])-N_rv*abs(m.s[k,5]))*m.s[k,4]+(-N_r-N_vr*abs(m.s[k,4]-N_rr*abs(m.s[k,5])))*m.s[k,5])))*dt
                                    if k<N-1 else Constraint.Skip)

        # r约束
        model.r_update = Constraint(model.state_range, rule=lambda m, k: 
                                    m.s[k+1,5] == m.s[k,5] +
                                    (M_inv_32*(m.tau_2[k]-((-Y_v-Y_vv*abs(m.s[k,4])-Y_rv*abs(m.s[k,5]))*m.s[k,4]+(M_11*m.s[k,3]-Y_r-Y_vr*abs(m.s[k,4])-Y_rr*abs(m.s[k,5]))*m.s[k,5])) +
                                     M_inv_33*(m.tau_3[k]-((M_22*m.s[k,4]+M_23*m.s[k,5])*m.s[k,3]+(-M_11*m.s[k,3]-N_v-N_vv*abs(m.s[k,4])-N_rv*abs(m.s[k,5]))*m.s[k,4]+(-N_r-N_vr*abs(m.s[k,4]-N_rr*abs(m.s[k,5])))*m.s[k,5])))*dt
                                    if k<N-1 else Constraint.Skip)

        # 优化函数
        
        model.stateobj = sum(sum((model.s[k,i]-self.s_r[k,i])**2 for i in RangeSet(0,s_dim-1)) for k in model.state_range)
        model.inputobj = sum(((model.tau_1[k])**2 + (model.tau_2[k])**2 + (model.tau_3[k])**2) for k in model.u_range)
        model.smoothobj = sum(((model.tau_1[k+1]-model.tau_1[k])**2 + (model.tau_2[k+1]-model.tau_2[k])**2 + (model.tau_3[k+1]-model.tau_3[k])**2) for k in RangeSet(0, N-3)) 
        model.sumobj = Objective(expr = 10000 * model.stateobj + 1 * model.inputobj + 10 * model.smoothobj, sense=minimize)
        
        self.model = model
        
    def Solve(self):
        
        SolverFactory('ipopt').solve(self.model)