"""
刘恩俊\n
USV仿真环境\n
"""

import numpy as np
import math

class USV():
    """
    USV仿真环境
    """

    def __init__(self):

        # 仿真时间步长
        self.integral_step = 0.1

        # 船舶动力学参数
        self.m = 23.80
        self.I_z = 1.76
        
        self.X_g = 0.05
        self.X_u = -0.72
        self.X_u_dot = -2.00
        self.X_uu = -1.33
        self.X_uuu = -5.8664

        self.Y_v = -0.86
        self.Y_v_dot = -10.00
        self.Y_r = 0.11
        self.Y_r_dot = -0.00
        self.Y_vv = -36.28
        self.Y_vr = -0.85
        self.Y_rv = -0.81
        self.Y_rr = -3.45

        self.N_v = 0.11
        self.N_r = -1.90
        self.N_r_dot = -1.00
        self.N_vv = 5.04
        self.N_vr = 0.08
        self.N_rv = 0.13
        self.N_rr = -0.75

        self.state = None

        self.return_1 = None
        self.return_2 = None

        M_11 = self.m - self.X_u_dot
        M_22 = self.m - self.Y_v_dot
        M_33 = self.I_z - self.N_r_dot
        M_23 = self.m*self.X_g - self.Y_r_dot
        M_32 = self.m*self.X_g - self.Y_r_dot

        self.M = np.array([[M_11, 0   , 0   ],
                           [0   , M_22, M_23],
                           [0   , M_32, M_33]])


    def step(self, action):
        """
        仿真执行
        """

        x, y, psi, u, v, r = self.state

        # [x, y, psi]
        position = np.array([x, 
                             y, 
                             psi]).reshape(3,1)

        # [u, v, r]
        velocity = np.array([u,
                             v,
                             r]).reshape(3,1)
        
        M_11 = self.m - self.X_u_dot
        M_22 = self.m - self.Y_v_dot
        M_23 = self.m*self.X_g - self.Y_r_dot

        C_13 = -M_22*v - M_23*r
        C_23 = M_11*u

        D_11 = -self.X_u - self.X_uu*abs(u) -  self.X_uuu*pow(u,2)
        D_22 = -self.Y_v - self.Y_vv*abs(v) - self.Y_rv*abs(r)
        D_33 = -self.N_r - self.N_vr*abs(v) - self.N_rr*abs(r)
        D_23 = -self.Y_r - self.Y_vr*abs(v) - self.Y_rr*abs(r)
        D_32 = -self.N_v - self.N_vv*abs(v) - self.N_rv*abs(r)

        R_11 = math.cos(psi)
        R_12 = -math.sin(psi)
        R_21 = math.sin(psi)
        R_22 = math.cos(psi)

        C = np.array([[ 0   , 0   , C_13],
                      [ 0   , 0   , C_23],
                      [-C_13,-C_23, 0   ]])
        
        D = np.array([[ D_11, 0   , 0   ],
                      [ 0   , D_22, D_23],
                      [ 0   , D_32, D_33]])      

        R = np.array([[ R_11, R_12, 0   ],
                      [ R_21, R_22, 0   ],
                      [ 0   , 0   , 1   ]])
        
        G = np.array([0,
                      0,
                      0]).reshape(3,1)
        
        M_inv = np.linalg.inv(self.M)
        
        position_dot = np.dot(R, velocity)

        velocity_dot = np.dot(M_inv, (action - G - np.dot(C+D, velocity)))

        self.return_1 = np.dot(C+D, velocity)
        
        self.return_2 = velocity_dot

        # 计算速度
        velocity = self.integral_step*velocity_dot + velocity

        # 计算位置
        position = self.integral_step*position_dot + position

        # 将psi归化至±pi之间
        position[2] = np.where(position[2] > np.pi, position[2] - 2*np.pi, np.where(position[2] < -np.pi, position[2] + 2*np.pi, position[2]))

        self.state = np.array([position[0], position[1], position[2],velocity[0], velocity[1], velocity[2]]).flatten()

        return self.state

    def reset(self):
        """
        仿真重置
        """

        # 初始化船舶位置与航向角
        x = -0.2
        y = 0.8
        psi = 0
        u = 0
        v = 0
        r = 0

        self.state = np.array([x, y, psi, u, v, r])
        state = self.state

        self.return_1 = 0.
        self.return_2 = 0.

        return state