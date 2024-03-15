"""
刘恩俊\n
USV参考轨迹仿真环境\n
"""

import numpy as np

class USV():
    """USV参考轨迹仿真环境"""

    def __init__(self):

        # 仿真时间步长
        self.integral_step = 0.1

        self.state = None

        self.position_dot = np.array([0, 0, 0]).reshape(3, 1)


    def step(self):
        """仿真执行"""

        x, y, psi, u, v, r = self.state

        # 消除累加误差
        x = np.clip(x, -1.0, 1.0)
        y = np.clip(y, -1.0, 1.0)

        # [x, y, psi]
        position = np.array([x, 
                             y, 
                             psi]).reshape(3,1)


        # [u, v, r]
        velocity = np.array([u,
                             v,
                             r]).reshape(3,1)

        R_11 = np.math.cos(psi)
        R_12 = -np.math.sin(psi)
        R_21 = np.math.sin(psi)
        R_22 = np.math.cos(psi)
    

        R = np.array([[ R_11, R_12, 0   ],
                      [ R_21, R_22, 0   ],
                      [ 0   , 0   , 1   ]])

        # 计算位置

        position_dot = np.dot(R, velocity)

        position_dot[0] = 0.1*position[1]

        position_dot[1] = -0.1*position[0]

        position_dot[2] = -0.1

        self.position_dot = position_dot

        position = self.integral_step*position_dot + position

        # 计算速度

        velocity = np.dot(R.transpose(), position_dot)

        # 将psi归化至±pi之间
        # position[2] = np.where(position[2] > np.pi, position[2] - 2*np.pi, np.where(position[2] < -np.pi, position[2] + 2*np.pi, position[2]))

        self.state = np.array([position[0], position[1], position[2],velocity[0], velocity[1], velocity[2]]).flatten() 

        return self.state

    def reset(self):
        """仿真重置"""

        # 初始化船舶位置与航向角
        x = 0
        y = 1
        psi = 0
        u = 0
        v = 0
        r = 0

        self.state = np.array([x, y, psi, u, v, r])
        state = self.state

        return state