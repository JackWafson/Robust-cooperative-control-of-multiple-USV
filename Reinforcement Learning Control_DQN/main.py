import warnings
warnings.filterwarnings('ignore')

import USV_ENV
import USV_ENV_reference
import numpy as np
from gym.envs.classic_control import rendering
from boat_transform import boat_transform
import torch
from DQN import DQN
from time import sleep

"""
渲染设置
"""

screen_width = 500 # 窗口宽度
screen_height = 500 # 窗口高度
viewer = rendering.Viewer(screen_width, screen_height)

boat_width = 8 # 船舶宽度
boat_height = 12 # 船舶长度

l, r, t, b = -boat_width/2, boat_width/2, boat_height, 0

# 受控船舶渲染 
boat = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
boat_trans = rendering.Transform()
boat.add_attr(boat_trans)
boat.set_color(0, 0, 1) # 蓝色
viewer.add_geom(boat)

# 领航船舶渲染 
boat_r = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
boat_r_trans = rendering.Transform()
boat_r.add_attr(boat_r_trans)
boat_r.set_color(1, 0, 0) # 红色
viewer.add_geom(boat_r)

"""
环境初始化
"""

# 动力学环境初始化
env = USV_ENV.USV()
env_r = USV_ENV_reference.USV()

print(torch.device('cuda'))

DQN = DQN(state_dim=6, action_dim=165, device=torch.device('cuda'))

tau_1 = [10, 5, 1, 0.1, 0.08, 0, -0.1, -0.08, -1, -5, -10]
tau_2 = [0]
tau_3 = [-2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2]

action_space = np.zeros((165, 3))

# 填充动作空间
index = 0
for t1 in tau_1:
    for t2 in tau_2:
        for t3 in tau_3:
            action_space[index] = [t1, t2, t3]
            index += 1


for episode in range(int(1e6)):

    # 每个回合开始,环境重置
    env.reset()
    env_r.reset()

    total_reward = 0

    for t in range(int(1e3)):

        # 获取初始状态
        s = env.state
        s_r = env_r.state

        # # 训练过程渲染
        # boat_transform(boat_trans, s)
        # boat_transform(boat_r_trans, s_r)
        # viewer.render()

        state = s - s_r

        action = DQN.select_action(state)

        tau = action_space[action].reshape(3,1)

        # 受控船舶步进
        s_ = env.step(np.array(tau).reshape(3, 1))

        # 领航船舶步进
        s_r_ = env_r.step()

        next_state = s_ - s_r_

        reward = -sum((next_state[k])**2 for k in range(3))

        total_reward += reward

        DQN.buffer.add(state, action, reward, next_state)

        if DQN.buffer.size() > 2e3:
            DQN.update()

    print(episode,':',total_reward)