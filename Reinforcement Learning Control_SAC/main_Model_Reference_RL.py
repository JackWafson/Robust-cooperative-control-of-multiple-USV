import warnings
warnings.filterwarnings('ignore')

import USV_ENV
import USV_ENV_reference
import numpy as np
from gym.envs.classic_control import rendering
from boat_transform import boat_transform
import torch
from time import sleep
from SAC import SAC
from BC import BC

# screen_width = 500 # 窗口宽度
# screen_height = 500 # 窗口高度
# viewer = rendering.Viewer(screen_width, screen_height)

# boat_width = 8 # 船舶宽度
# boat_height = 12 # 船舶长度

# l, r, t, b = -boat_width/2, boat_width/2, boat_height, 0

# # 受控船舶渲染 
# boat = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
# boat_trans = rendering.Transform()
# boat.add_attr(boat_trans)
# boat.set_color(0, 0, 1) # 蓝色
# viewer.add_geom(boat)

# # 领航船舶渲染 
# boat_r = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
# boat_r_trans = rendering.Transform()
# boat_r.add_attr(boat_r_trans)
# boat_r.set_color(1, 0, 0) # 红色
# viewer.add_geom(boat_r)

# 动力学环境初始化
env = USV_ENV.USV()
env_r = USV_ENV_reference.USV()

# SAC算法初始化
SAC = SAC(state_dim=9, action_dim=3, hidden_dim=64,
          actor_lr=1e-4, critic_lr=1e-4, alpha_lr=1e-4, target_entropy=-1, tau=0.005, gamma=0.99, device=torch.device("cuda"))
training = False

for episode in range(int(1e6)):
    # 每个回合开始,环境重置
    env.reset()
    env_r.reset()
    # 参数初始化
    total_reward = 0
    w_last = 0.
    for t in range(int(1e3)):

        if not training:# 数据收集阶段,应用反步控制策略采集数据
            
            # 获取初始状态
            s = env.state
            s_r = env_r.state

            # 训练过程渲染
            # boat_transform(boat_trans, s)
            # boat_transform(boat_r_trans, s_r)
            # viewer.render()

            tau_bc, w_last = BC(w_last, env_r, env, t)
            state = s - s_r

            s_ = env.step(tau_bc)# 施加控制力, 受控船舶步进
            s_r_ = env_r.step()# 领航船舶步进

            state_next = s_ - s_r_
            tau_bc_next, _ = BC(w_last, env_r, env, t)

            reward = -sum((state_next[k])**2 for k in range(3))-sum((tau_bc[k])**2 for k in range(3))/100000000
            total_reward += reward

            State = np.concatenate((10000*state, tau_bc.flatten()))
            State_next = np.concatenate((10000*state_next, tau_bc_next.flatten()))

            SAC.buffer.add(State, np.array([0,0,0]), reward, State_next, 0)

        if training:# 训练阶段,应用强化控制策略采集数据并训练网络

            SAC.update()

            s = env.state
            s_r = env_r.state

            # 训练过程渲染
            # boat_transform(boat_trans, s)
            # boat_transform(boat_r_trans, s_r)
            # viewer.render()

            state = s - s_r
            tau_bc, w_last = BC(w_last, env_r, env, t)
            State = np.concatenate((10000*state, tau_bc.flatten()))

            action = SAC.take_action(State)
            tau = np.array([action[0][0].item(), action[0][1].item(), action[0][2].item()]).reshape(3,1)
            
            s_ = env.step(tau+tau_bc)
            s_r_ = env_r.step()

            state_next = s_ - s_r_
            tau_bc_next, _ = BC(w_last, env_r, env, t)
            
            State_next = np.concatenate((10000*state_next, tau_bc_next.flatten()))


            x=tau+tau_bc
            reward = -sum((state_next[k])**2 for k in range(3))-sum((x[k])**2 for k in range(3))/100000000
            total_reward += reward

            SAC.buffer.add(State, tau.flatten(), reward, State_next, 0)
        
        if SAC.buffer.__len__() > 5e3:
            training  = True

    print(episode,':',total_reward)
    if training and total_reward>-0.1:
        print('Training finished!')
        #SAC.save()
        break