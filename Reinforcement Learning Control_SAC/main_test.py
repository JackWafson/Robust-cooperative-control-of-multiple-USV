import warnings
warnings.filterwarnings('ignore')

import USV_ENV
import USV_ENV_reference
import numpy as np
from gym.envs.classic_control import rendering
from boat_transform import boat_transform
import torch
from time import sleep
import time


import torch.nn.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体样式为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 设置公式字体样式为 stix
plt.rcParams["mathtext.fontset"]="stix"

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样函数
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)  # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * 10
        return action, log_prob

actor = PolicyNet(state_dim=6, hidden_dim=256, action_dim=3)

file_path = "C:/model.pth"

actor.load_state_dict(torch.load(file_path))


"""
渲染设置
"""

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

"""
环境初始化
"""

# 动力学环境初始化
env = USV_ENV.USV()
env_r = USV_ENV_reference.USV()

# 每个回合开始,环境重置
env.reset()
env_r.reset()

total_reward = 0
w_last = 0.

tau_plot = np.zeros((3,300))
position_plot = np.zeros((3,300))
position_plot_r = np.zeros((3,300))
time_total = 0
for i in range(int(300)):

    # 获取初始状态
    s = env.state
    s_r = env_r.state

    position_plot[0,i] = s[0]
    position_plot[1,i] = s[1]
    position_plot[2,i] = s[2]

    position_plot_r[0,i] = s_r[0]
    position_plot_r[1,i] = s_r[1]
    position_plot_r[2,i] = s_r[2]

    # 训练过程渲染
    # boat_transform(boat_trans, s)
    # boat_transform(boat_r_trans, s_r)
    # viewer.render()

    state = s - s_r
    state = torch.tensor([state], dtype=torch.float)
    
    start = time.time()
    action = actor(state)[0]
    end = time.time()
    time_step = end-start
    time_total += time_step

    tau = np.array([action[0][0].item(), action[0][1].item(), action[0][2].item()]).reshape(3,1)

    tau_plot[0,i] = tau[0]
    tau_plot[1,i] = tau[1]
    tau_plot[2,i] = tau[2]

    # 施加控制力, 受控船舶步进
    s_ = env.step(np.array(tau).reshape(3, 1))

    # 领航船舶步进
    s_r_ = env_r.step()

    next_state = s_ - s_r_

    reward = -sum((next_state[k])**2 for k in range(3))

    total_reward += reward
print("time_total:",time_total)

num_points = len(tau_plot[0])  # 获取数据点数量
time_step = 0.1  # 每个点代表的时间间隔（秒）
time_points = np.arange(0, num_points * time_step, time_step)  # 生成时间点序列
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# 绘制每行数据

axs[0].plot(time_points, tau_plot[0, :], label=r'$\tau_1$', color='green')
axs[0].legend(loc='lower right',fontsize=16)


axs[1].plot(time_points, tau_plot[1, :], label=r'$\tau_2$', color='green')
axs[1].legend(loc='lower right',fontsize=16)

axs[2].plot(time_points, tau_plot[2, :], label=r'$\tau_3$', color='green')
axs[2].legend(loc='lower right', fontsize=16)
axs[2].set_xlabel('$t/s$', fontsize=18)

for ax in axs:
    ax.tick_params(axis='x', labelsize=14)  # 设置横坐标轴刻度字体大小为12
    ax.tick_params(axis='y', labelsize=14)  # 设置纵坐标轴刻度字体大小为12

# 创建子图
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

axs[0].plot(time_points, position_plot_r[0, :], label='$x_r$', color='blue')
axs[0].plot(time_points, position_plot[0, :], label='$x$', color='red', linestyle='--')
axs[0].legend(loc='lower right', fontsize=16)


axs[1].plot(time_points, position_plot_r[1, :], label='$y_r$', color='blue')
axs[1].plot(time_points, position_plot[1, :], label='$y$', color='red', linestyle='--')
axs[1].legend(loc='lower right', fontsize=16)


axs[2].plot(time_points, position_plot_r[2, :], label='$\psi_r$', color='blue')
axs[2].plot(time_points, position_plot[2, :], label='$\psi$', color='red', linestyle='--')
axs[2].legend(loc='lower right', fontsize=16)
axs[2].set_xlabel('$t/s$', fontsize=18)

for ax in axs:
    ax.tick_params(axis='x', labelsize=14)  # 设置横坐标轴刻度字体大小为12
    ax.tick_params(axis='y', labelsize=14)  # 设置纵坐标轴刻度字体大小为12

# 显示图像
plt.show()