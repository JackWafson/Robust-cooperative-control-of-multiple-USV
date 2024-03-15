import USV_ENV
import USV_ENV_1
import USV_ENV_2
import USV_ENV_3
import USV_ENV_reference
import numpy as np
from gym.envs.classic_control import rendering
from BC import BC
from boat_transform import boat_transform
from time import sleep
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

# 设置全局字体样式为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 设置公式字体样式为 stix
plt.rcParams["mathtext.fontset"]="stix"

font_TimesNewsman = FontProperties(family='Times New Roman', size =8)


"""
渲染设置
"""

# screen_width = 500 # 窗口宽度
# screen_height = 500 # 窗口高度
# viewer = rendering.Viewer(screen_width, screen_height)

# boat_width = 8 # 船舶宽度
# boat_height = 12 # 船舶长度

# l, r, t, b = -boat_width/2, boat_width/2, boat_height, 0

# 受控船舶渲染 
# boat = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
# boat_trans = rendering.Transform()
# boat.add_attr(boat_trans)
# boat.set_color(0, 0, 1) # 蓝色
# viewer.add_geom(boat)

# # 受控船舶1渲染 
# boat_1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
# boat_trans_1 = rendering.Transform()
# boat_1.add_attr(boat_trans_1)
# boat_1.set_color(0, 0, 1) # 蓝色
# viewer.add_geom(boat_1)

# # 受控船舶2渲染 
# boat_2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
# boat_trans_2 = rendering.Transform()
# boat_2.add_attr(boat_trans_2)
# boat_2.set_color(0, 0, 1) # 蓝色
# viewer.add_geom(boat_2)

# # 受控船舶3渲染 
# boat_3 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
# boat_trans_3 = rendering.Transform()
# boat_3.add_attr(boat_trans_3)
# boat_3.set_color(0, 0, 1) # 蓝色
# viewer.add_geom(boat_3)

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
env_1 = USV_ENV_1.USV()
env_2 = USV_ENV_2.USV()
env_3 = USV_ENV_3.USV()

env_r = USV_ENV_reference.USV()

# 输入力初始化
tau = np.array([0, 0, 0]).reshape(3,1)
tau_1 = np.array([0, 0, 0]).reshape(3,1)
tau_2 = np.array([0, 0, 0]).reshape(3,1)
tau_3 = np.array([0, 0, 0]).reshape(3,1)

# 每个回合开始,环境重置
env.reset()
env_1.reset()
env_2.reset()
env_3.reset()

env_r.reset()

# w计算参数初始化
w_last = 0.
w_last_1 = 0.
w_last_2 = 0.
w_last_3 = 0.

t = 0
time_total = 0

tau_plot = np.zeros((3,300))
position_plot = np.zeros((3,300))
position_plot_1 = np.zeros((3,300))
position_plot_2 = np.zeros((3,300))
position_plot_3 = np.zeros((3,300))
position_plot_r = np.zeros((3,300))

for i in range(300):

    t += 1

    # 获取初始状态
    s = env.state
    s_1 = env_1.state
    s_2 = env_2.state
    s_3 = env_3.state
    s_r = env_r.state

    position_plot[0,i] = s[0]
    position_plot[1,i] = s[1]
    position_plot[2,i] = s[2]

    position_plot_1[0,i] = s_1[0]
    position_plot_1[1,i] = s_1[1]
    position_plot_1[2,i] = s_1[2]

    position_plot_2[0,i] = s_2[0]
    position_plot_2[1,i] = s_2[1]
    position_plot_2[2,i] = s_2[2]

    position_plot_3[0,i] = s_3[0]
    position_plot_3[1,i] = s_3[1]
    position_plot_3[2,i] = s_3[2]

    position_plot_r[0,i] = s_r[0]
    position_plot_r[1,i] = s_r[1]
    position_plot_r[2,i] = s_r[2]

    # 训练过程渲染
    # boat_transform(boat_trans, s)
    # boat_transform(boat_trans_1, s_1)
    # boat_transform(boat_trans_2, s_2)
    # boat_transform(boat_trans_3, s_3)
    # boat_transform(boat_r_trans, s_r)
    # viewer.render()
        
    # 反步控制律ub
    start = time.time()
    tau, w_last = BC(w_last, env_r, env, t, 0, 0)
    end = time.time()
    time_step = end-start
    tau_1, w_last_1 = BC(w_last_1, env_r, env_1, t, 0.2, -0.2)
    tau_2, w_last_2 = BC(w_last_2, env_r, env_2, t, -0.2, -0.2)
    tau_3, w_last_3 = BC(w_last_3, env_r, env_3, t, -0.2, 0.2)

    tau_plot[0,i] = tau[0]
    tau_plot[1,i] = tau[1]
    tau_plot[2,i] = tau[2]

    # tau_f = tau + np.random.rand() # 随机干扰建模

    # 施加控制力, 受控船舶步进
    env.step(tau)
    env_1.step(tau_1)
    env_2.step(tau_2)
    env_3.step(tau_3)

    # 领航船舶步进
    env_r.step()

    time_total+=time_step

print("time_total:",time_total)
plt.figure()
plt.plot(position_plot_r[0], position_plot_r[1], color='red', label='USV 0')
plt.plot(position_plot[0], position_plot[1], color='blue', linestyle='--', label='USV 1')
plt.plot(position_plot_1[0], position_plot_1[1], color='green', linestyle='--', label='USV 2')
plt.plot(position_plot_2[0], position_plot_2[1], color='black', linestyle='--', label='USV 3')
plt.plot(position_plot_3[0], position_plot_3[1], color='orange', linestyle='--', label='USV 4')

plt.scatter(position_plot_r[0, 0], position_plot_r[1, 0], color='red', marker='*', s=100)
plt.scatter(position_plot[0, 0], position_plot[1, 0], color='blue', marker='*', s=100)
plt.scatter(position_plot_1[0, 0], position_plot_1[1, 0], color='green', marker='*', s=100)
plt.scatter(position_plot_2[0, 0], position_plot_2[1, 0], color='black', marker='*', s=100)
plt.scatter(position_plot_3[0, 0], position_plot_3[1, 0], color='orange', marker='*', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='lower right')
plt.show()

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
axs[2].legend(loc='lower right',fontsize=16)
axs[2].set_xlabel('$t/s$',fontsize=18)

for ax in axs:
    ax.tick_params(axis='x', labelsize=14)  # 设置横坐标轴刻度字体大小为12
    ax.tick_params(axis='y', labelsize=14)  # 设置纵坐标轴刻度字体大小为12


# 创建子图
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

axs[0].plot(time_points, position_plot_r[0, :], label='$x_r$', color='blue')
axs[0].plot(time_points, position_plot[0, :], label='$x$', color='red', linestyle='--')
axs[0].legend(loc='lower right',fontsize=16)


axs[1].plot(time_points, position_plot_r[1, :], label='$y_r$', color='blue')
axs[1].plot(time_points, position_plot[1, :], label='$y$', color='red', linestyle='--')
axs[1].legend(loc='lower right',fontsize=16)


axs[2].plot(time_points, position_plot_r[2, :], label='$\psi_r$', color='blue')
axs[2].plot(time_points, position_plot[2, :], label='$\psi$', color='red', linestyle='--')
axs[2].legend(loc='lower right',fontsize=16)
axs[2].set_xlabel('$t/s$',fontsize=18)

for ax in axs:
    ax.tick_params(axis='x', labelsize=14)  # 设置横坐标轴刻度字体大小为12
    ax.tick_params(axis='y', labelsize=14)  # 设置纵坐标轴刻度字体大小为12

# 显示图像
plt.show()