import USV_ENV
import USV_ENV_reference
import numpy as np
import math
from gym.envs.classic_control import rendering
from MPC import MPC
from boat_transform import boat_transform
import matplotlib.pyplot as plt
import time
import matplotlib

# 设置全局字体样式为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 设置公式字体样式为 stix
plt.rcParams["mathtext.fontset"]="stix"

"""
动画渲染设置
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

# 输入力初始化
tau = np.array([0, 0, 0]).reshape(3,1)

# 环境重置
env.reset()
env_r.reset()


def simulation(state):
    '''
    参考轨迹模拟
    '''

    x, y, ψ, u, v, r = state

    # 消除累加误差
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)

    # [x, y, ψ]
    position = np.array([x, 
                         y, 
                         ψ]).reshape(3,1)

    # [u, v, r]
    velocity = np.array([u,
                         v,
                         r]).reshape(3,1)

    R_11 = math.cos(ψ)
    R_12 = -math.sin(ψ)
    R_21 = math.sin(ψ)
    R_22 = math.cos(ψ)

    R = np.array([[ R_11, R_12, 0   ],
                  [ R_21, R_22, 0   ],
                  [ 0   , 0   , 1   ]])

    # 计算位置

    position_dot = np.dot(R, velocity)

    position_dot[0] = 0.1*position[1]

    position_dot[1] = -0.1*position[0]

    position_dot[2] = -0.1

    position = 0.1*position_dot + position

    # 计算速度
    velocity = np.dot(R.transpose(), position_dot)

    # 由于pyomo优化求解器无法处理角度在±\pi处的突变,故不归一化角度(USV_ENV与USV_ENV_reference同理)
    # position[2] = np.where(position[2] > np.pi, position[2] - 2*np.pi, np.where(position[2] < -np.pi, position[2] + 2*np.pi, position[2]))

    state = np.array([position[0], position[1], position[2],velocity[0], velocity[1], velocity[2]]).flatten() 

    return state

tau_plot = np.zeros((3,300))
position_plot = np.zeros((3,300))
position_plot_r = np.zeros((3,300))

time_total = 0

for j in range(300):

    print(j)

    # 获取受控船舶当前状态
    s = env.state

    # 计算领航船舶从现在开始N个时间节点的参考轨迹
    s_r = np.zeros((9, 6))

    position_plot[0,j] = s[0]
    position_plot[1,j] = s[1]
    position_plot[2,j] = s[2]

    s_r[0] = env_r.state
    for i in range(8):
        s_r[i+1] = simulation(s_r[i])

    position_plot_r[0,j] = s_r[0,0]
    position_plot_r[1,j] = s_r[0,1]
    position_plot_r[2,j] = s_r[0,2]

    # 动画渲染
    # boat_transform(boat_trans, s)
    # boat_transform(boat_r_trans, s_r[0])
    # viewer.render()
       
    MPCsolver = MPC(s_0 = s, s_r = s_r)

    start = time.time()
    MPCsolver.Solve()
    end = time.time()
    time_step = end-start
    time_total += time_step

    tau = np.array([MPCsolver.model.tau_1[0](), MPCsolver.model.tau_2[0](), MPCsolver.model.tau_3[0]()]).reshape(3,1)

    tau_plot[0,j] = tau[0]
    tau_plot[1,j] = tau[1]
    tau_plot[2,j] = tau[2]

    # 施加控制力, 受控船舶步进
    env.step(tau)

    # 领航船舶步进
    env_r.step()

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



# 显示图像
plt.show()