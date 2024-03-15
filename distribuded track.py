import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot

# 设置全局字体样式为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 设置公式字体样式为 stix
plt.rcParams["mathtext.fontset"]="stix"

# A矩阵设置
A = np.array([[0, -2],
              [2, 0]])

K = np.array([[-5, 0],
              [0, -5]])

x_0 = np.array([[1],
                [0]])

x_1 = np.array([[0.8],
                [0.2]])

x_2 = np.array([[1.1],
                [-0.1]])

x_3 = np.array([[0.6],
                [0.3]])

x_4 = np.array([[0.8],
                [-0.4]])

x_5 = np.array([[1.2],
                [0.3]])

trajectory_0 = [x_0.flatten()]
trajectory_1 = [x_1.flatten()]
trajectory_2 = [x_2.flatten()]
trajectory_3 = [x_3.flatten()]
trajectory_4 = [x_4.flatten()]
trajectory_5 = [x_5.flatten()]

for t in range(2500):

    x_0_dots = np.dot(A, x_0)
    x_0 = x_0 + 0.001 * x_0_dots
    trajectory_0.append(x_0.flatten())

    x_1_dots = np.dot(A, x_1)+np.dot(K, x_1-x_2)
    x_1 = x_1 + 0.001 * x_1_dots
    trajectory_1.append(x_1.flatten())

    x_2_dots = np.dot(A, x_2)+np.dot(K, x_2-x_4)
    x_2 = x_2 + 0.001 * x_2_dots
    trajectory_2.append(x_2.flatten())

    x_3_dots = np.dot(A, x_3)+np.dot(K, x_3-x_0)
    x_3 = x_3 + 0.001 * x_3_dots
    trajectory_3.append(x_3.flatten())

    x_4_dots = np.dot(A, x_4)+np.dot(K, x_4-x_3)
    x_4 = x_4 + 0.001 * x_4_dots
    trajectory_4.append(x_4.flatten())
    
    x_5_dots = np.dot(A, x_5)+np.dot(K, x_5-x_2)
    x_5 = x_5 + 0.001 * x_5_dots
    trajectory_5.append(x_5.flatten())


trajectory_0 = np.array(trajectory_0)
trajectory_1 = np.array(trajectory_1)
trajectory_2 = np.array(trajectory_2)
trajectory_3 = np.array(trajectory_3)
trajectory_4 = np.array(trajectory_4)
trajectory_5 = np.array(trajectory_5)
plt.plot(trajectory_0[:, 0], trajectory_0[:, 1], label='USV 0', color='red')
plt.plot(trajectory_1[:, 0], trajectory_1[:, 1], linestyle='--', label='USV 1', color='blue')
plt.plot(trajectory_2[:, 0], trajectory_2[:, 1], linestyle='--', label='USV 2', color='green')
plt.plot(trajectory_3[:, 0], trajectory_3[:, 1], linestyle='--', label='USV 3', color='black')
plt.plot(trajectory_4[:, 0], trajectory_4[:, 1], linestyle='--', label='USV 4', color='orange')
plt.plot(trajectory_5[:, 0], trajectory_5[:, 1], linestyle='--', label='USV 5', color='brown')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(trajectory_0[0, 0], trajectory_0[0, 1], marker='*', s=100, color='red')
plt.scatter(trajectory_1[0, 0], trajectory_1[0, 1], marker='*', s=100, color='blue')
plt.scatter(trajectory_2[0, 0], trajectory_2[0, 1], marker='*', s=100, color='green')
plt.scatter(trajectory_3[0, 0], trajectory_3[0, 1], marker='*', s=100, color='black')
plt.scatter(trajectory_4[0, 0], trajectory_4[0, 1], marker='*', s=100, color='orange')
plt.scatter(trajectory_5[0, 0], trajectory_5[0, 1], marker='*', s=100, color='brown')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(12, 4))

num_points = len(trajectory_0)  # 获取数据点数量
time_step = 0.001  # 每个点代表的时间间隔（秒）
time_points = np.arange(0, num_points * time_step, time_step)  # 生成时间点序列

plt.plot(time_points, trajectory_0[:, 0], label='USV 0', color='red')
plt.plot(time_points, trajectory_1[:, 0], linestyle='--', label='USV 1', color='blue')
plt.plot(time_points, trajectory_2[:, 0], linestyle='--', label='USV 2', color='green')
plt.plot(time_points, trajectory_3[:, 0], linestyle='--', label='USV 3', color='black')
plt.plot(time_points, trajectory_4[:, 0], linestyle='--', label='USV 4', color='orange')
plt.plot(time_points, trajectory_5[:, 0], linestyle='--', label='USV 5', color='brown')
plt.xlabel('$t/s$', fontsize=14)
plt.ylabel('$x$', fontsize=14)
plt.legend(loc='upper right')
plt.xlim(0, max(time_points))
plt.show()



plt.figure(figsize=(12, 4))

plt.plot(time_points, trajectory_0[:, 1], label='USV 0', color='red')
plt.plot(time_points, trajectory_1[:, 1], linestyle='--', label='USV 1', color='blue')
plt.plot(time_points, trajectory_2[:, 1], linestyle='--', label='USV 2', color='green')
plt.plot(time_points, trajectory_3[:, 1], linestyle='--', label='USV 3', color='black')
plt.plot(time_points, trajectory_4[:, 1], linestyle='--', label='USV 4', color='orange')
plt.plot(time_points, trajectory_5[:, 1], linestyle='--', label='USV 5', color='brown')
plt.xlabel('$t/s$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.legend(loc='upper right')
plt.xlim(0, max(time_points))
plt.show()
