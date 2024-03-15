import math
import numpy as np

K_1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

K_2 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

def BC(w_last, env_r, env, t, x, y):
    '''反步控制策略'''

    s_r = env_r.state    
    s = env.state

    z_1_1 = s[0] - s_r[0] - x*np.clip(math.sin(0.1*t),-1, 1)
    z_1_2 = s[1] - s_r[1] - y*np.clip(math.cos(0.1*t),-1, 1)
    z_1_3 = s[2] - s_r[2]
    if z_1_3 > np.pi:
        z_1_3 = z_1_3 - 2 * np.pi
    elif z_1_3 < -np.pi:
        z_1_3 = z_1_3 + 2 * np.pi

    z_1 = np.array([z_1_1, z_1_2, z_1_3]).reshape(3,1)

    J_T = np.array([
    [ math.cos(s[2]), math.sin(s[2]), 0],
    [-math.sin(s[2]), math.cos(s[2]), 0],
    [0                , 0           , 1]])

    w = np.dot(J_T, (env_r.position_dot - np.dot(K_1, z_1)))

    w_dot = (w - w_last)/env.integral_step

    w_last = w

    z_2 = np.array([s[3], s[4], s[5]]).reshape(3,1) - w
    z_2_Norm = np.linalg.norm(z_2, 2)

    cdv = env.return_1

    M = env.M

    tau = cdv + np.dot(M, w_dot) - np.dot(K_2, z_2) - 2*np.sign(z_2)

    return tau, w_last