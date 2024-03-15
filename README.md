# Robust-cooperative-control-of-multiple-USV
Enjun Liu  
$`\color{green}{liuenjun\_1010@163.com}`$  
## 协同编队策略  
基于领导跟随架构的USV集群编队，优化通信约束问题  
设USV集群中有N个单体，包括1个领导者和N-1个跟随者。领导者轨迹满足  
$$\dot{\eta}_0=K\eta+b$$
则如下分布式状态观测器可保证跟随者得到渐进收敛于领导者轨迹$`\eta`$的参考轨迹$`\eta_0`$  
$$\dot{\hat{\eta}}_i=K\hat{\eta}_i+b+k[\sum \limits ^{N-1} _{j=1} a _{ij} (\hat{\eta}_i-\hat{\eta}_j)+a _{i0}(\hat{\eta}_i-\eta_0)]$$  
distributed track.py 改进领导跟随法编队策略仿真  
<font face="逐浪新宋">我是逐浪新宋</font>

## 轨迹跟踪控制  
USV运动学与动力学公式：  
$$\dot{\eta}=J(\eta)(\upsilon)$$
$$\dot{\upsilon}=M^{-1}[-C(\upsilon)\upsilon-D(\upsilon)\upsilon-\sigma+\tau]$$

反步控制公式：


模型预测控制公式：

强化学习控制公式(SAC)：

毕设快结束再更新...
