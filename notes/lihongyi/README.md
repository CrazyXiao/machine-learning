## 李宏毅机器学习任务

**这里是Datawhale第7期组队学习之李宏毅机器学习的相关笔记，非常感谢Datawhale团队。**

#### **参考内容**

- [**李宏毅机器学习课程**](https://www.bilibili.com/video/av35932863?from=search&seid=2134843831238226258) 
- [Datawhale整理开源笔记《李宏毅机器学习》](<https://github.com/datawhalechina/Leeml-Book>)
- **《白话大数据与机器学习》** 
- **周志华《机器学习》** 
- **《统计学习方法》** 
- **《机器学习实战》** 
- **吴恩达机器学习教程**

### 第1-3天：线性回归任务一

**学习视频内容**

- 观看李宏毅课程内容：P1、P2。

- 视频地址：[**点我**](<https://www.bilibili.com/video/av35932863?from=search&seid=2134843831238226258>)

**学习打卡任务内容**

- 了解什么是Machine learning

- 学习中心极限定理，学习正态分布，学习最大似然估计

- 推导回归Loss function

- 学习损失函数与凸函数之间的关系

- 了解全局最优和局部最优

- 学习导数，泰勒展开

- 推导梯度下降公式

- 写出梯度下降的代码

- 学习L2-Norm，L1-Norm，L0-Norm

- 推导正则化公式

- 说明为什么用L1-Norm代替L0-Norm

- 学习为什么只对w/Θ做限制，不对b做限制

**我的打卡**

[线性回归任务一](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/lihongyi/day1-3.md>)

### 第4-7天：线性回归任务二

**学习视频内容：**

- **观看**李宏毅课程内容：P4、P5、P6、P7。

- 视频地址：[**点我**](<https://www.bilibili.com/video/av35932863?from=search&seid=8120828691691969718>)

**学习打卡内容：**

- 理解偏差和方差

- 学习误差为什么是偏差和方差而产生的，并且推导数学公式

- 过拟合，欠拟合，分别对应bias和variance什么情况

- 学习鞍点，复习上次任务学习的全局最优和局部最优

- 解决办法有哪些

- 梯度下降

- 学习Mini-Batch与SGD

- 学习Batch与Mini-Batch，SGD梯度下降的区别

- 如何根据样本大小选择哪个梯度下降(批量梯度下降，Mini-Batch）

- 写出SGD和Mini-Batch的代码

- 学习交叉验证

- 学习归一化 

- 学习回归模型评价指标

 **更多的对梯度下降优化将在《李宏毅深度学习》中会有学习任务介绍(指数加权平均，动量梯度下降等)**

**我的打卡**

[线性回归任务二](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/lihongyi/day4-7.md>)

### 第8-10天：线性回归任务三

**学习打卡内容：**

**大作业：** **预测PM2.5的值**

##### 要求

 1、要求python3.5+
 2、只能用（1）numpy（2）scipy（3）pandas
 3、请用梯度下降手写线性回归
 4、最好的公共简单基线

**我的打卡**

[代码](<https://github.com/CrazyXiao/machine-learning/tree/master/code/lihongyi/homework1>)

### 第11-13天：机器学习任务四

**学习视频内容：**

- 观看李宏毅课程内容：p8

- 视频连接：[**点我**](<https://www.bilibili.com/video/av35932863/?p=8>)

- 学习[Datawhale整理笔记](https://datawhalechina.github.io/Leeml-Book/#/chapter8/chapter8)

**学习打卡内容：**

- 从基础概率推导贝叶斯公式，朴素贝叶斯公式(1)
- 学习先验概率(2)
- 学习后验概率(3)
- 学习LR和linear regreeesion之间的区别(4)
- 推导sigmoid function公式(5)

##### 我的打卡

[机器学习任务四](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/lihongyi/day11-13.md>)

### 第14-16天：机器学习任务五

**学习视频内容：**

- 观看李宏毅课程内容：p8
- 视频连接：[**点我**](<https://www.bilibili.com/video/av35932863/?p=9>)
- 学习[Datawhale整理笔记](https://datawhalechina.github.io/Leeml-Book/#/chapter9/chapter9)

**学习打卡内容：**

- 推导LR损失函数(1)
- 学习LR梯度下降(2)
- 利用代码描述梯度下降(选做)(3)
- Softmax原理(4)
- softmax损失函数(5)
- softmax梯度下降(6)

##### 我的打卡

[机器学习任务五](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/lihongyi/day14-16.md>)

### 第17-19天：逻辑回归代码实现

##### 要求

- 学习LR学习算法的核心代码
- 写出详细的注释说明

##### 我的打卡

[机器学习任务六](<https://github.com/CrazyXiao/machine-learning/blob/master/code/lihongyi/task6.py>)

### 第20-22天：决策树

##### 学习目标

- 信息量计算，原理
- 信息熵
- 证明$0⩽H(p)⩽log_2n$
- 联合概率，边缘概率
- 联合熵，条件熵，条件熵公式推导
- 互信息，互信息公式推导
- 相对熵，交叉熵
- 回顾LR中的交叉熵
- 计算给定数据集中的香农熵

[机器学习任务七](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/lihongyi/day20-22.md>)

### 第23-26天：决策树任务二

**学习打卡内容：**

- 总结决策树模型结构

- 理解决策树递归思想

- 学习信息增益

- 学习信息增益率

- 学习ID3算法优缺点

- 学习C4.5算法优缺点

- 理解C4.5算法在ID3算法上有什么提升

- 学习C4.5算法在连续值上的处理

- 学习决策树如何生成

- 划分数据集代码

- 选择最好的数据集划分方式代码

- 创建树的函数代码

[机器学习任务八](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/lihongyi/day23-26.md>)

