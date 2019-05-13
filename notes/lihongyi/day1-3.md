## 李宏毅机器学习Day1-3：线性回归任务一

### 什么是机器学习？

机器学习的核心是“**使用算法解析数据，从中学习，然后对世界上的某件事情做出决定或预测**”。

Tom Mitchell提供了一个更现代的定义：“据说计算机程序从经验E中学习某些任务T和绩效测量P，如果它在T中的任务中的表现，由P测量，随经验E而改善。 “

示例：玩跳棋。

E =玩许多跳棋游戏的经验

T =玩跳棋的任务。

P =程序赢得下一场比赛的概率。

### 中心极限定理

中心极限定理指的是给定一个任意分布的总体。我每次从这些总体中随机抽取 n 个抽样，一共抽 m 次。 然后把这 m 组抽样分别求出平均值。 这些平均值的分布接近**正态分布**。



### 推导回归Loss Function

$$
\begin{aligned}  
L(f) & = \sum_{i=1}^{m}\left ( y - y^i \right )^2，其中y = f(x) =  b + w·x,代入得到 \\
& = \sum_{i=1}^{m}\left ( b + w·x^i - y^i \right )^2\\
\end{aligned}
$$

其中 $（x^i，y^i）$为 样本点，m为样本数，$y$为模型函数。

### 损失函数与凸函数

在使用梯度下降进行最小化损失函数的时候，如果损失函数是凸函数，那么不管怎么初始化，总能找到全局最优解。否则，很有可能陷入局部最优解。

### 推导梯度下降

梯度下降算法如下：

$θ_j:=θ_j−α\frac∂{∂θ_j}L(θ)$

![img](https://upload-images.jianshu.io/upload_images/3850035-7333735bfd8bfc97.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/774/format/webp)

![img](https://upload-images.jianshu.io/upload_images/3850035-6c5244dfeab39681.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/888/format/webp)

### 梯度下降代码

[代码](<https://github.com/CrazyXiao/machine-learning/blob/master/code/linear_regression/linear_regession_simple.py>)

### 推导正则化公式

### 为什么用L1-Norm代替L0-Norm

$L0$是指向量中非0的元素的个数。如果我们用$L0​$范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0。换句话说，让参数W是稀疏的。

不幸的是，L0范数的最优化问题是一个NP hard问题，而且理论上有证明，L1范数是L0范数的最优凸近似，因此通常使用L1范数来代替。

### 为什么只对w/Θ做限制，不对b做限制

b的变化只对函数的位置有影响，并不改变函数的平滑性；相反，对w的限制可以实现对特征的惩罚，留取更重要的特征，惩罚不重要的特征权重，从而使loss func更平滑，提高泛化能力，防止过拟合。