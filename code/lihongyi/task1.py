"""
    自己实现
    梯度下降解决线性回归问题
"""



import numpy as np
import matplotlib.pyplot as plt


def costFunction(X, y, theta=[0, 0]):
    """
        损失函数
    """
    m = y.size
    h = X.dot(theta)
    J = 1.0 / (2 * m) * (np.sum(np.square(h - y)))
    return J

def gradientDescent(X, y, theta=[0, 0], alpha=0.01, num_iters=1500):
    """
        梯度下降
    """
    m = y.size
    J_history = np.zeros(num_iters)
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1.0 / m) * (X.T.dot(h - y))
        J_history[iter] = costFunction(X, y, theta)
    return (theta, J_history)


def MaxMinNormalization(x):
    """
        归一化
    """
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x

# 使用外部训练集
# data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
# X = np.c_[np.ones(data.shape[0]),data[:,0]]
# y = data[:,1]

# 自己构造数据集
X_row = 100 * np.random.rand(100)
X = MaxMinNormalization(X_row)
y = 0.5*X + 2 + np.random.normal(0,0.01,(100,))

# 数据可视化
plt.subplot(1, 2, 1)
plt.scatter(X_row, y, color='black')
plt.xlabel('x')
plt.ylabel('y')


X = np.c_[np.ones((X.shape[0],1)), X]


# training set
X_train = X[:80]
y_train = y[:80]
# test set
X_test = X[80:]
y_test = y[80:]



print(costFunction(X,y))

b = 0
w = 0
lr = 0.01
iteration = 10000

# 画出每一次迭代和损失函数变化
theta , Cost_J = gradientDescent(X_train, y_train, theta=[b, w], alpha= lr, num_iters= iteration)

print('最终b, w结果: ',theta)
testCost = costFunction(X_test, y_test, theta)
print('测试集误差: ',testCost)

h = X.dot(theta)
plt.plot(X_row, h, "b--")
plt.subplot(1, 2, 2)
plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()
