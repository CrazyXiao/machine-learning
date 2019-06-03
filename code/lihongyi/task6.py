"""
    自己实现逻辑回归
    基于python3
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score # 用于计算准备率，可以自己实现


def sigmoid(z):
    """ sigmoid函数
    """
    return(1 / (1 + np.exp(-z)))


def costFunction(theta, X, y):
    """ 损失函数
    """
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
    if np.isnan(J):
        return (np.inf)
    return J


def gradient(theta, X, y):
    """ 求解梯度
    """
    m = y.size
    h = sigmoid(X.dot(theta))
    grad = (1.0 / m) * X.T.dot(h - y)
    return (grad)

def predict(theta, X, threshold=0.5):
    """ 预测
    """
    p = sigmoid(X.dot(theta)) >= threshold
    return (p.astype('int'))

def loaddata(file, delimeter):
    """
        加载数据
    """
    data = np.loadtxt(file, delimiter=delimeter)
    return data


def gradientDescent(X, y, theta, alpha=0.01, num_iters=1500):
    """
        梯度下降
    """
    J_history = np.zeros(num_iters)
    for iter in np.arange(num_iters):
        theta = theta - alpha * gradient(theta, X, y)
        J_history[iter] = costFunction(theta, X, y)
    return (theta, J_history)


def MaxMinNormalization(x):
    """
        归一化
    """
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x

def plot(X, y,res):
    """
        数据的分布
        画出决策边界
    """
    plt.scatter(X[y == 0, 1], X[y == 0, 2])
    plt.scatter(X[y == 1, 1], X[y == 1, 2])
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max(),
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, 1, linewidths=1, colors='b')
    plt.show()

# 数据集每行前两列为学生两门考试的成绩，第三列为学生是否通过考试
# 0 表示不通过 1 表示通过
# 数据以','号隔开
data = loaddata('task6_data.txt', ',')

# 打乱数据
# 并做归一化
data = np.random.permutation(data)
X_raw = data[:,0:2]
X = MaxMinNormalization(X_raw)

X= np.c_[np.ones((data.shape[0],1)), X]
y = data[:,2]

print(X, y)

# 初始化参数
# b = initial_theta[0]  w = initial_theta[1:]
initial_theta = np.zeros(X.shape[1])
lr = 0.1
iteration = 30000
theta , Cost_J = gradientDescent(X, y, initial_theta, alpha= lr, num_iters= iteration)
print('最终b, w结果：',theta)
print('准确率：', accuracy_score(y, predict(theta, X)))

# 绘制损失函数下降趋势
plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()

# 绘制决策边界
plot(X, y, theta)


