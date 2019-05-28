#!/usr/bin/env python

"""
    逻辑回归用于线性模型
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))


# 定义损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
    if np.isnan(J):
        return (np.inf)
    return J


# 求解梯度
def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    grad = (1.0 / m) * X.T.dot(h - y)
    return (grad)

def predict(theta, X, threshold=0.5):
    """
        预测
    """
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))


def loaddata(file, delimeter):
    """
        加载数据
    """
    data = np.loadtxt(file, delimiter=delimeter)
    return data

def plot(X, y,res):
    """
        数据的分布
    """
    plt.scatter(X[y == 0, 1], X[y == 0, 2])
    plt.scatter(X[y == 1, 1], X[y == 1, 2])

    plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max(),
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, 1, linewidths=1, colors='b')

    plt.show()

# 考试1成绩 考试2成绩 是否通过
data = loaddata('data1.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = data[:,2]



initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
# print('Cost: \n', cost)
# print('Grad: \n', grad)

res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})
print(res.x)

plot(X,y,res)
