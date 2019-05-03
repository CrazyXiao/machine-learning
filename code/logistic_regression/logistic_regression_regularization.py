"""
    逻辑回归
    正则化
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


def loaddata(file, delimeter):
    """
        加载数据
    """
    data = np.loadtxt(file, delimiter=delimeter)
    return data

#定义sigmoid函数
def sigmoid(z):
    return(1.0 / (1 + np.exp(-z)))


# 定义损失函数
def costFunction(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg/(2.0*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J):
        return (np.inf)
    return J

# 求解梯度
def gradient(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    temp = np.zeros(theta.shape[0])
    temp[0] = 0
    temp[1:] = theta[1:]
    grad = (1.0 / m) * X.T.dot(h - y) + (reg / m) * temp
    return (grad)

def predict(theta, X, threshold=0.5):
    """
        预测
    """
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

def plot(X, y, axes=None):
    """
        数据的分布
    """
    if axes == None:
        axes = plt.gca()
    axes.scatter(X[y == 0, 1], X[y == 0, 2])
    axes.scatter(X[y == 1, 1], X[y == 1, 2])


data = loaddata('data2.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = data[:,2]

# plt.scatter(X[y == 0, 1], X[y == 0, 2])
# plt.scatter(X[y == 1, 1], X[y == 1, 2])
# plt.show()

poly = PolynomialFeatures(6)
XX = poly.fit_transform(data[:,0:2])

initial_theta = np.zeros(XX.shape[1])
cost = costFunction(initial_theta,1, XX, y)
print(cost)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

# Lambda = 0 : 没有正则化，过拟合
# Lambda = 1 : 正常
# Lambda = 100 : 正则化项太大，没拟合出决策边界
for i, C in enumerate([0.0, 1.0, 100.0]):
    # 最优化 costFunctionReg
    res = minimize(costFunction, initial_theta, args=(C, XX, y), jac=gradient, options={'maxiter': 3000})

    print(res.x)
    # 准确率
    accuracy = 100.0 * sum(predict(res.x, XX) == y.ravel()) / y.size
    # 对X,y的散列绘图
    plot(X, y, axes.flatten()[i])
    # 画出决策边界
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max(),
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
plt.show()