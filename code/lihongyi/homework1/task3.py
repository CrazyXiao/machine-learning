"""
    预测pm2.5
    这里我们用前九个小时pm2.5来预测第10小时的pm2.5
    基于AdaGrad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_train_data():
    """
    从训练数据中提取出连续十个小时的观测数据，
    最后一个小时的PM2.5作为该条数据的类标签，
    而前九个小时的PM2.5值作为特征。
    一天24个小时，一天内总共有24-10+1 =15条记录
    """
    data = pd.read_csv("data/train.csv")
    pm2_5 = data[data['observation'] == 'PM2.5'].iloc[:, 3:]  # 获取所有的pm2.5信息
    pm2_5 = pm2_5.apply(pd.to_numeric, axis=0)
    xlist=[]
    ylist=[]
    for i in range(15):
        tempx = pm2_5.iloc[:,i:i+9]
        tempx.columns = np.array(range(9))
        tempy=pm2_5.iloc[:,i+9]
        tempy.columns=[1]
        xlist.append(tempx)
        ylist.append(tempy)
    xdata = pd.concat(xlist)
    ydata = pd.concat(ylist)
    # 去除异常值
    xdata, ydata = filter(xdata,ydata)
    X = np.array(xdata, float)
    # 加上bias
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = np.array(ydata, float)
    return X, y


def get_test_data():
    data = pd.read_csv("data/test.csv")
    pm2_5 = data[data['AMB_TEMP'] == 'PM2.5'].iloc[:, 2:]
    X = np.array(pm2_5, float)
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X

def filter(X, y):
    """
        这里过滤掉存在异常数值的行
    """
    cond1 = X.apply(lambda x: x > 0).all(axis=1)
    X = X[cond1]
    y = y[cond1]
    cond2 = y.apply(lambda x: x > 0)
    X = X[cond2]
    y = y[cond2]
    return X, y


class AdaGrad():
    def __init__(self, lr=20, epochs=20000):
        self.lr = lr
        self.epochs = epochs

    def costFunction(self, X, y, theta):
        """
            损失函数
            这里的cost不同于直接意义上的mse
        """
        m = y.size
        h = X.dot(theta)
        J = 1.0 / (2 * m) * (np.sum(np.square(h - y)))
        return J

    def fit(self, X, y):
        """
        """
        self.X = X
        self.y = y
        m = y.size
        w = np.zeros(X.shape[1])
        s_grad = np.zeros(X.shape[1])
        cost = []
        for j in range(self.epochs):
            h = X.dot(w)
            grad = X.T.dot(h - y)
            s_grad += grad ** 2
            ada = np.sqrt(s_grad)
            w = w - self.lr * (1.0 / m) * grad / ada
            J = self.costFunction(X,y,w)
            self.cost = J
            cost.append(J)
        self.w = w
        return w, cost

# 1 获取训练集
X_train, y_train = get_train_data()

# 2 训练数据 输出模型
gd = AdaGrad()
w, cost = gd.fit(X_train, y_train)
print('cost:', cost[-1])

# 3 获取测试集
X_test = get_test_data()
real= pd.read_csv('data/answer.csv')
# 4 模型评估
y_hat = np.dot(X_test,w)

sse = ((y_hat - real.value)**2).sum()
# mse = ((y_hat - real.value)**2).mean()
ssr = ((y_hat- real.value.mean())**2).sum()
sst = ((real.value- real.value.mean())**2).sum()

r2 = 1 - sse / sst
print('r2系数为:', r2)  # 0.86

# 可视化损失函数下降趋势
plt.title("linear regression")
plt.plot(cost)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
