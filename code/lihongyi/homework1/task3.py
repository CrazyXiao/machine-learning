"""
    预测pm2.5
    这里我们用前九个小时pm2.5来预测第10小时的pm2.5
    基于AdaGrad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
从训练数据中提取出连续十个小时的观测数据，
最后一个小时的PM2.5作为该条数据的类标签，
而前九个小时的PM2.5值作为特征。
一天24个小时，一天内总共有24-10+1 =15条记录
"""
def get_train_data():
    data = pd.read_csv("data/train.csv")
    # 获取所有的pm2.5信息
    pm2_5 = data[data['observation'] == 'PM2.5'].iloc[:, 3:]
    xlist=[]
    ylist=[]
    for i in range(15):
        tempx = pm2_5.iloc[:,i:i+9]        #使用前9小时数据作为feature
        tempx.columns = np.array(range(9))
        tempy=pm2_5.iloc[:,i+9]         #使用第10个小数数据作为lable
        tempy.columns=[1]
        xlist.append(tempx)
        ylist.append(tempy)
    xdata = pd.concat(xlist)
    ydata = pd.concat(ylist)
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

class AdaGrad():
    def __init__(self, lr=10, epochs=10000):
        self.lr = lr
        self.epochs = epochs


    def fit(self, x, y):
        """
        """
        m = y.size
        w = np.zeros(x.shape[1])
        s_grad = np.zeros(x.shape[1])
        cost = []
        for j in range(self.epochs):
            y_hat = np.dot(x,w)  # 模型函数
            error = y_hat -y
            grad = np.dot(x.transpose(), error) * 2
            s_grad += grad ** 2
            ada = np.sqrt(s_grad)
            w -= self.lr * grad / ada
            J = 1.0 / (m) * (np.sum(np.square(y_hat - y)))
            cost.append(J)
        return w, cost


X_train, y_train = get_train_data()
X_test = get_test_data()


gd = AdaGrad()
w, cost = gd.fit(X_train, y_train)
print(w, cost[-1])

# 预测
h = np.dot(X_test,w)
real=pd.read_csv('data/answer.csv')
erro=abs(h-real.value).sum()/len(real.value)
print('平均绝对值误差',erro)


plt.title("linear regression")
plt.plot(cost)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
