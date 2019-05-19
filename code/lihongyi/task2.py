"""
    实现 SGD 和 mini-batch
"""
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD():
    def __init__(self, lr=0.0001, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def forward(self, x):
        """
            预测模型
        """
        return np.dot(x, self.w.T) + self.b

    def SGD(self, x, y):
        """ 随机梯度下降
        """
        self.w = np.zeros((1, x.shape[1]))
        self.b = np.zeros((1, 1))
        self.cost = []
        for j in range(self.epochs):
            for i in range(x.shape[0]):
                y_hat = self.forward(x[i])
                error = y_hat - y[i]
                self.w -= self.lr * error * x[i]
                self.b -= self.lr * error
                J = 1.0 / 2 * (np.sum(np.square(error)))
                self.cost.append(J)
        return self.w, self.b, self.cost

    def miniBatch(self, x, y, batch_size=20):
        """
            小批量梯度下降
            这里每次随机获取batch_size个样本
            下次获取的数据集不变是不严谨的
        """
        self.w = np.zeros((1, x.shape[1]))
        self.b = np.zeros((1, 1))
        self.cost = []
        num = max(x.shape[0] // batch_size, 1)
        print(num)
        for i in range(self.epochs):
            # 将总样本划分成num个mini-batch
            for j in range(num):
                # 随机选取样本更新参数
                choose = np.random.choice(x.shape[0], batch_size, replace=False)
                x_ = x[choose]
                y_hat = self.forward(x_)
                error = y_hat - y[choose]
                self.w -= self.lr * np.dot(x_.T, error)
                self.b -= self.lr * error.sum()
                J = 1.0 / (2 * batch_size) * (np.sum(np.square(error)))
                self.cost.append(J)
        return self.w, self.b, self.cost

X = np.linspace(0, 10, 100).reshape(100,1)
y = 1+2*X + np.random.normal(0,0.5,(100,1))

gd = LinearRegressionGD(epochs=1000)
# w, b, cost = gd.miniBatch(X, y, batch_size=30)
w, b, cost= gd.SGD(X, y)
h = b + w*X
print(w, b, cost[-1])

plt.title("simple linear regression")
plt.scatter(X, y, color='black')
plt.plot(X, h, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


