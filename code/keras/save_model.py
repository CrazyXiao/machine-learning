"""
    保存训练后的模型
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model

def MaxMinNormalization(x):
    """
        归一化
    """
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x

# 自己构造测试集
X = 100 * np.random.rand(200)
X = MaxMinNormalization(X)
y = 0.5*X + 2 + np.random.normal(0,0.01,(200,))

# training set
X_train = X[:160]
y_train = y[:160]
# test set
X_test = X[160:]
y_test = y[160:]

model = Sequential()
# 添加神经层
model.add(Dense(1, input_shape=(1,)))

# 激活模型 选择误差和优化方法
# MSE 均方误差 MSE
model.compile(optimizer=SGD(lr=0.1), loss='mse')

# 训练模型
for i in range(1000):
    model.train_on_batch(X_train, y_train)

cost = model.evaluate(X_test, y_test, batch_size=40)
print('误差：', cost)

# 预测测试集
y_predict = model.predict(X_test)

print('test before save:', y_predict)

model.save('mymodel.h5')

del model

model = load_model('mymodel.h5')

y_predict = model.predict(X_test)

print('test after save:', y_predict)