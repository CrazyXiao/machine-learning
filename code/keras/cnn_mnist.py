"""
    卷积神经网络

"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils

# 加载数据
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 可视化
plt.title('Example %d. Label: %d' % (0, y_train[0]))
plt.imshow(x_train[0].reshape((28,28)), )
plt.show()

# 数据预处理
x_train = x_train.reshape(-1, 1, 28, 28) # channel 1 代表黑白照片
x_test = x_test.reshape(-1, 1, 28, 28)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 构建模型
model = Sequential()
# 卷积层
model.add(Conv2D(
    filters=32, # 滤波器
    kernel_size=(5,5), # 2D 卷积窗口的宽度和高度
    padding='same',
    input_shape=(1,28,28)
))
model.add(Activation('relu'))
# 池化层 向下取样 -- 池化不会压缩层高度
model.add(MaxPool2D(
    pool_size=(2,2), #  沿（垂直，水平）方向缩小比例的因数
    strides=(2,2), # 步长
    padding='same',
))
# output shape (32,14,14)

# 卷积层
model.add(Conv2D(
    filters=64, # 滤波器
    kernel_size=(5,5), # 2D 卷积窗口的宽度和高度
    padding='same',
))
model.add(Activation('relu'))
# 池化层 向下取样
model.add(MaxPool2D(
    pool_size=(2,2), #  沿（垂直，水平）方向缩小比例的因数
    strides=(2,2), # 步长
    padding='same',
))
# output shape (64,7,7)

# 全连接层
model.add(Flatten()) # 3维转1维
model.add(Dense(1024,))
model.add(Activation('relu'))

# 全连接层2
model.add(Dense(10,))
# softmax 用于分类
model.add(Activation('softmax'))
adam = Adam(lr=0.0001)

# 激活模型
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train, epochs=1, batch_size=32)
# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: %s, accuracy: %s' % (loss, accuracy))
