"""
    循环神经网络
"""
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.optimizers import Adam
from keras.utils import np_utils


# 加载数据
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 可视化
plt.title('Example %d. Label: %d' % (0, y_train[0]))
plt.imshow(x_train[0].reshape((28,28)), )
plt.show()

x_train = x_train.reshape(-1, 28, 28) /255
x_test = x_test.reshape(-1, 28, 28) /255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(SimpleRNN(
    batch_input_shape=(None, 28, 28), # 批次 和 输入尺寸
    units= 50, # 输出
))
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=0.001)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=50)

loss, accuracy = model.evaluate(x_test, y_test)

print('loss: %s, accuracy: %s' % (loss, accuracy))


