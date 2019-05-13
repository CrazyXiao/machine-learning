"""
    分类器
    神经网络
"""
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

# 加载数据
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 可视化
plt.title('Example %d. Label: %d' % (0, y_train[0]))
plt.imshow(x_train[0].reshape((28,28)), )
plt.show()

x_train = x_train.reshape(x_train.shape[0], -1) /255
x_test = x_test.reshape(x_test.shape[0], -1) /255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential([
    Dense(32, input_dim=784,),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08,decay=0.0)

model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

loss, accuracy = model.evaluate(x_test, y_test)

print('loss: %s, accuracy: %s' % (loss, accuracy))

