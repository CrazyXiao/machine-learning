from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
    实现简单knn
"""
def euc(a, b):
    return distance.euclidean(a, b)

class SimpleKNN():
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 构造分类器
classifier = SimpleKNN()
classifier.fit(X_train, y_train)

# 测试集预测值
predictions = classifier.predict(X_test)
print(predictions)

# 准确率
print(accuracy_score(y_test, predictions))