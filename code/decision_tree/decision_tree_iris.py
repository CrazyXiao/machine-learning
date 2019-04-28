#!/usr/bin/env python

"""
    iris 决策树
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 构造分类器
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# 测试集预测值
predictions = classifier.predict(X_test)
print(predictions)

# 准确率
print(accuracy_score(y_test, predictions))



