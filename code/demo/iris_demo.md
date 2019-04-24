#### 例子

本例中，我们在 Iris 数据集 上训练一个简单的分类器，它与scikit-learn捆绑在一起。

数据集具有花的四个特征：萼片长度，萼片宽度，花瓣长度和花瓣宽度，并将它们分为三个花种 （标签）：setosa、versicolor或virginica。 标签已经被表示为数据集中的数字： 0（setosa），1（versicolor）和2（virginica）。

我们清洗Iris数据集，并将其分为独立的训练和测试集：保留最后10个数据点进行测试， 剩余的进行训练。然后我们在训练集训练分类器，并对测试集进行预测。

```
#!/usr/bin/env python

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

#loading the iris dataset
iris = load_iris()

x = iris.data #array of the data
y = iris.target #array of labels (i.e answers) of each data entry

#getting label names i.e the three flower species
y_names = iris.target_names

#taking random indices to split the dataset into train and test
test_ids = np.random.permutation(len(x))

#splitting data and labels into train and test
#keeping last 10 entries for testing, rest for training

x_train = x[test_ids[:-10]]
x_test = x[test_ids[-10:]]

y_train = y[test_ids[:-10]]
y_test = y[test_ids[-10:]]

#classifying using decision tree
clf = tree.DecisionTreeClassifier()

#training (fitting) the classifier with the training set
clf.fit(x_train, y_train)

#predictions on the test dataset
pred = clf.predict(x_test)

print(pred) #predicted labels i.e flower species
print(y_test) #actual labels
print((accuracy_score(pred, y_test))*100) #prediction accuracy
```

由于我们在每次迭代中随机分割和分类训练，所以准确性可能会有所不同。运行上面的代码得到：

```
[0 1 1 1 0 2 0 2 2 2]
[0 1 1 1 0 2 0 2 2 2]
100.0
```

第一行包含由我们的分类器预测的测试数据的标签（即花种），第二行包含数据集中给出的实际花种。 我们这次准确率达到100％。