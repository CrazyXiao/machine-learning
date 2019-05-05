import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

"""
    决策树
    完成分类问题
"""
# load data
adult_data = pd.read_csv('./DecisionTree.csv')

feature_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
label_column = ['income']

#区分特征和目标列
features = adult_data[feature_columns]
label = adult_data[label_column]

features = pd.get_dummies(features)
print(features.shape)
print(label.shape)
# 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features.values, label.values.ravel(), test_size=0.2)

#初始化一个决策树分类器
classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
# 随机森林分类器
# classifier = RandomForestClassifier(n_estimators=30)
#用决策树分类器拟合数据
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
# 准确率
print(accuracy_score(y_test, predictions))