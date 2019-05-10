# 机器学习笔记
Python拥有大量的数据分析、统计和机器学习库，使其成为许多数据科学家的首选语言。

以下是基于Python的机器学习总结，包括一些广泛使用的机器学习方法和工具。

**注：该项目所有的代码实现基于python3,文章中数学公式要在github上正常显示，需安装 chrome插件 [mathjax plugin for github](<https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima>) 或者clone后离线使用markdown编辑器阅读。**

## 机器学习方法

[从零开始掌握Python机器学习](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/Python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0.md>)

[机器学习相关算法](<https://github.com/CrazyXiao/machine-learning/blob/master/notes/机器学习相关算法.md>)

[机器学习应用实现步骤](https://github.com/CrazyXiao/machine-learning/blob/master/notes/算法应用步骤.md)

[特征工程](https://github.com/CrazyXiao/machine-learning/blob/master/notes/特征工程.md)

[自然语言处理](https://github.com/CrazyXiao/machine-learning/blob/master/notes/自然语言处理.md)

## 实战

[吴恩达老师的机器学习课程个人笔记及python实现](https://github.com/CrazyXiao/machine-learning/tree/master/notes/AndrewNg)

**李宏毅机器学习任务30天**

机器学习实战

《统计学习方法》算法实现

## 工具

### SciPy 栈

SciPy 栈由数据科学所使用的一组核心帮助包组成，用于统计分析和数据可视化。 由于其庞大的功能和易用性，scripy栈被认为是大多数数据科学应用的必备条件。该栈主要包含[NumPy](http://www.numpy.org/)，[Matplotlib](http://matplotlib.org/)，[pandas](http://pandas.pydata.org/)等包。

### scikit-learn

Scikit是一个用于Python的免费开源机器学习库。 它提供了现成的功能来实现诸如线性回归、 分类器、SVM、k-均值和神经网络等多种算法。它还有一些可直接用于训练和测试的样本数据集。由于其速度、鲁棒性和易用性，它是许多机器学习应用程序中使用最广泛的库之一。这里有一个使用scikit-learn进行机器学习的[demo](<https://github.com/CrazyXiao/machine-learning/tree/master/code/demo/iris_demo.md>),关于scikit-learn的更多内容可以在 [官方文档](http://scikit-learn.org/stable/user_guide.html) 中阅读。

### tensorflow

TensorFlow 是一款用于数值计算的强大的开源软件库，特别适用于大规模机器学习的微调。这里有一个使用tf解决手写识别问题的[demo](<../master/code/tf/mnist.py>),关于tensorflow的更多教程可以在[中文社区](<http://www.tensorfly.cn/tfdoc/get_started/introduction.html>)中阅读。

### keras

Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow、Theano以及CNTK]后端。Keras 为支持快速实验而生，能够把你的idea迅速转换为结果，如果你有如下需求，请选择Keras：

- 简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
- 支持CNN和RNN，或二者的结合
- 无缝CPU和GPU切换

## 基础

-  [pandas数据处理](../master/code/demo/pandas_demo.ipynb)
- [seaborn绘图](<https://elitedatascience.com/python-seaborn-tutorial>)

## 入门

|                             知识                             | 实现                                                         |
| :----------------------------------------------------------: | ------------------------------------------------------------ |
| [线性回归(Linear Regression)](<../master/notes/AndrewNg/线性回归.md>) | [代码](<../master/code/linear_regression>)                   |
| [逻辑回归(Logistic Regression)](<../master/notes/AndrewNg/逻辑回归.md>) | [代码](<../master/code/logistic_regression>)                 |
| [正则化(Regularization)](<../master/notes/AndrewNg/正则化.md>) | [代码](<../master/code/logistic_regression/logistic_regression_regularization.py>) |
|                    决策树(Decision Tree)                     | [代码](<../master/code/decision_tree>)                       |
|      [K近邻(k-Nearest Neighbour)](<../master/code/knn>)      | [代码](<../master/code/knn/simpleKnn.py>)                    |

------

#### 持续更新中...

