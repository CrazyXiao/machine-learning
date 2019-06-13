"""
    实现简单的决策树
"""

import numpy as np
import pandas as pd
from collections import Counter

def calcShannonEnt(dataSet):
    """ 计算信息熵
    """
    # 获取最后一列的数据
    labels = dataSet[:,-1]
    # 统计所有类别对应出现的次数
    labelCounts = Counter(labels)
    # 数据已准备好，计算熵
    shannonEnt = 0.0
    dataLen = len(dataSet)
    for key in labelCounts:
        pro = labelCounts[key] / dataLen
        shannonEnt -= pro * np.log2(pro)
    return shannonEnt

def chooseFeature(dataSet):
    """
        选择最优属性
        gain = baseEntropy - newEntropy
    """
    baseEntropy = calcShannonEnt(dataSet) # 整个数据集的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历所有属性
    for i in range(len(dataSet[0]) -1):
        splitDict = Counter(dataSet[:, i])
        newEntropy = 0.0
        for v in splitDict:
            subDataSet = dataSet[dataSet[:, i]==v]
            prob = splitDict[v]/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        gain = baseEntropy - newEntropy
        if gain > bestInfoGain:
            bestInfoGain = gain
            bestFeature = i
    return bestFeature


def createTree(dataSet, feature_labels):
    """
        生成决策树
        返回字典树
        dataSet: 数据集
        feature_labels： 属性标签
    """
    labels = dataSet[:, -1]
    # 数据集样本类别相同
    if len(set(labels)) == 1:
        return labels[0]
    # 属性值为空或者唯一属性值相同，返回样本数最多的类别
    if len(dataSet[0]) == 1 or (len(dataSet[0]) == 2 and len(set(dataSet[:, 0])) == 1):
        resDict = dict(Counter(labels))
        sortedClassCount = sorted(resDict.items(), key=lambda item: item[1], reverse=True)
        return sortedClassCount[0][0]

    # 选择最优属性
    bestFeat = chooseFeature(dataSet)
    bestFeatLabel = feature_labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(feature_labels[bestFeat])
    # 对选择属性进行划分
    for v in Counter(dataSet[:, bestFeat]):
        # 划分后的子集不应该包含我们选择属性对应的列
        subDataSet = np.delete(dataSet[dataSet[:, bestFeat]==v], bestFeat, axis=1)
        subLabels = feature_labels[:]
        # 生成子树
        myTree[bestFeatLabel][v] = createTree(subDataSet,subLabels)
    return myTree


if __name__ == '__main__':
    # 读取数据集，这里忽略 ids 及 连续属性列
    dataset = pd.read_csv("watermelon_3a.csv", usecols=['color', 'root', 'knocks', 'texture', 'navel', 'touch', 'label'])
    feature_labels = list(dataset.columns)
    dataset = dataset.values
    res = createTree(dataset, feature_labels)
    print(res)