"""
    绘制树图形

"""
import pandas as pd
from task8 import createTree
import matplotlib.pyplot as plt


def plotNode(text, centerPt, parentPt, nodeType):
    """
        绘制注解，带箭头
        annotate 函数用来绘制注解
        parentPt： 父节点位置
        centerPt: 被指向的位置
        nodeType: 节点类型
    """
    createPlot.ax1.annotate(text, xy=parentPt,  xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction',
    va="center", ha="center", bbox=nodeType, arrowprops= {'arrowstyle': '<-'} )


# 内部节点文本框样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# 叶节点文本框样式
leafNode = dict(boxstyle="round4", fc="0.8")

def getNumLeafs(myTree):
    """
        获取叶节点的数目
        确定横轴x的长度
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs


def getTreeDepth(myTree):
    """
        获取树的深度
        确定纵轴y的长度
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def createPlot(tree):
    """
        创建画布
    """
    fig = plt.figure(1, facecolor='gray')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree, (0.5,1.0), '')
    plt.show()


def plotMidText(cntrPt, parentPt, txtString):
    """
        连线添加文字
    """
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=0)


def plotTree(myTree, parentPt, text):
    """
        绘制决策树
    """
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, text)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


if __name__ == '__main__':
    # 读取数据集，这里忽略 ids 及 连续属性列
    dataset = pd.read_csv("watermelon_3a.csv", usecols=['color', 'root', 'knocks', 'texture', 'navel', 'touch', 'label'])
    feature_labels = list(dataset.columns)
    dataset = dataset.values
    res = createTree(dataset, feature_labels)
    print(res)
    createPlot(res)