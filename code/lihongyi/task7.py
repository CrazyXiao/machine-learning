"""
    计算信息熵
"""

import numpy as np
import pandas as pd
from collections import Counter

def calcShannonEnt(data):
    """ 计算信息熵
    """
    # 获取最后一列的数据
    labels = data[data.columns.values[-1]]
    # 统计所有类别对应出现的次数
    labelCounts = Counter(labels)
    # 数据已准备好，计算熵
    shannonEnt = 0.0
    dataLen = len(data)
    for key in labelCounts:
        pro = labelCounts[key] / dataLen
        shannonEnt -= pro * np.log2(pro)
    return shannonEnt

data = pd.read_csv("watermelon_3a.csv")
res = calcShannonEnt(data)
print("香浓熵为:", res)