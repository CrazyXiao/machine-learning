#!/usr/bin/env python

"""
    自然语言处理之情感分析
"""

from collections import defaultdict
import os
import re
import jieba

def readLines(filename):
    data = []
    with open(filename, encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if not line.strip():
                continue
            data.append(line)
    return data



"""
1. 文本切割
"""
def sent2word(sentence):
    """
    Segment a sentence to words
    Delete stopwords
    """
    segList = jieba.cut(sentence.strip())
    stopwords = readLines('data/stopwords.txt')
    newSent = []
    for word in segList:
        if not word:
            continue
        if word in stopwords:
            continue
        else:
            newSent.append(word)
    return newSent


"""
2. 情感定位
"""
def classifyWords(wordDict):
    # (1) 情感词
    senList = readLines('data/BosonNLP_sentiment_score.txt')
    senDict = defaultdict()
    for s in senList:
        senDict[s.split(' ')[0]] = s.split(' ')[1]
    # (2) 否定词
    notList = readLines('not.txt')
    # (3) 程度副词
    degreeList = readLines('adverd_dict.txt')
    degreeDict = defaultdict()
    for d in degreeList:
        degreeDict[d.split(',')[0]] = d.split(',')[1]

    senWord = defaultdict()
    notWord = defaultdict()
    degreeWord = defaultdict()

    for word in wordDict.keys():
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[wordDict[word]] = senDict[word]
        elif word in notList and word not in degreeDict.keys():
            notWord[wordDict[word]] = -1
        elif word in degreeDict.keys():
            degreeWord[wordDict[word]] = degreeDict[word]
    return senWord, notWord, degreeWord

print(sent2word('这样的酒店配这样的价格还算不错'))


