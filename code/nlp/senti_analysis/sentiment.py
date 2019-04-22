#!/usr/bin/env python

"""
    自然语言处理之中文
    基于情感字典的情感分析

    缺陷：
    1 段落的得分是其所有句子得分的平均值，这一方法并不符合实际情况。
    正如文章中先后段落有重要性大小之分，一个段落中前后句子也同样有重要性的差异。
    2 有一类文本使用贬义词来表示正向意义，这类情况常出现与宣传文本中
"""

from collections import defaultdict
import re
import jieba


def read_lines(filename):
    """
        读取文件，每行为元素，返回列表
    """
    data = []
    with open(filename, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(line)
    return data



def sent2word(sentence):
    """
        分词
    """
    segList = jieba.cut(sentence.strip())
    stopwords = read_lines('data/stopwords.txt')
    return [w for w in segList if w and w not in stopwords]


def list_to_dict(word_list):
    """将分词后的列表转为字典，key为单词，value为单词在列表中的索引，索引相当于词语在文档中出现的位置"""
    data = {}
    for x in range(len(word_list)):
        data[word_list[x]] = x
    return data


def classify_words(word_dict):
    """词语分类,找出情感词、否定词、程度副词"""
    # 读取情感字典文件
    sen_list = read_lines('data/BosonNLP_sentiment_score.txt')
    sen_dict = defaultdict()
    for s in sen_list:
        sen_dict[s.split(' ')[0]] = s.split(' ')[1]

    # 读取否定词文件
    not_word_list = read_lines('data/not.txt')

    # 读取程度副词文件
    degree_list = read_lines('data/adverb_dict.txt')
    degree_dic = defaultdict()
    for d in degree_list:
        values = re.split('\s+', d)
        degree_dic[values[0]] = values[1]

    # 分类结果，词语的index作为key,词语的分值作为value，否定词分值设为-1
    sen_word = dict()
    not_word = dict()
    degree_word = dict()

    # 分类
    for word in word_dict.keys():
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dic.keys():
            sen_word[word_dict[word]] = float(sen_dict[word])
        elif word in not_word_list and word not in degree_dic.keys():
            not_word[word_dict[word]] = -1
        elif word in degree_dic.keys():
            degree_word[word_dict[word]] = float(degree_dic[word])

    # 将分类结果返回
    return sen_word, not_word, degree_word


def socre_sentiment(sen_word, not_word, degree_word, seg_result):
    """计算得分"""
    W = 1
    score = 0
    for i in range(len(seg_result)):
        if i in not_word.keys():
            W *= -1
        elif i in degree_word.keys():
            W *= degree_word[i]
        elif i in sen_word.keys():
            score += W * sen_word[i]
            W = 1
    return score

def get_score(sententce):
    seg_list = sent2word(sententce)
    print(seg_list)
    sen_word, not_word, degree_word = classify_words(list_to_dict(seg_list))
    print(sen_word, not_word, degree_word)
    score = socre_sentiment(sen_word, not_word, degree_word, seg_list)
    return score


print(get_score('我的心很痛'))


