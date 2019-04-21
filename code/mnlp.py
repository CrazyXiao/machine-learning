#!/usr/bin/env python

"""
    nltk为我们提供了一些内建语料库
    1 孤立语料库
    gutenberg
    webtext: 网络文本语料库，网络和聊天文本
    2 分类语料库
    brown: 布朗语料库，按照文本分类好的500个不同来源的文本
    3 重叠语料库
    reuters: 路透社语料库，1万多个新闻文档
    4 时序语料库
    inaugural: 就职演说语料库，55个总统的演说

    通用接口：
        fileids()：返回语料库中的文件

        categories()：返回语料库中的分类

        raw()：返回语料库的原始内容

        words()：返回语料库中的词汇

        sents()：返回语料库句子

        abspath()：指定文件在磁盘上的位置

        open()：打开语料库的文件流


"""

import nltk

print(nltk.corpus.gutenberg.fileids())

# 输出文章的原始内容
data = nltk.corpus.gutenberg.raw('chesterton-brown.txt')
print(data)

# 输出文章的单词列表
data = nltk.corpus.gutenberg.words('chesterton-brown.txt')
print(data)

# 输出文章的句子列表
data = nltk.corpus.gutenberg.sents('chesterton-brown.txt')
print(data)