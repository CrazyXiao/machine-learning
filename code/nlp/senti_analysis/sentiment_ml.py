#!/usr/bin/env python

"""
    自然语言处理之中文
    基于机器学习
"""
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
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    # 去除停用词
    stopwords = read_lines('data/stopwords.txt')
    return [w for w in seg_list if w and w not in stopwords]


def clear_text(line):
    """ 清洗文本 """
    if line:
        line = line.strip()
        line = re.sub("[a-zA-Z0-9]", "", line) # 去除文本中的英文和数字
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line) # 去除文本中的中文符号和英文符号
    return line


def prepare_data(sourceFile, targetFile):
    """
        处理文本
    """
    with open(sourceFile, 'r', encoding='utf-8') as f:
        with open(targetFile, 'w', encoding='utf-8') as target:
            target_lines = []
            for line in f:
                line = clear_text(line)
                if not line:
                    continue
                seg_line = ' '.join(sent2word(line))
                target_lines.append(seg_line)
            target.write('\n'.join(target_lines))


def process_input():
    sourceFile = 'data/2000_neg.txt'
    targetFile = 'data/2000_neg_cut.txt'
    prepare_data(sourceFile, targetFile)

    sourceFile = 'data/2000_pos.txt'
    targetFile = 'data/2000_pos_cut.txt'
    prepare_data(sourceFile, targetFile)


if __name__ == '__main__':
    process_input()