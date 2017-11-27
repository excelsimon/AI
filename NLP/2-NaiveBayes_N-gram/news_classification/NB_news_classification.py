#coding: utf-8
import os
import time
import random
import jieba  #处理中文
import nltk  #处理英文
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import codecs
import warnings
"""
使用贝叶斯多项式分布进行新闻主题分类
"""

#粗暴的词去重
def make_word_set(words_file):
    words_set = set()
    with codecs.open(words_file, 'r',encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip().encode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set


# 文本处理，也就是样本生成过程
def text_processing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 遍历文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 读取文件
        j = 1
        for file in files:
            if j > 100:  # 怕内存爆掉，只取100个样本文件，你可以注释掉取完
                break

            with codecs.open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as fp:
                raw = fp.read()
            ## 是的，随处可见的jieba中文分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式

            data_list.append(word_list)  # 训练集list
            class_list.append(folder)  # 类别
            j += 1

    ## 粗暴地划分训练集和测试集
    # data_class_list = zip(data_list, class_list)
    # random.shuffle(data_class_list)
    # index = int(len(data_class_list)*test_size)+1
    # train_list = data_class_list[index:]
    # test_list = data_class_list[:index]
    # train_data_list, train_class_list = zip(*train_list)
    # test_data_list, test_class_list = zip(*test_list)

    # 其实可以用sklearn自带的部分做
    train_data_list, test_data_list, train_class_list, test_class_list = train_test_split(data_list, class_list,
                                                                                          test_size=test_size)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    all_words_list = list(list(zip(*all_words_tuple_list))[0])

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度1000
            break

        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and len(all_words_list[t]) < 5\
                and len(all_words_list[t]) > 1:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words

# 文本特征
def text_features(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

# 分类，同时输出准确率等
def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## 使用nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    return test_accuracy

print("start")

## 文本预处理
folder_path = './Database/SogouC/Sample'
all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path, test_size=0.2)

# 生成stopwords_set
stopwords_file = './stopwords_cn.txt'
stopwords_set = make_word_set(stopwords_file)

## 文本特征提取和分类
# flag = 'nltk'
flag = 'sklearn'
deleteNs = range(0, 1000, 20)
test_accuracy_list = []
for deleteN in deleteNs:
    # feature_words = words_dict(all_words_list, deleteN)
    feature_words = words_dict(all_words_list, deleteN, stopwords_set)
    train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, feature_words, flag)
    test_accuracy = text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    test_accuracy_list.append(test_accuracy)
print(test_accuracy_list)

# 结果评价
#plt.figure()
plt.plot(deleteNs, test_accuracy_list)
plt.title('Relationship of deleteNs and test_accuracy')
plt.xlabel('deleteNs')
plt.ylabel('test_accuracy')
plt.show()
#plt.savefig('result.png')

print("finished")

"""
train_test_data:[['好','的','伟大'],[],][]] 一个list对应一篇文章分词以后的结果
all_words_list:['','',]所有文档中的词按词频排序后的list
"""
"""
output: 在test set最高达到约83%的准确率
[0.66666666666666663, 0.66666666666666663, 0.66666666666666663, 0.66666666666666663, 0.61111111111111116, 
0.66666666666666663, 0.66666666666666663, 0.66666666666666663, 0.66666666666666663, 0.66666666666666663, 
0.66666666666666663, 0.66666666666666663, 0.66666666666666663, 0.72222222222222221, 0.72222222222222221, 
0.72222222222222221, 0.72222222222222221, 0.66666666666666663, 0.72222222222222221, 0.72222222222222221, 
0.77777777777777779, 0.72222222222222221, 0.83333333333333337, 0.77777777777777779, 0.72222222222222221, 
0.77777777777777779, 0.77777777777777779, 0.72222222222222221, 0.72222222222222221, 0.72222222222222221, 
0.72222222222222221, 0.72222222222222221, 0.72222222222222221, 0.72222222222222221, 0.77777777777777779, 
0.83333333333333337, 0.83333333333333337, 0.83333333333333337, 0.83333333333333337, 0.77777777777777779,
 0.72222222222222221, 0.72222222222222221, 0.72222222222222221, 0.72222222222222221, 0.72222222222222221, 
 0.77777777777777779, 0.72222222222222221, 0.77777777777777779, 0.83333333333333337, 0.72222222222222221]

"""




















