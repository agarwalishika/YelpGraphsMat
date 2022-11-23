#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import datetime
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

stops = set(stopwords.words("english"))

# def readData():
#     train = pd.read_csv('./data/YelpNYC/reviewContent_test','\t',header=None)
#     train.columns = ['user_id', 'prod_id', 'date', 'review']
#     return train

def generateBOW(df_features,vocabSize, name):
    now = datetime.datetime.now()
    #LogUtil.log("INFO", "Start to generate attribute BOW!")
    vocab = list()
    vocabPath = ''
    if vocabSize == 8000:
        vocabPath= name + '8kVocab.txt'
    elif vocabSize == 10000:
        vocabPath= name + '10kVocab.txt'
    else:
        vocabPath= name + 'fullVocab.txt'
    with open(vocabPath,'r',encoding='utf8') as fi:
        for line in fi:
            segs = line.rstrip().split('\t')
            if len(segs) <1:
                continue
            vocab.append(segs[0])

    BagOfWordsExtractor = CountVectorizer(vocabulary=vocab,
                                          analyzer='word',
                                          lowercase=True)
    bow_features = BagOfWordsExtractor.fit_transform(df_features)
    #LogUtil.log("INFO", "End to generate attribute BOW!")
    return bow_features.toarray()

def generateBOW_charLevel(df_features):
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    maxNumFeatures = 4001
    # bag of letter sequences (chars)
    BagOfWordsExtractor = CountVectorizer(max_df=1.0, min_df=1, max_features=maxNumFeatures,
                                          analyzer='char', ngram_range=(1, 3),
                                          binary=True, lowercase=True)
    trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:, 'question1'])
    trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:, 'question2'])
    X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
    df_features['f_bag_words'] = [X[i, :].toarray()[0] for i in range(0, len(df_features))]
    for j in range(0, len(df_features['f_bag_words'][0])):
        df_features['z_bag_words' + str(j)] = [df_features['f_bag_words'][i][j] for i in range(0, len(df_features))]
    df_features.fillna(0.0)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    return df_features

# train = readData()
# print(train)
# train = generateBOW(train)
# print(train)
# # col = [c for c in train.columns if c[:1]=='z']
# train.to_csv('train_bagofwords400.csv', index=False, columns = col)
# test = generateBOW(test)
# test.to_csv('test_bagofwords400.csv', index=False, columns = col)
# print ("done bag of words")
