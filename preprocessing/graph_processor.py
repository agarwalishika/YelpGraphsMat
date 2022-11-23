import io
import numpy as np
# import configparser
import random
import re

import nltk
import pandas as pd
from  preprocessing.nlp_processor import TextPreProcessor
from preprocessing.bag_of_words import generateBOW
import scipy as sp
import networkx as nx
from scipy.sparse import csr_matrix, csc_matrix
import string

class Graph_processor(object):
    @staticmethod
    def generate_adjacency_matrix(data):
        #LogUtil.log("INFO", "Start to generate adjacency matrix!")
        # dataLeft = data[['idx','prod_id']]
        # dataRight = data[['idx','prod_id']]
        prod_ids = data['prod_id'].tolist()
        # print(len(prod_ids))
        prod_ids = list(set(prod_ids))
        # print(len(prod_ids))
        # source, target = [], []
        edgeList = []
        for p in prod_ids:
            nodes = data.loc[data['prod_id'] == p]['idx'].tolist()
            # print(nodes)
            # input('...')
            for i in range(len(nodes)):
                edgeList.append((nodes[i], nodes[i]))
                for j in range(i + 1, len(nodes)):
                    edgeList.append((nodes[i], nodes[j]))
        # edgeList = pd.DataFrame({'source': source, 'target': target})
        # print(len(edgeList))
        # edgeList['weight'] = 1
        # innerJoin= pd.merge(data, data, how = 'inner', on = 'prod_id')
        # innerJoin = innerJoin[['idx_x','idx_y']]
        # innerJoin = innerJoin.drop_duplicates(keep = False)
        # innerJoin['val'] = 1
        # edgeList = edgeList.sort_values(by=['source','target'])
        # print("inner join:")
        # print(innerJoin)
        #LogUtil.log("INFO", "Ready to generate adjacency matrix!")
        G = nx.from_edgelist(edgeList)
        nodeList = set(data['idx'].values.tolist())
        nodeList = sorted(nodeList)
        # print(nodeList)
        # print(nodeList)
        adjacency_matrix = nx.adjacency_matrix(G, nodelist=nodeList)
        # print(adjacency_matrix.shape)
        # print(adjacency_matrix)
        #LogUtil.log("INFO", "End to generate adjacency matrix!")
        res = csc_matrix(adjacency_matrix, dtype=np.float64)
        print(res.shape)
        return res

    # @staticmethod
    # def generate_chi_adjacency_matrix(data):
    #     LogUtil.log("INFO", "start to generate adjacency matrix!")
    #     dataLeft = data[['user_id','prod_id']]
    #     dataRight = data[['user_id','prod_id']]
    #     innerJoin= pd.merge(dataLeft,dataRight,how = 'inner',on = 'prod_id')
    #     innerJoin = innerJoin[['user_id_x','user_id_y']]
    #     df = pd.crosstab(innerJoin.user_id_x, innerJoin.user_id_y)
    #     idx = df.columns.union(df.index)
    #     df_adjacency_matrix = df.reindex(index=idx, columns=idx, fill_value=0)
    #     print(df_adjacency_matrix)
    #     # print(adjacency_matrix)
    #     LogUtil.log("INFO", "end to generate adjacency matrix!")
    #     print(csr_matrix(df_adjacency_matrix))
    #     return  df_adjacency_matrix

    @staticmethod
    def generate_attribute_matrix(data,vocabSize,name):
        #LogUtil.log("INFO", "Start to generate attribute matrix!")
        # print(data)
        data = data[['idx','review']]
        data = data.sort_values(by=['idx'])
        data['review'] = data['review'].apply(lambda x: str(x).rstrip() + " ")
        # print(data.shape)
        data = data.groupby('idx').sum()
        print('in generating attribute matrix')
        # print(data.shape)
        # print(data)
        # input('...')
        data['review'] = data.review.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        data['review'] = data.review.map(lambda x:TextPreProcessor.remove_stopwords(x,'nltk'))
        BOW_Review = generateBOW(data['review'].tolist(), vocabSize, name)
        df_BOW_Review =pd.DataFrame(np.matrix(BOW_Review))
        # print(data)
        # data = data.reset_index(drop=True)
        # print(data)
        attribute_matrix = pd.merge(data, df_BOW_Review, left_index=True, right_index=True)
        # print(attribute_matrix)
        attribute_matrix = attribute_matrix.drop(columns=['review'])
        attribute_matrix = attribute_matrix.sort_values(by=['idx'])
        #LogUtil.log("INFO", "End to generate attribute matrix!")
        res = csc_matrix(attribute_matrix, dtype=np.float64)
        print(res.shape)
        # print(np.sum(res, axis=1))
        return res

    @staticmethod
    def generate_all_review(data):
        #LogUtil.log("INFO", "Start to generate attribute matrix!")
        # print(data)
        data = data[['idx','review']]
        data = data.sort_values(by=['idx'])
        data['review'] = data['review'].apply(lambda x: str(x).rstrip()+" ")
        # print(data.shape)
        data = data.groupby('idx').sum()
        # print(data)
        data['review'] = data.review.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        data['review'] = data.review.map(lambda x:TextPreProcessor.remove_stopwords(x,'nltk'))
        data['review'] = data.review.map(lambda x:x.translate(str.maketrans("","", string.punctuation)))
        # BOW_Review = generateBOW(data['review'].tolist(),vocabSize)
        # df_BOW_Review =pd.DataFrame(np.matrix(BOW_Review))
        # print(data)
        # data = data.reset_index(drop=True)
        # print(data)
        # attribute_matrix = pd.merge(data,df_BOW_Review, left_index=True, right_index=True)
        # print(attribute_matrix)
        # attribute_matrix = attribute_matrix.drop(columns=['review'])
        # attribute_matrix = attribute_matrix.sort_values(by=['idx'])
        # LogUtil.log("INFO", "End to generate attribute matrix!")
        return data