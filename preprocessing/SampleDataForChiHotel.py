from scipy import io
import numpy as np
import configparser
import re
import csv
import nltk
import pandas as pd
from nltk.stem import SnowballStemmer
from preprocessing.bag_of_words import generateBOW
import networkx as nx
from preprocessing.graph_processor import Graph_processor
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix


def get_label():
    data = dict()
    meta1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_meta_yelpResData_NRYRcleaned.txt',' ',header=None)
    meta1.columns = ['date','reviewId','user_id','prod_id','label','num1','num2','num3','num4']
    label1 = meta1[['user_id','prod_id','label']]
    id2User = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\idx_userIdYelpRes.tsv','\t',header=None)
    id2User.columns = ['idx','user_id']
    print(label1)
    print(id2User)
    label2 = pd.merge(label1,id2User,how = 'inner',on = 'user_id')
    label2 = label2.drop_duplicates(keep = False)
    print(label2)    
    label2 = label2[['idx','label']]
    label2 = label2.groupby(['idx','label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    print(label2)
    label2['label'] = label2['label'].replace('Y',1).replace('N',0)
    label2 = label2.drop(columns=['idx'])
    label2 = label2.as_matrix(columns=None)
    print(label2)
    data['label'] = label2
    # print(label2)
    io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\label_matrix_yelpRes_mat', data)
    data2 = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\label_matrix_yelpRes_mat')
    for k,v in data2.items():
        print(k,v)


def sampleHotelData(posCount,negCount,vocabSize):
    data = dict()
    meta1 = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_meta_yelpHotelData_NRYRcleaned.txt',' ',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    meta1.columns = ['date','reviewId','user_id','prod_id','label','num1','num2','num3','num4']
    review_content1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_review_yelpHotelData_NRYRcleaned.txt','\t',header=None)
    review_content1.columns = ['review']
    join_review1 =meta1.join(review_content1)
    label1 = join_review1[['user_id','prod_id','label','review']]
    id2User = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\idx_userIdYelpHotel.tsv','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    id2User.columns = ['idx','user_id']
    label2 = pd.merge(label1,id2User,how = 'inner',on = 'user_id')
    label2 = label2.drop_duplicates(keep = False)   
    label2 = label2[['idx','label']]
    label2 = label2.groupby(['idx','label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    label2['label'] = label2['label'].replace('Y',1).replace('N',0)
    data2Pos = label2.loc[label2['label']==1]
    data2Neg = label2.loc[label2['label']==0]
    data2Pos = data2Pos.sample(n=posCount,random_state=1)
    data2Neg = data2Neg.sample(n=negCount,random_state=1)
    dataNew = pd.concat([data2Pos,data2Neg])
    label1 = pd.merge(label1,id2User,how='inner',on='user_id')
    jointDataNew = pd.merge(label1,dataNew,how='inner',on = 'idx')
    newIdx, unique = pd.factorize(jointDataNew['idx'])
    jointDataNew['idx'] = newIdx
    jointDataNewAdjacency = jointDataNew[['idx','prod_id']]
    jointDataNewAttribute = jointDataNew[['idx','review']]

    adjacency = Graph_processor.generate_adjacency_matrix(jointDataNewAdjacency)
    attribute = Graph_processor.generate_attribute_matrix(jointDataNewAttribute,vocabSize)

    dataNew = dataNew.drop(columns=['idx'])
    dataNew = dataNew.as_matrix(columns=None)
    data['Label'] = dataNew

    data['Attributes'] = csr_matrix(attribute)
    data['Network'] = csc_matrix(adjacency)
    return  data

def sampleHotelDataReview(posCount,negCount):
    # data = dict()
    meta1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_meta_yelpHotelData_NRYRcleaned.txt',' ',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    meta1.columns = ['date','reviewId','user_id','prod_id','label','num1','num2','num3','num4']
    review_content1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_review_yelpHotelData_NRYRcleaned.txt','\t',header=None)
    review_content1.columns = ['review']
    join_review1 =meta1.join(review_content1)
    label1 = join_review1[['user_id','prod_id','label','review']]
    id2User = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\idx_userIdYelpHotel.tsv','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    id2User.columns = ['idx','user_id']
    label2 = pd.merge(label1,id2User,how = 'inner',on = 'user_id')
    label2 = label2.drop_duplicates(keep = False)
    label2 = label2[['idx','label']]
    label2 = label2.groupby(['idx','label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    label2['label'] = label2['label'].replace('Y',1).replace('N',0)
    data2Pos = label2.loc[label2['label']==1]
    data2Neg = label2.loc[label2['label']==0]
    data2Pos = data2Pos.sample(n=posCount,random_state=1)
    data2Neg = data2Neg.sample(n=negCount,random_state=1)
    dataNew = pd.concat([data2Pos,data2Neg])
    label1 = pd.merge(label1,id2User,how='inner',on='user_id')
    jointDataNew = pd.merge(label1,dataNew,how='inner',on = 'idx')
    newIdx, unique = pd.factorize(jointDataNew['idx'])
    jointDataNew['idx'] = newIdx
    jointDataNewAdjacency = jointDataNew[['idx','prod_id']]
    jointDataNewAttribute = jointDataNew[['idx','review']]
    data = Graph_processor.generate_all_review(jointDataNewAttribute)
    return  data
def ProcessFullHotelData(posCount,negCount):
    data = dict()
    meta1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_meta_yelpHotelData_NRYRcleaned.txt',' ',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    meta1.columns = ['date','reviewId','user_id','prod_id','label','num1','num2','num3','num4']
    review_content1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_review_yelpHotelData_NRYRcleaned.txt','\t',header=None)
    review_content1.columns = ['review']
    join_review1 = meta1.join(review_content1)
    label1 = join_review1[['user_id','prod_id','label','review']]
    id2User = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\idx_userIdYelpHotel.tsv','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    id2User.columns = ['idx','user_id']
    label2 = pd.merge(label1,id2User,how = 'inner',on = 'user_id')
    label2 = label2.drop_duplicates(keep = False)
    label2 = label2[['idx','label']]
    label2 = label2.groupby(['idx','label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    label2['label'] = label2['label'].replace('Y',1).replace('N',0)
    data2Pos = label2.loc[label2['label']==1]
    data2Neg = label2.loc[label2['label']==0]
    # data2Pos = data2Pos.sample(n=posCount,random_state=1)
    # data2Neg = data2Neg.sample(n=negCount,random_state=1)
    dataNew = pd.concat([data2Pos,data2Neg])
    label1 = pd.merge(label1,id2User,how='inner',on='user_id')
    jointDataNew = pd.merge(label1,dataNew,how='inner',on = 'idx')
    newIdx, unique = pd.factorize(jointDataNew['idx'])
    jointDataNew['idx'] = newIdx
    jointDataNewAdjacency = jointDataNew[['idx','prod_id']]
    jointDataNewAttribute = jointDataNew[['idx','review']]

    adjacency = Graph_processor.generate_adjacency_matrix(jointDataNewAdjacency)
    attribute = Graph_processor.generate_attribute_matrix(jointDataNewAttribute,5000)
    dataNew = dataNew.drop(columns=['idx'])
    dataNew = dataNew.as_matrix(columns=None)
    data['Label'] = dataNew
    data['Attributes'] = csr_matrix(attribute)
    data['Network'] = csc_matrix(adjacency)
    return  data


def sampleResData(posCount,negCount,vocabSize):
    data = dict()
    meta1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_meta_yelpResData_NRYRcleaned.txt',' ',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    meta1.columns = ['date','reviewId','user_id','prod_id','label','num1','num2','num3','num4']
    review_content1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_review_yelpResData_NRYRcleaned.txt','\t',header=None)
    review_content1.columns = ['review']
    join_review1 = meta1.join(review_content1)
    # join_review1 = pd.merge(meta1,review_content1,left_index=True,right_index=True)
    label1 = join_review1[['user_id','prod_id','label','review']]
    id2User = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\idx_userIdYelpRes.tsv','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    id2User.columns = ['idx','user_id']
    label2 = pd.merge(label1,id2User,how = 'inner',on = 'user_id')
    label2 = label2.drop_duplicates(keep = False)
    label2 = label2[['idx','label']]
    label2 = label2.groupby(['idx','label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    label2['label'] = label2['label'].replace('Y',1).replace('N',0)
    data2Pos = label2.loc[label2['label']==1]
    data2Neg = label2.loc[label2['label']==0]
    # print(data2Neg.shape)
    # print(data2Pos.shape)
    data2Pos = data2Pos.sample(n=posCount,random_state=1)
    data2Neg = data2Neg.sample(n=negCount,random_state=1)
    dataNew = pd.concat([data2Pos,data2Neg])
    label1 = pd.merge(label1,id2User,how='inner',on='user_id')
    jointDataNew = pd.merge(label1,dataNew,how='inner',on = 'idx')
    newIdx, unique = pd.factorize(jointDataNew['idx'])
    jointDataNew['idx'] = newIdx
    jointDataNewAdjacency = jointDataNew[['idx','prod_id']]
    jointDataNewAttribute = jointDataNew[['idx','review']]

    adjacency = Graph_processor.generate_adjacency_matrix(jointDataNewAdjacency)
    attribute = Graph_processor.generate_attribute_matrix(jointDataNewAttribute,vocabSize)

    dataNew = dataNew.drop(columns=['idx'])
    dataNew = dataNew.as_matrix(columns=None)
    data['Label'] = dataNew

    data['Attributes'] = csr_matrix(attribute)
    data['Network'] = csc_matrix(adjacency)
    return  data

def sampleResDataReview(posCount,negCount):
    meta1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_meta_yelpResData_NRYRcleaned.txt',' ',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    meta1.columns = ['date','reviewId','user_id','prod_id','label','num1','num2','num3','num4']
    review_content1=pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\output_review_yelpResData_NRYRcleaned.txt','\t',header=None)
    review_content1.columns = ['review']
    join_review1 = meta1.join(review_content1)
    # join_review1 = pd.merge(meta1,review_content1,left_index=True,right_index=True)
    label1 = join_review1[['user_id','prod_id','label','review']]
    id2User = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\idx_userIdYelpRes.tsv','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
    id2User.columns = ['idx','user_id']
    label2 = pd.merge(label1,id2User,how = 'inner',on = 'user_id')
    label2 = label2.drop_duplicates(keep = False)
    label2 = label2[['idx','label']]
    label2 = label2.groupby(['idx','label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    label2['label'] = label2['label'].replace('Y',1).replace('N',0)
    data2Pos = label2.loc[label2['label']==1]
    data2Neg = label2.loc[label2['label']==0]
    # print(data2Neg.shape)
    # print(data2Pos.shape)
    data2Pos = data2Pos.sample(n=posCount,random_state=1)
    data2Neg = data2Neg.sample(n=negCount,random_state=1)
    dataNew = pd.concat([data2Pos,data2Neg])
    label1 = pd.merge(label1,id2User,how='inner',on='user_id')
    jointDataNew = pd.merge(label1,dataNew,how='inner',on = 'idx')
    newIdx, unique = pd.factorize(jointDataNew['idx'])
    jointDataNew['idx'] = newIdx
    jointDataNewAdjacency = jointDataNew[['idx','prod_id']]
    jointDataNewAttribute = jointDataNew[['idx','review']]
    data = Graph_processor.generate_all_review(jointDataNewAttribute)

    return  data
def mergeMATFiles():
    dataHotel = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\data_matrix_yelpHotel_mat.mat')
    dataRes = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\data_matrix_yelpRes_mat.mat')
    labelHotel = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\label_matrix_yelpHotel_mat.mat')
    labelRes = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\label_matrix_yelpRes_mat.mat')
    dataHotelNew = dict()
    dataResNew = dict()
    dataHotelNew['label'] = labelHotel['label']
    dataHotelNew['attribute'] = dataHotel['attribute']
    dataHotelNew['adjacency'] = dataHotel['adjacency']

    dataResNew['label'] = labelRes['label']
    dataResNew['attribute'] = dataRes['attribute']
    dataResNew['adjacency'] = dataRes['adjacency']
    io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\full_matrix_yelpHotel_20190320', dataHotelNew)
    io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\full_matrix_yelpRes_20190320', dataResNew)

# data = sampleResDataReview(250,4250)
# data.to_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_fullReview_Res_20190403.tsv',index=True,header=False,sep='\t')
# # io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_fullVocab_Res_20190331',data)
# data = sampleHotelDataReview(250,4250)
#
# data.to_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_fullReview_Hotel_20190403.tsv',index=True,header=False,sep='\t')
# io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_fullVocab_Hotel_20190331',data)
# data = sampleResData(750,4250,50000)
# io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\750Pos_fullVocab_Res_20190331',data)
# data = sampleHotelData(750,4250,50000)
# io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\750Pos_fullVocab_Hotel_20190331',data)
#
# data = sampleResData(6857,27107,50000)
# io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\fullPos_fullVocab_Res_20190331',data)
# data = sampleHotelData(750,4276,50000)
# io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\fullPos_fullVocab_Hotel_20190331',data)
dataRes = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_vocab8k_Res_20190329.mat')
dataHotel = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_vocab8k_Hotel_20190329.mat')
bertVectorRes = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\Res_250Pos_Bert100DVector','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
bertVectorHotel = pd.read_csv('..\\YelpChi-20190315T163455Z-001\\YelpChi\\Hotel_250Pos_Bert100DVector','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
dataRes['Attributes'] = bertVectorRes
dataHotel['Attributes'] = bertVectorHotel
io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_Bert100D_Res_20190406',dataRes)
io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_Bert100D_Hotel_20190406',dataHotel)
print('finish!')
# print(bertVectorRes)
# print('finish')
# data2 = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\full_Hotel_20190324.mat')
# data3 = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_Hotel_20190324.mat')
# data4 = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_Res_20190324.mat')
# dataNew = dict()
# dataNew['Attributes']=csr_matrix(data['Attributes'].toarray())
# dataNew['Label'] = data['Label']
# dataNew['Network'] = data['Network']
# io.savemat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\250Pos_hotel_20190324_2',data)
# print(data['Network'])
# print(data['Label'])
# for k,v in data:
#     print(k)
#     print(v)