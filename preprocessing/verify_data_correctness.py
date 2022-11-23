import numpy as np
import configparser
import random
import re
from scipy import io
import nltk
import pandas as pd
from nltk.stem import SnowballStemmer
from utils import DataUtil, LogUtil
from  nlp_processor import TextPreProcessor
from preprocessing.bag_of_words import generateBOW
import scipy as sp
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from preprocessing.graph_processor import Graph_processor
import csv

# review_content=pd.read_csv('..\\YelpNYC-20190315T163619Z-001\\YelpNYC\\reviewContent','\t',header=None,quoting=csv.QUOTE_NONE,encoding='utf-8')
# review_content.columns = ['user_id','prod_id','date','review']
# print(review_content.shape)
# attribute_matrix = Graph_processor.generate_attribute_matrix(review_content,5000)
# data = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\top5kVocab_matrix_yelpRes_20190321.mat')
# data = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\attribute_matrix_yelpHotel_top5k_0321.mat')
data2 = io.loadmat('..\\YelpNYC-20190315T163619Z-001\\YelpNYC\\top5KVocab_matrix_nyc_20190322.mat')
for k,v in data2.items():
    print(k)
    print(v)
# new_attributes = csr_matrix(attribute_matrix)
# adjacency = data2['Network']
# label = data2['Label']
# print(np.count_nonzero(label == 1))
# label = 1-label
#
# print(np.count_nonzero(label == 1))
data_dict = dict()
data_dict['Label'] = data2['Label']
data_dict['Network'] = data2['Network']
attribute = data2['Attributes']
attribute = attribute.toarray()
new_attribute = csr_matrix(attribute)
data_dict['Attributes'] = new_attribute
#
io.savemat('..\\YelpNYC-20190315T163619Z-001\\YelpNYC\\top5kVocab_matrix_nyc_20190322_2.mat',data_dict)
# data2 = io.loadmat('..\\YelpChi-20190315T163455Z-001\\YelpChi\\full_matrix_yelpRes_20190321.mat')
# newData['Label'] = data['Label']
# newData['Attributes'] = csr_matrix(data['attribute'])
# newData['Network'] = csc_matrix(data2['Network'])
