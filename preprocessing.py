import numpy as np
import pandas as pd
import random
import csv
from preprocessing.graph_processor import Graph_processor
from scipy.sparse import csr_matrix
from scipy.io import savemat
from datetime import date

def ProcessMeta(name):
    metaPath = 'data/Yelp' + name + '/metadata'
    reviewPath = 'data/Yelp' + name +'/reviewContent'
    idx2userPath = 'data/Yelp' + name + '/userIdMapping'
    data = dict()
    review_content = []
    with open(reviewPath) as f:
        content = f.readlines()
    for line in content:
        r = line.split('\t')[-1][:-1]
        review_content.append(r)
        
    reviewdata = pd.DataFrame(review_content, columns=['review'])
    df_meta = pd.read_csv(metaPath, sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    df_meta.columns = ['user_id', 'prod_id', 'rating', 'label', 'date']
    df_joint = df_meta.assign(review=reviewdata)
    df_joint = df_joint[['user_id', 'prod_id', 'label', 'review']]
    print("review:", df_joint.shape)

    # get all user_id
    idx2user = pd.read_csv(idx2userPath, sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    idx2user.columns = ['idx', 'user_id']
    # user_id = list(set(df_meta['user_id'].values.tolist()))
    # user_id = pd.DataFrame(user_id, columns=['user_id'])
    # sorted(user_id)
    # idx = np.arange(len(user_id)).tolist()
    # idx2user = pd.DataFrame({'idx': idx})
    # idx2user = idx2user.assign(user_id = user_id)
    # print(idx2user)
    # print(idx2user.shape)
    df_joint_idx = pd.merge(df_joint, idx2user, how='inner', on='user_id')
    df_joint_idx = df_joint_idx.drop_duplicates(keep = False)
    newIdx, unique = pd.factorize(df_joint_idx['idx'])
    # print(df_joint_idx.iloc[:10])
    df_joint_idx['idx'] = newIdx
    print("idx:", df_joint_idx.shape)
    # print(df_joint_idx.iloc[:10])
    label = df_joint_idx[['idx', 'label']]
    label = label.groupby(['idx', 'label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    print("label shape:", label.shape)
    # data2Pos = label.loc[label['label'] == -1]
    # data2Neg = label.loc[label['label'] == 1]
    # dataNew = pd.concat([data2Pos, data2Neg])
    # print(dataNew.shape)
    # print(dataNew.iloc[:10])
    # dataNew = label
    # df_joint = pd.merge(df_joint, idx2user, how='inner', on='user_id')
    # df_joint_final = pd.merge(df_joint_idx, label, how='inner', on='idx')

    # df_joint_final['idx'] = newIdx
    jointAdjacency = df_joint_idx[['idx', 'prod_id']]
    jointAttribute = df_joint_idx[['idx', 'review']]

    adjacency = Graph_processor.generate_adjacency_matrix(jointAdjacency)
    attribute = Graph_processor.generate_attribute_matrix(jointAttribute, 5000, name)
    # dataNew = dataNew.drop(columns=['idx']).as_matrix(columns=None)
    # dataNew = pd.merge()

    data['Label'] = label
    data['Network'] = adjacency
    data['Attributes'] = csr_matrix(attribute)
    print('Label: ', label.shape)
    print('Adjacency matrix shape:', adjacency.shape)
    print('Attribute matrix shape:', attribute.shape)

    return data


def ProcessMetaDF(name):
    metaPath = 'data/Yelp' + name + '/metadata'
    reviewPath = 'data/Yelp' + name +'/reviewContent'
    idx2userPath = 'data/Yelp' + name + '/userIdMapping'
    data = dict()
    review_content = []
    with open(reviewPath, encoding='utf-8') as f:
        content = f.readlines()
    for line in content:
        r = line.split('\t')[-1][:-1]
        review_content.append(r)
    reviewdata = pd.DataFrame(review_content, columns=['review'])
    df_meta = pd.read_csv(metaPath, sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    df_meta.columns = ['user_id', 'prod_id', 'rating', 'label', 'date']
    df_joint = df_meta.assign(review=reviewdata)
    print("original df shape: ", df_joint.shape)
    # print("review:", df_joint.shape)
    # print(df_joint['user_id'])
    idx2user = pd.read_csv(idx2userPath, sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    idx2user.columns = ['idx', 'user_id']
    df_joint = pd.merge(df_joint, idx2user, how='inner', on='user_id')
    df_joint = df_joint.drop_duplicates(keep = False)
    newIdx, unique = pd.factorize(df_joint['idx'])
    df_joint['idx'] = newIdx
    # df_joint = df_joint[['idx', 'prod_id', 'label', 'review']]
    df_joint_nolabel = df_joint[['idx', 'prod_id', 'rating', 'date', 'review']]
    print(df_joint_nolabel.shape)
    # print(df_joint)
    # get all user_ids
    user_ids = list(set(df_joint['idx'].values.tolist()))
    print("number of users: ", len(user_ids))
    # get all prod_ids
    prod_ids = list(set(df_joint['prod_id'].values.tolist()))
    print('number of prods:', len(prod_ids))
    print("num_reviews / num_prods: ", df_joint.shape[0] / len(prod_ids))
    idx_fake = []
    label_fake = []
    idx_real = []
    label_real = []
    idx_labels = df_joint.groupby(['idx', 'label']).size().to_frame().reset_index()
    for u in user_ids:
        tmp = idx_labels.loc[idx_labels['idx'] == u]
        if len(tmp) == 1 and tmp[['label']].values[0][0] == -1:
            idx_fake.append(u)
            label_fake.append(1)
        else:
            idx_real.append(u)
            label_real.append(0)
    df_fake = pd.DataFrame([idx_fake, label_fake]).transpose()
    df_fake.columns = ['idx', 'label']
    df_real = pd.DataFrame([idx_real, label_real]).transpose()
    df_real.columns = ['idx', 'label']
    df_label = pd.concat([df_fake, df_real])
    df_joint_final = df_joint_nolabel.merge(df_label, how='inner', on='idx')
    print('number of fake users: ', len(idx_fake))
    print('number of real users', len(idx_real))
    print("final df shape: ", df_joint_final.shape)
    print("Saving dataframe...")
    df_joint_final.to_csv('df_YelpZip_comp.csv', sep='\t', header=None, index=False)

    # print(df_joint_idx.iloc[:10])
    # label = df_joint[['idx', 'label']]
    # label = label.groupby(['idx', 'label']).size().groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='label')
    # print("label shape:", label.shape)
    # data2Pos = label.loc[label['label' == -1]
    # data2Neg = label.loc[label['label'] == 1]
    # dataNew = pd.concat([data2Pos, data2Neg])
    # print(dataNew.shape)

    # print(dataNew.iloc[:10])
    # dataNew = label
    # df_joint = pd.merge(df_joint, idx2user, how='inner', on='user_id')
    # df_joint_final = pd.merge(df_joint_idx, label, how='inner', on='idx')

    # # df_joint_final['idx'] = newIdx
    # jointAdjacency = df_joint[['idx', 'prod_id']]
    # jointAttribute = df_joint[['idx', 'review']]
    #
    # adjacency = Graph_processor.generate_adjacency_matrix(jointAdjacency)
    # attribute = Graph_processor.generate_attribute_matrix(jointAttribute, 5000, name)
    # # dataNew = dataNew.drop(columns=['idx']).as_matrix(columns=None)
    # # dataNew = pd.merge()
    #
    # data['Label'] = label
    # data['Network'] = adjacency
    # data['Attributes'] = csr_matrix(attribute)
    # print('Label: ', label.shape)
    # print('Adjacency matrix shape:', adjacency.shape)
    # print('Attribute matrix shape:', attribute.shape)

    # return data

def SampleMeta(name, nb_graphs):
    file = 'df_Yelp' + name + '.csv'
    df = pd.read_csv(file)
    # print(df)
    label = df[['idx', 'label']].drop_duplicates()
    # print(df.columns)
    # print(label.shape[0])
    # print(df.shape[0])
    # get all user_ids
    user_ids = list(set(df['idx'].values.tolist()))
    print("number of users: ", len(user_ids))
    # get all prod_ids
    prod_ids = list(set(df['prod_id'].values.tolist()))
    print('number of prods:', len(prod_ids))
    print("num_reviews / num_prods: ", df.shape[0] / len(prod_ids))
    # input('...')
    nb_users = []
    ratio_anomaly = []
    nb_products = []

    for i in range(nb_graphs):
        data = dict()
        prod_selected = random.sample(prod_ids, 1)
        df_selected = df.loc[df['prod_id'].isin(prod_selected)]
        user_selected = list(set(df_selected['idx'].values.tolist()))
        count = 0

        while len(user_selected) < 500:
            user_next = random.sample(user_selected, 1)
            prod_available = list(set(df.loc[df['idx'].isin(user_next)]['prod_id'].values.tolist()))
            prod_next = random.sample(prod_available, 1)
            prod_selected = list(set(prod_next + prod_selected))
            df_selected = df.loc[df['prod_id'].isin(prod_selected)]
            user_selected = list(set(df_selected['idx'].values.tolist()))
            count += 1
        ####### start from random user
        # user_selected = random.sample(user_ids, 1)
        # print('selected user: ', user_selected)
        # global df_selected
        # df_selected = df.loc[df['idx'].isin(user_selected)]
        # print('df init: ', df_selected.shape[0])
        # prod_selected = list(set(df_selected['prod_id'].values.tolist()))
        # print('prod init: ', len(prod_selected))
        # count = 0
        # while len(user_selected) < 2000:
        #     df_selected = df.loc[df['prod_id'].isin(prod_selected)]
        #     user_selected = list(set(df_selected['idx'].values.tolist()))
        #     df_selected = df.loc[df['idx'].isin(user_selected)]
        #     prod_selected = list(set(df_selected['prod_id'].values.tolist()))
        #     # prod_selected = random.sample(prod_selected, max(1, int(len(prod_selected) * 0.1)))
        #     count += 1
        ######## start with sampling from products ############
        # prod_selected = random.sample(prod_ids, 1)
        # print(prod_selected)
        # df_selected = df.loc[df['prod_id'].isin(prod_selected)]
        # print(df_selected.shape[0])
        # # prod_selected = [prod_init]
        # user_selected = list(set(df_selected['idx'].values.tolist()))
        # print('selected users: ', len(user_selected))
        # count = 0
        # while len(user_selected) < 4000:
        #     df_selected = df.loc[df['idx'].isin(user_selected)]
        #     prod_selected = list(set(df_selected['prod_id'].values.tolist()))
        #     df_selected = df.loc[df['prod_id'].isin(prod_selected)]
        #     user_selected = list(set(df_selected['idx'].values.tolist()))
        #     count += 1
        # print('count: ', count)
        # df_selected = df.loc[df['idx'].isin(user_selected)]
        print('#users: ', len(user_selected))
        print('#products: ', len(prod_selected))
        print(df_selected.shape[0])
        # input('......')
        nb_products.append(len(prod_selected))
        new_id, unique = pd.factorize(df_selected['idx'])
        df_selected['idx'] = new_id
        data_adjacency = df_selected[['idx', 'prod_id']]
        print(data_adjacency.shape)
        print(len(np.unique(data_adjacency[['idx']].to_numpy())))

        data_attribute = df_selected[['idx', 'review']]

        adjacency_matrix = Graph_processor.generate_adjacency_matrix(data_adjacency)
        print(adjacency_matrix.shape)
        print(adjacency_matrix.count_nonzero()/adjacency_matrix.shape[0])

        attribute_matrix = Graph_processor.generate_attribute_matrix(data_attribute, vocabSize=5000, name=name)
        print(attribute_matrix.count_nonzero()/attribute_matrix.shape[0])
        input('...')
        # print(attribute_matrix)
        label_selected = df_selected[['idx', 'label']].drop_duplicates().sort_values(by='idx')[['label']].to_numpy()
        ratio_anomaly.append(np.count_nonzero(label_selected == -1) / label_selected.shape[0])
        # print(label_selected.shape)
        # print('non zero', attribute_matrix.count_nonzero())
        # input('...')
        data['Label'] = label_selected
        data['Attributes'] = attribute_matrix
        data['Network'] = adjacency_matrix
        savemat('graphs/' + name + '_' + str(i) + '.mat', data)
        print('%d-th graph finished.' % (i + 1))

    data_stats = [nb_users, ratio_anomaly, nb_products]
    data_stats_df = pd.DataFrame(data_stats).transpose()
    data_stats_df.columns = ['graph size', 'anomaly ratio', 'number of prods']
    data_stats_df.to_csv('graphs/graph_stats.csv')


def sampleZip_s(name, nb_graphs):
    file = 'df_Yelp' + name + '_comp.csv'
    df = pd.read_csv(file, sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    df.columns = ['idx', 'prod_id', 'rating', 'date', 'review', 'label']
    data_pos = df.loc[df['label']==1]
    data_neg = df.loc[df['label']==-1]
    nb_users = []
    ratio_anomaly = []
    nb_edges = []
    nb_anomalies = []
    for i in range(nb_graphs):
        data = {}
        nb_pos = random.randint(data_pos.shape[0] - 8000, data_pos.shape[0])
        # nb_anomalies.append(nb_pos)
        # print("pos count: ", nb_pos)
        nb_neg = random.randint(data_neg.shape[0] - 8000, data_neg.shape[0])
        # print("neg count: ", nb_neg)
        # nb_user = nb_pos + nb_neg
        # ratio_anomaly.append(nb_pos / nb_user)
        # nb_users.append(nb_user)
        d_pos = data_pos.sample(n=nb_pos, replace=False)
        d_neg = data_neg.sample(n=nb_neg, replace=False)
        d_new = pd.concat([d_pos, d_neg])
        new_idx, _ = pd.factorize(d_new['idx'])
        d_new['idx'] = new_idx
        data_adjacency = d_new[['idx', 'prod_id']]
        data_attribute = d_new[['idx', 'review']]
        # print('label: \n', d_new[['idx', 'label']].drop_duplicates().sort_values(by='idx'))
        adjacency_matrix = Graph_processor.generate_adjacency_matrix(data_adjacency)
        # print(adjacency_matrix.count_nonzero() / adjacency_matrix.shape[0] ** 2)
        attribute_matrix = Graph_processor.generate_attribute_matrix(data_attribute, vocabSize=8000, name=name)
        label_selected = d_new[['idx', 'label']].drop_duplicates().sort_values(by='idx')[['label']].to_numpy()
        # print('label: \n', label_selected)
        ratio_anomaly.append(np.count_nonzero(label_selected == 1) / label_selected.shape[0])
        nb_anomalies.append(np.count_nonzero(label_selected == 1))
        nb_users.append(label_selected.shape[0])
        data['Label'] = label_selected
        data['Attributes'] = attribute_matrix
        data['Network'] = adjacency_matrix
        count_edge = adjacency_matrix.count_nonzero()
        print("nb edges: ", count_edge)
        nb_edges.append(count_edge)
        savemat('graphs/' + name + '_' + str(i) + '_' + date.today().strftime("%Y_%m_%d") + '.mat', data)
        print('%d-th graph finished.' % (i + 1))
    data_stats = [nb_users, nb_edges, nb_anomalies, ratio_anomaly]
    data_stats_df = pd.DataFrame(data_stats).transpose()
    data_stats_df.columns = ['graph size', 'number of edges', 'number of anomalies', 'anomaly ratio', ]
    data_stats_df.to_csv('graphs/graph_stats_' + date.today().strftime("%Y_%m_%d") + '.csv', index=False)


sampleZip_s('Zip', 40)
