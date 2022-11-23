import csv
import nltk
import pandas as pd
from  nlp_processor import TextPreProcessor
from sklearn.feature_extraction.text import CountVectorizer

def processData(metaPath,reviewPath, idx2userPath):
    data = dict()
    review_content = []
    with open(reviewPath) as f:
        content = f.readlines()
    for line in content:
        r = line.split('\t')[-1][:-1]
        review_content.append(r)
        # print(r)
        # input('...')
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
    print(df_joint_idx.iloc[:10])
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
    df_joint_final = pd.merge(df_joint_idx, label, how='inner', on='idx')

    # df_joint_final['idx'] = newIdx
    # jointAdjacency = df_joint_final[['idx', 'prod_id']]
    jointAttribute = df_joint_final[['idx', 'review']]

    # adjacency = Graph_processor.generate_adjacency_matrix(jointAdjacency)
    # attribute = Graph_processor.generate_attribute_matrix(jointAttribute, 5000)
    # dataNew = dataNew.drop(columns=['idx']).as_matrix(columns=None)
    # dataNew = pd.merge()

    # data['Label'] = dataNew
    # data['Network'] = csr_matrix(adjacency)
    # data['Attributes'] = csr_matrix(attribute)
    return  jointAttribute

def build_vocab(name):

    data = processData('data/Yelp' + name + '/metadata', 'data/Yelp' + name + '/reviewContent', 'data/Yelp' + name + '/userIdMapping')
    print("finishing processData.")
    data['review'] = data['review'].apply(lambda x: str(x).rstrip() + " ")
    # print(data.shape)
    data = data.groupby('idx').sum()
    # print(data)
    data['review'] = data.review.map(lambda x: ' '.join(
        [TextPreProcessor._stemmer.stem(word) for word in
         nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
    data['review'] = data.review.map(lambda x: TextPreProcessor.remove_stopwords(x, 'nltk'))

    print('finishing cleaning data.')
    BagOfWordsExtractor_8k = CountVectorizer(max_features=8000,
                                             analyzer='word',
                                             lowercase=True)
    bow_features_8k = BagOfWordsExtractor_8k.fit(data['review'].tolist())
    vocab_8k = bow_features_8k.vocabulary_
    print('bow features vocab 8k:')
    print(len(vocab_8k))
    vocab_8k = sorted(vocab_8k.items(),key=lambda item:item[1])
    res = [x[0] for x in vocab_8k]
    with open(name + '8kVocab.txt','w',encoding='utf8') as fo:
        for v in res:
            fo.write(str(v)+'\n')

    BagOfWordsExtractor_full = CountVectorizer(
        analyzer='word',
        lowercase=True)
    bow_features_full = BagOfWordsExtractor_full.fit(data['review'].tolist())
    vocab_full = bow_features_full.vocabulary_
    print('bow features vocab full:')
    print(len(vocab_full))
    vocab_full = sorted(vocab_full.items(),key=lambda item:item[1])
    res = [x[0] for x in vocab_full]
    with open(name + 'fullVocab.txt','w',encoding='utf8') as fo:
        for v in res:
            fo.write(str(v)+'\n')

    BagOfWordsExtractor_10k = CountVectorizer(max_features=10000,
                                              analyzer='word',
                                              lowercase=True)
    bow_features_10k = BagOfWordsExtractor_10k.fit(data['review'].tolist())
    vocab_10k = bow_features_10k.vocabulary_
    print('bow features vocab 10k:')
    print(len(vocab_10k))
    vocab_10k = sorted(vocab_10k.items(),key=lambda item:item[1])
    res = [x[0] for x in vocab_10k]
    with open(name + '10kVocab.txt','w',encoding='utf8') as fo:
        for v in res:
            fo.write(str(v)+'\n')

build_vocab('Zip')
    # print(bow_features)
    # BOW_Review = generateBOW(data['review'].tolist(), maxFeatures)
    # df_BOW_Review = pd.DataFrame(np.matrix(BOW_Review))