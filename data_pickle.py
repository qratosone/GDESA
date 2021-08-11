import pickle
from torch.utils.data import Dataset, DataLoader
current_filename='data_cv/train_sample.data'
train_sample = pickle.load(open(current_filename, 'rb'))
ts = pickle.load(open('data/test_sample_dq.data', 'rb'))
from type import DiversityQuery
DiversityQuery.load_alpha_nDCG_global_best('data/best_alpha_nDCG.data')
test_samples = pickle.load(open('data_cv/test_sample.data', 'rb'))
longest_context_length=0
class Config(object):
    cell_type = 'vanilla' # rnn cell type ('vanilla', 'LSTM', 'GRU')
    interaction = 'general' # interaction type ('general', 'dot')
    n_rel_feat = 18 # number of relevance feature
    n_doc_emb = 25 # dimension of document embedding
    n_query_emb = 25 # dimension of query embedding
    hidden_size = 10 # rnn hidden layer size
    # don't need to set the follow 3 params manually, they will be calculated from train samples
    most_n_doc = None # maximum context document sequence length
    most_n_pair = None # maximum pair number per context
    most_n_subtopic = None # maximum subtopic number per query
    lambdaa = 0.5 # trade-off between relevance score and diversity score
    learning_rate = 0.001 # learning rate
    n_epochs = 10 # iteration number
class CvConfig(object):
    cell_type = 'LSTM' # rnn cell type ('vanilla', 'LSTM', 'GRU')
    interaction = 'general' # interaction type ('general', 'dot')
    n_rel_feat = 18 # number of relevance feature
    n_doc_emb = 100 # dimension of document embedding
    n_query_emb = 100 # dimension of query embedding
    hidden_size = 50 # rnn hidden layer size
    # don't need to set the follow 3 params manually, they will be calculated from train samples
    most_n_doc = None # maximum context document sequence length
    most_n_pair = None # maximum pair number per context
    most_n_subtopic = None # maximum subtopic number per query
    lambdaa = 0.5 # trade-off between relevance score and diversity score
    learning_rate = 0.001 # learning rate
    n_epochs = 10 # iteration number
config=Config()
def pad_sample(sample:list,delta:int):
    sample.extend(['pad_doc' for i in range(delta)])
for qid in train_sample:
    #print('query is {}'.format(qid))
    for sample in train_sample[qid]:
        #print('context is {}'.format(sample[0]))
        if len(sample[0])>longest_context_length:
            longest_context_length=len(sample[0])
        #for pair in sample[1]:
            #print('pair is {}>{} with weight {}'.format(pair[1], pair[0], pair[2]))
print("actual longest context length:",longest_context_length)
longest_context_length=24
sample_dic={}
for qid in train_sample:
    if qid not in sample_dic.keys():
        sample_dic[qid]=[]
    for sample in train_sample[qid]:
        context=sample[0]
        for pair in sample[1]:
            seq_pos=[]
            seq_pos.extend(context)
            seq_pos.append(pair[1]) #negative 0 positive 1
            #seq_pos.append(pair[0])
            real_length=len(seq_pos)
            if len(seq_pos)<(longest_context_length+1):
                pad_sample(seq_pos,longest_context_length+1-len(seq_pos))
            assert len(seq_pos)==longest_context_length+1
            seq_neg=[]
            seq_neg.extend(context)
            seq_neg.append(pair[0])
            #seq_neg.append(pair[1])
            if len(seq_neg)<longest_context_length+1:
                pad_sample(seq_neg,longest_context_length+1-len(seq_neg))
            assert len(seq_neg)==longest_context_length+1
            pad_seq=[1]*real_length+[0]*(longest_context_length+1-real_length)
            assert len(pad_seq)==longest_context_length+1
            sample_dic[qid].append([seq_neg,seq_pos,pair[2],pad_seq])

import numpy as np
import pandas as pd


from prep import load_query_suggestion, load_emb



from public_tools import *
from sklearn.preprocessing import StandardScaler
doc_emb = load_emb('data_cv/doc_BERT.emb')

query_emb = load_emb('data_cv/query_BERT.emb')
rel_feat = pd.read_csv('data_cv/rel_feat.csv')
rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values),
                    rel_feat[rel_feat_names].values.tolist()))
suggestion = load_query_suggestion('data/query_suggestion.xml')
most_n_subtopic = max([len(suggestion[qid][1]) for qid in suggestion]) + 1
subtopic_list=[]
import torchtext
for qid in sample_dic.keys():
    subtopic=get_subtopics(qid,suggestion,query_emb,most_n_subtopic)
    subtopic_list.append(subtopic)
for qid in test_samples.keys():
    subtopic = get_subtopics(qid, suggestion, query_emb, most_n_subtopic)
    subtopic_list.append(subtopic)
print('querys in train:',len(sample_dic.keys()))
print('querys in test:',len(test_samples.keys()))
ST_FIELD=torchtext.legacy.data.Field(sequential=True,lower=False)

vectors = torchtext.vocab.Vectors(name='data_cv/query_vector.emb')
ST_FIELD.build_vocab(subtopic_list, vectors=vectors)

sample_dic_full={}

import json
item_list=list(SUBTOPIC.keys())
#with open("subtopics.json",'w') as f:
    #json.dump(SUBTOPIC,f)
#exit(0)
subtopic_proc_list=[SUBTOPIC[item] for item in item_list]

subtopic_proc_list=ST_FIELD.process(subtopic_proc_list)
subtopic_proc_list=subtopic_proc_list.permute(1,0)

subtopic_dic_proc={}
for i in range(len(item_list)):
    subtopic_dic_proc[item_list[i]]=subtopic_proc_list[i]

print('building...')

debug_flag=False
count=0
import os
if not os.path.exists('pickle_bert_subtopic'):
    os.mkdir('pickle_bert_subtopic')
print('data file:',current_filename)
import gzip
for qid in sample_dic.keys():
    #subtopic_emb_list=[]
    #subtopic_num.append(len(subtopic))
    querys =subtopic_dic_proc[qid]
    sample_dic_list=[]
    print('current qid:',qid)
    for sample in sample_dic[qid]:
        arr_neg=docid_to_emb_array(qid,sample[0],most_n_subtopic,doc_emb=doc_emb,rel_feat=rel_feat)
        arr_pos=docid_to_emb_array(qid,sample[1],most_n_subtopic,doc_emb=doc_emb,rel_feat=rel_feat)
        weight=sample[2]
        pad_seq=sample[3]
        #import pdb;pdb.set_trace()
        sample_dic_simple={'arr_neg':arr_neg,'arr_pos':arr_pos,'weight':weight,'pad_seq':pad_seq,'querys':querys}
        sample_dic_list.append(sample_dic_simple)
    filepath='pickle_bert_subtopic/'+str(qid)+'.pkl.gz'
    with gzip.open(filepath,'wb') as f:
        pickle.dump(sample_dic_list,f)
    count+=1
    print('saved:',count)

print('all pickles saved')
