import pickle
from torch.utils.data import Dataset, DataLoader
train_sample = pickle.load(open('data_cv/train_sample.data', 'rb'))
ts = pickle.load(open('data/test_sample_dq.data', 'rb'))
from type import DiversityQuery
DiversityQuery.load_alpha_nDCG_global_best('data/best_alpha_nDCG.data')
test_samples = pickle.load(open('data_cv/test_sample.data', 'rb'))
longest_context_length=0
SEED = 2017
import torch
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def pad_sample(sample:list,delta:int):
    sample.extend(['pad_doc' for i in range(delta)])
for qid in train_sample:
    for sample in train_sample[qid]:
        if len(sample[0])>longest_context_length:
            longest_context_length=len(sample[0])
print("actual longest context length:",longest_context_length)
longest_context_length=50-1

import pandas as pd


from prep import load_query_suggestion, load_emb



from public_tools import *
from sklearn.preprocessing import StandardScaler
doc_emb = load_emb('data_cv/doc.emb')
query_emb = load_emb('data_cv/query.emb')
rel_feat = pd.read_csv('data_cv/rel_feat.csv')
rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values),
                    rel_feat[rel_feat_names].values.tolist()))
suggestion = load_query_suggestion('data/query_suggestion.xml')
most_n_subtopic = max([len(suggestion[qid][1]) for qid in suggestion]) + 1
subtopic_list=[]
import torchtext
for qid in train_sample.keys():
    subtopic=get_subtopics(qid,suggestion,query_emb,most_n_subtopic)
    subtopic_list.append(subtopic)
for qid in test_samples.keys():
    subtopic = get_subtopics(qid, suggestion, query_emb, most_n_subtopic)
    subtopic_list.append(subtopic)
ST_FIELD=torchtext.legacy.data.Field(sequential=True,lower=False)

vectors = torchtext.vocab.Vectors(name='data_cv/query_vector.emb')
ST_FIELD.build_vocab(subtopic_list, vectors=vectors)




item_list=list(SUBTOPIC.keys())

subtopic_proc_list=[SUBTOPIC[item] for item in item_list]

subtopic_proc_list=ST_FIELD.process(subtopic_proc_list)
subtopic_proc_list=subtopic_proc_list.permute(1,0)

subtopic_dic_proc={}
for i in range(len(item_list)):
    subtopic_dic_proc[item_list[i]]=subtopic_proc_list[i]

BATCH_SIZE=256
EPOCH=10
DOC_EMBED_LENGTH=298
QUERY_EMBED_LENGTH=100
HIDDEN_SIZE=100
LAYERS=3
LR=0.0008
LENGTH=50
DROPOUT=0.3
DIR_PATH="pickle_train/"
from dataset_gen import DictDataset
import gzip
def get_dataloader(filename_list:list,dir_path="pickle_train/"):
    data_list=[]
    for item in filename_list:
        filename_full=dir_path+item+'.pkl.gz'
        with gzip.open(filename_full,'rb') as f:
            dic_list=pickle.load(f)
        data_list.extend(dic_list)
    dataset=DictDataset(data_list)
    loader=DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return loader
all_qids=np.load('all_qids.npy')
print('all qids:',len(all_qids))
print('numpy shuffle:',all_qids)



print('data loaded')

import pickle


import time
time_start=time.time()
time_prev=time_start
print("start time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
debug_flag=True
metrics = []


from sklearn.model_selection import KFold
import datetime
#path_archive='reproduce_lstm'
#path_archive='layer2_bert'
#path_archive="combine_overall/"
path_archive="./"
dropout_search_result=[]
avg_metrics=[]
avg_implicit=[]
reproduce=1
for i in range(reproduce):
    fold=0
    final_metrics=[]
    final_metrics_implicit=[]
    full_result={}
    full_result_implicit={}
    #attention_dict={}
    overall_time=0
    for train_idx,test_idx in KFold(5).split(all_qids):
        fold+=1
        train_idx.sort()
        test_idx.sort()
        train_qids=[all_qids[i] for i in train_idx]
        test_qids=[all_qids[i] for i in test_idx]
        print("Fold ",str(fold),' ...')
        #model_filename='fold_'+str(fold)+'layer_'+str(LAYERS)+'_dropout_'+str(DROPOUT)+'.pkl'
        #model_filename='base_fold_'+str(fold)+'layer_'+str(LAYERS)+'_dropout_'+str(DROPOUT)+'lstm-attn-finegrain.pkl'
        model_filename='base_fold_'+str(fold)+'layer_'+str(LAYERS)+'_dropout_'+str(DROPOUT)+'lstm-attn-dotscore-finegrained.pkl'
        from subtopic_self_attn_reproduce import Query_Self_Attn
        model=Query_Self_Attn(
            doc_emb_length=DOC_EMBED_LENGTH,
            query_dic_length=len(ST_FIELD.vocab),
            query_emb_length=QUERY_EMBED_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            num_head=8,
            max_len=LENGTH,
            dropout=DROPOUT
        )
        #model.query_embed.weight.requires_grad=False
        #model.query_embed.weight.copy_(ST_FIELD.vocab.vectors)
        model_path=path_archive+'/'+model_filename
        print(model_path)
        if torch.cuda.is_available():
            model=model.cuda()
            model.load_state_dict(torch.load(model_path))
        else:
            print("using cpu for prediction")
            model.load_state_dict(torch.load(model_path,map_location=torch.device("cpu")))
        
        
        model.eval()
        last_qid=''
        print('trained model loaded')
        print('test in val qids...')
        metrics = []
        
        for item in test_qids:
            if item in test_samples.keys():
                
                ls =[doc for doc in test_samples[item]]
                dq = ts[item]
                
                map_dic={}
                for i in range(len(ls)):
                    map_dic[ls[i]]=i
                
                #random.shuffle(ls)
                model.time_counter=0
                result_list = get_ranking_selection_delete(model, item, ls, most_n_subtopic=most_n_subtopic, doc_emb=doc_emb,
                                            rel_feat=rel_feat, subtopic_dic_proc=subtopic_dic_proc)
                overall_time+=model.time_counter
                revert_result_list=[]
                for result in result_list:
                    doc_id=map_dic[ls[result[0]]]
                    revert_result_list.append([doc_id,result[0],result[1]])

                #attention_dict[item]=[]
                #attention_dict[item].append(ls)
                #attention_dict[item].append(model.attn_doc.cpu().numpy())
                #attention_dict[item].append(model.attn_query.cpu().numpy())
                ranks = []
                full_result[item]=[]
                for result in revert_result_list:
                    ranks.append(int(result[0]))
                    full_result[item].append([ls[result[1]],result[2].tolist()])  
                try:
                    ndq = DiversityQuery(dq.query, dq.qid, dq.subtopics, dq.docs[ranks])
                except Exception as e:
                    print(ranks)
                    raise e
                metrics.append(ndq.get_metric('alpha_nDCG'))
                last_qid=item
        print('{}:{:>5.3f}'.format('test', np.mean(metrics)))
        
        final_metrics.append(np.mean(metrics))
            
    print('All folds finished at layers:',LAYERS,'dropout:',DROPOUT)
    print('Final metrics:',np.mean(final_metrics))
    #with gzip.open('attention.pkl.gz','wb') as f:
        #pickle.dump(attention_dict,f)
    import json
    with open('result_selection_decoder.json','w',encoding='utf-8') as f:
        json.dump(full_result,f)
    avg_metrics.append(np.mean(final_metrics))
print("Full reproduce test at:",reproduce)
np.set_printoptions(4)
print("metrics:",np.mean(avg_metrics))
print("overall_time:",overall_time)
#print("metrics implicit:",np.mean(avg_implicit))
        
 
            

