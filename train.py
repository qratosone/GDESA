import datetime
import gzip
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torchtext
from dataset_gen import DictDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from prep import load_emb, load_query_suggestion
from public_tools import *
from type import DiversityQuery

train_sample = pickle.load(open('data_cv/train_sample.data', 'rb'))
ts = pickle.load(open('data/test_sample_dq.data', 'rb'))
DiversityQuery.load_alpha_nDCG_global_best('data/best_alpha_nDCG.data')
test_samples = pickle.load(open('data_cv/test_sample.data', 'rb'))
longest_context_length = 0
SEED = 2017
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def pad_sample(sample: list, delta: int):
    sample.extend(['pad_doc' for i in range(delta)])


for qid in train_sample:
    for sample in train_sample[qid]:
        if len(sample[0]) > longest_context_length:
            longest_context_length = len(sample[0])
print("actual longest context length:", longest_context_length)
longest_context_length = 50-1


doc_emb = load_emb('data_cv/doc.emb')
query_emb = load_emb('data_cv/query.emb')
rel_feat = pd.read_csv('data_cv/rel_feat.csv')
rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values),
                    rel_feat[rel_feat_names].values.tolist()))
suggestion = load_query_suggestion('data/query_suggestion.xml')
most_n_subtopic = max([len(suggestion[qid][1]) for qid in suggestion]) + 1
subtopic_list = []
for qid in train_sample.keys():
    subtopic = get_subtopics(qid, suggestion, query_emb, most_n_subtopic)
    subtopic_list.append(subtopic)
for qid in test_samples.keys():
    subtopic = get_subtopics(qid, suggestion, query_emb, most_n_subtopic)
    subtopic_list.append(subtopic)
ST_FIELD = torchtext.legacy.data.Field(sequential=True, lower=False)

# ,cache=".vector_cache_bert")
vectors = torchtext.vocab.Vectors(name='data_cv/query_vector.emb')
ST_FIELD.build_vocab(subtopic_list, vectors=vectors)


item_list = list(SUBTOPIC.keys())

subtopic_proc_list = [SUBTOPIC[item] for item in item_list]

subtopic_proc_list = ST_FIELD.process(subtopic_proc_list)
subtopic_proc_list = subtopic_proc_list.permute(1, 0)

subtopic_dic_proc = {}
for i in range(len(item_list)):
    subtopic_dic_proc[item_list[i]] = subtopic_proc_list[i]

BATCH_SIZE = 256
EPOCH = 10
DOC_EMBED_LENGTH = 298
QUERY_EMBED_LENGTH = 100
HIDDEN_SIZE = 100
LAYERS = 3
LR = 0.0008
LENGTH = 50
DROPOUT = 0.3
DIR_PATH = "pickle_train/"


def get_dataloader(filename_list: list, dir_path="pickle_train/"):
    data_list = []
    for item in filename_list:
        filename_full = dir_path+item+'.pkl.gz'
        with gzip.open(filename_full, 'rb') as f:
            dic_list = pickle.load(f)
        data_list.extend(dic_list)
    dataset = DictDataset(data_list)
    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return loader


all_qids = np.load('all_qids.npy')
print('all qids:', len(all_qids))
print('numpy shuffle:', all_qids)


print('data loaded')


time_start = time.time()
time_prev = time_start
print("start time:", time.strftime(
    '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


t = datetime.datetime.now()
filename = t.strftime("best_%m%d%H%M%S")+'.json'
path = './'
filename = path+filename

fold = 0
final_metrics = []
for train_idx, test_idx in KFold(5).split(all_qids):
    fold += 1
    train_idx.sort()
    test_idx.sort()
    train_qids = [all_qids[i] for i in train_idx]
    test_qids = [all_qids[i] for i in test_idx]
    print("Fold ", str(fold), ' ...')
    model_filename = 'base_fold_' + \
        str(fold)+'layer_'+str(LAYERS)+'_dropout_'+str(DROPOUT)+'.pkl'
    from subtopic_self_attn_reproduce import Query_Self_Attn
    model = Query_Self_Attn(
        doc_emb_length=DOC_EMBED_LENGTH,
        query_dic_length=len(ST_FIELD.vocab),
        query_emb_length=QUERY_EMBED_LENGTH,
        hidden_size=HIDDEN_SIZE,
        num_layers=LAYERS,
        num_head=8,
        max_len=LENGTH,
        dropout=DROPOUT
    )
    model.query_embed.weight.requires_grad = False
    model.query_embed.weight.copy_(ST_FIELD.vocab.vectors)
    taginfo = model.tags
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    print('model reloaded')
    metrics_max = 0
    print('metrics_reloaded')
    for i in range(EPOCH):
        loss_list = []
        accu_list = []
        epoch_step = 0
        pickle_step = 20
        for j in range(0, len(train_qids), pickle_step):
            train_pkl_list = train_qids[j:j+pickle_step]
            train_pkl_list = [str(i) for i in train_pkl_list]
            pkl_loader = get_dataloader(train_pkl_list, dir_path=DIR_PATH)
            model.train()
            #print('pkl loaded:',[train_pkl_list])
            for step, (batch_tensor_neg, batch_tensor_pos, batch_query, batch_weight, batch_mask_doc, batch_mask_query, batch_query_weights) in enumerate(pkl_loader):
                if torch.cuda.is_available():
                    batch_tensor_neg = batch_tensor_neg.cuda()
                    batch_tensor_pos = batch_tensor_pos.cuda()
                    batch_query = batch_query.cuda()
                    batch_weight = batch_weight.cuda()
                    batch_mask_doc = batch_mask_doc.cuda()
                    batch_mask_query = batch_mask_query.cuda()
                   # batch_query_weights=batch_query_weights.cuda()
                result_pos = model(batch_tensor_pos, batch_query,
                                   batch_mask_doc, batch_mask_query)
                result_neg = model(batch_tensor_neg, batch_query,
                                   batch_mask_doc, batch_mask_query)
                accu, loss = loss_function(
                    result_pos, result_neg=result_neg, weights=batch_weight)
                loss_list.append(loss.cpu().data.numpy())
                accu_list.append(accu.cpu().data.numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('\rstep:', epoch_step, 'current accu:', np.mean(
                    accu_list), 'current loss:', np.mean(loss_list), end="")
                epoch_step += 1
        print('\nEPO{},time{}'.format(i, epoch_step, time.time() - time_prev))
        print("loss:", np.mean(loss_list))
        print("accu:", np.mean(accu_list))
        time_prev = time.time()
        metrics = []

        print('test in val qids...')
        model.eval()
        for item in test_qids:
            if item in test_samples.keys():
                ls = test_samples[item]
                dq = ts[item]
                result_list = get_ranking_selection_delete(model, item, ls, most_n_subtopic=most_n_subtopic, doc_emb=doc_emb,
                                          rel_feat=rel_feat, subtopic_dic_proc=subtopic_dic_proc)
                ranks = []
                for result in result_list:
                    ranks.append(int(result[0]))
                try:
                    ndq = DiversityQuery(
                        dq.query, dq.qid, dq.subtopics, dq.docs[ranks])
                except Exception as e:
                    print(ranks)
                    raise e
                metrics.append(ndq.get_metric('alpha_nDCG'))
        print('{}:{:>5.3f}'.format('test', np.mean(metrics)))
        if np.mean(metrics) > metrics_max:
            metrics_max = np.mean(metrics)
            print('max metrics updated:', metrics_max)
            torch.save(model.state_dict(), model_filename)
            print('save file at:', model_filename)
        else:
            pass
        if i == EPOCH-1:
            final_metrics.append(metrics_max)
            print(f'got final metrics for fold {fold}:{metrics_max}')
print('All folds finished at layers:', LAYERS, 'dropout:', DROPOUT)
print('Final metrics:', np.mean(final_metrics))
