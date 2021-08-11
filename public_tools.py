import torch
import torch.nn.functional as F
import numpy as np
DOC_DICT = {}#储存某个文档对应的矩阵编号
SUBTOPIC = {}#编号对应子话题？
N_REL_FEATURES=18
MAX_LENGTH=50
def get_subtopics(qid, suggestion, query_emb, most_n_subtopic):#填充subtopics
    query = suggestion[qid][0]
    subtopics = [sug[0] for sug in suggestion[qid][1]]
    # put the query in the first place
    subtopics = [query] + sorted(set(subtopics) - {query})[:most_n_subtopic - 1]#包括query本身
    subtopics = [item.replace(' ','_') for item in subtopics]
    #print(subtopics)
    SUBTOPIC[qid] = subtopics
    return subtopics
def loss_function(result_pos:torch.FloatTensor,result_neg:torch.FloatTensor,weights):
    if torch.cuda.is_available():
        result_pos=result_pos.cuda()
        result_neg=result_neg.cuda()
        weights=weights.cuda()
    actual_batch=result_pos.shape[0]
    assert result_pos.shape==result_neg.shape
    #print("output shape:",result_pos.shape)
    total=torch.cat((result_pos,result_neg),1)
    #print(total.shape)
    prob=F.softmax(total,dim=1)
    assert prob.shape[0]==actual_batch
    assert prob.shape[1]==2
    target_prob=prob[:,0]
    #print("positive prob:",target_prob)
    target_prob_result=target_prob>0.5
    accuracy=torch.Tensor.sum(target_prob_result.float()/torch.Tensor.prod(torch.FloatTensor(list(target_prob.shape))))
    weights=weights.float()
    log_loss=-torch.Tensor.sum(weights*torch.Tensor.log(target_prob))
    return accuracy,log_loss

def get_doc_emb_single(qid,docid,doc_emb,rel_feat:dict,most_n_subtopic,cross_implicit=False):
    rel=[]
    if not cross_implicit:
        for subq in SUBTOPIC[qid]:
            rel.append(rel_feat[(subq.replace('_',' '),docid)])
        rel.append([0] * N_REL_FEATURES * (most_n_subtopic - len(SUBTOPIC[qid])))
    else:
        query=SUBTOPIC[qid][0]
        rel.append(rel_feat[(query.replace('_',' '),docid)])
        rel.append([0] * N_REL_FEATURES * (most_n_subtopic - 1))
    doc_emb_single=np.concatenate(rel+[doc_emb[docid]])
    
    return doc_emb_single
def docid_to_emb_array(qid,docid_list:list,most_n_subtopic,doc_emb,rel_feat,cross_implicit=False):
    doc_emb_list=[]
    doc_emb_single_length=None
    #print(docid_list)
    for docid in docid_list:
        #print(docid)
        if docid!='pad_doc':
            doc_emb_single=get_doc_emb_single(qid,docid,doc_emb,rel_feat,most_n_subtopic,cross_implicit)
            if doc_emb_single_length is None:
                doc_emb_single_length=len(doc_emb_single)
                #print("doc emb single length:",doc_emb_single_length)
            else:
                assert doc_emb_single_length==len(doc_emb_single)
            doc_emb_list.append(doc_emb_single)
        else:
            pad_doc=[0]*doc_emb_single_length
            doc_emb_list.append(pad_doc)
    doc_emb_array=np.vstack(doc_emb_list)
    #print(doc_emb_array.shape)
    return doc_emb_array



def get_ranking_selection_delete(model,qid,doc_list:list,most_n_subtopic,doc_emb,rel_feat,subtopic_dic_proc,implicit=False):
    actual_length = len(doc_list)
    doc_list_pad = [i for i in doc_list]
    arr_rank = docid_to_emb_array(qid, doc_list_pad, most_n_subtopic, doc_emb=doc_emb, rel_feat=rel_feat,
                                  cross_implicit=implicit)
    tensor_rank = torch.from_numpy(arr_rank).float()
    tensor_rank.requires_grad = False
    mask_doc = torch.ByteTensor([1] * actual_length)
    mask_doc.requires_grad = False
    query = subtopic_dic_proc[qid]
    if implicit:
        query[1:] = 1
    mask_query = (query != 1)
    tensor_rank = tensor_rank.unsqueeze(0)
    query = query.unsqueeze(0)
    mask_doc = mask_doc.unsqueeze(0)
    mask_query = mask_query.unsqueeze(0)
    if torch.cuda.is_available():
        tensor_rank = tensor_rank.cuda()
        query = query.cuda()
        mask_doc = mask_doc.cuda()
        mask_query = mask_query.cuda()
    result_list=[]
    #count=0
    context_lstm=None
    map_doclist_pad={}
    for i in range(len(doc_list_pad)):
        map_doclist_pad[doc_list_pad[i]]=i
    selected_index=[]
    with torch.no_grad():
        for i in range(actual_length):
            result = model(tensor_rank, query, mask_doc, mask_query, predict=True,predict_context=context_lstm)
            result=result.squeeze(2)
            result = result.squeeze(0)
            for index in selected_index:
                result[index]=-1e9
            score,index=torch.max(result,0)
            score = score.cpu().numpy()
            index = index.cpu().numpy()
            index_id=index.tolist()
            result_list.append([index_id,score])
            selected_index.append(index_id)
            lstm_select_h=model.lstm_hidden[0][:,index_id,:]
            lstm_select_c=model.lstm_hidden[1][:,index_id,:]
            context_lstm=[lstm_select_h,lstm_select_c]
    return result_list