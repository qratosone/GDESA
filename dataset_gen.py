import torch
from torch.utils.data import Dataset


class DictDataset(Dataset):
    # 初始化
    def __init__(self, dict_list,implicit=False):
        # 读入数据
        self.data = dict_list
        self.implicit=implicit

    # 返回df的长度
    def __len__(self):
        return len(self.data)

    # 获取第idx+1列的数据
    def __getitem__(self, idx):
        #{'arr_neg': arr_neg, 'arr_pos': arr_pos, 'weight': weight, 'pad_seq': pad_seq, 'querys': querys}
        tensor_neg=torch.from_numpy(self.data[idx]['arr_neg']).float()
        tensor_neg.requires_grad=False
        tensor_pos = torch.from_numpy(self.data[idx]['arr_pos']).float()
        tensor_pos.requires_grad = False
        #print(self.data[idx]['weight'])
        weight = torch.FloatTensor([self.data[idx]['weight']])
        weight.requires_grad = False
        mask_doc=torch.ByteTensor(self.data[idx]['pad_seq'])
        mask_doc.requires_grad=False
        query=self.data[idx]['querys']
        if self.implicit:
            query=query[0]
        query.requires_grad=False
        mask_query=(query!=1)
        mask_query.requires_grad=False
        sum_queries=torch.sum(mask_query==1).numpy()-1
        if sum_queries!=0:
            subquery_weight=1.0/sum_queries
        else:
            subquery_weight=0.1
        query_weights=[1]+[subquery_weight]*10#max_subtopic=10
        query_weights=torch.Tensor(query_weights)
        return tensor_neg,tensor_pos,query,weight,mask_doc,mask_query,query_weights

