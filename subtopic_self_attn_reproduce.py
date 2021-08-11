import numpy as NP
import torch
from torch import nn
from torch.nn import functional as F
from transformer_block import TransformerEncoder,TransformerDecoder
from torch.nn import init
from public_tools import N_REL_FEATURES

# from TransOrigin import StarTransformer as StarTransOrigin
max_subtopic = 10

class LSTM_select(nn.Module):
    def __init__(self,embedding_length,hidden_size):
        super(LSTM_select, self).__init__()
        self.lstm_cell=nn.LSTM(embedding_length,hidden_size,batch_first=True)
    def forward(self,input,predict_hidden=None):
        #predict_hidden=[h_last,c_last]
        batch_size=input.shape[0]

        if predict_hidden is not None:
            hidden_size = predict_hidden[0].shape[1]
            init_hidden=(predict_hidden[0].unsqueeze(0).expand(1,batch_size,hidden_size).contiguous(),predict_hidden[1].unsqueeze(0).expand(1,batch_size,hidden_size).contiguous())
            #print(init_hidden[0].is_contiguous())
            output,out_hidden=self.lstm_cell(input,init_hidden)
        else:
            output,out_hidden=self.lstm_cell(input)
        return output, out_hidden
    def predict(self,input,predict_hidden):
        #switch input into parallel computing
        #input:[batch=1,seq,embedding_length]
        #predict_hidden:[1,hidden_size]
        input_swap=input.permute(1,0,2)
        assert input_swap.shape[1]==1
        output_single,hidden=self.forward(input_swap,predict_hidden)
        return output_single,hidden



class Query_Self_Attn(nn.Module):
    """
        :param int hidden_size: 输入维度的大小。同时也是输出维度的大小。
        :param int num_layers: star-transformer的层数
        :param int num_head: head的数量。
        :param int head_dim: 每个head的维度大小。
        :param float dropout: dropout 概率. Default: 0.1
        :param int max_len: int or None, 如果为int，输入序列的最大长度，
            模型会为输入序列加上position embedding。
            若为`None`，忽略加上position embedding的步骤. Default: `None`
        """

    def __init__(self, doc_emb_length,
                 query_dic_length,
                 query_emb_length,
                 hidden_size,
                 num_layers,
                 num_head,
                 max_len,
                 dropout=0.1):
        super(Query_Self_Attn, self).__init__()
        self.embedding_length = query_emb_length
        self.features_total = doc_emb_length - self.embedding_length
        self.linear_features_single = nn.Linear(N_REL_FEATURES, 1,bias=False)
        #self.linear_querys = nn.Linear(max_subtopic + 1, 1)
        # self.linear_score_subtopic=nn.Linear(max_subtopic,1)
        init.xavier_normal_(self.linear_features_single.weight)
        # init.xavier_normal_(self.linear_score_subtopic.weight)

        self.query_embed = nn.Embedding(query_dic_length, query_emb_length)
        self.projection_encoder = nn.Linear(hidden_size, 160)
        init.xavier_normal_(self.projection_encoder.weight)
        self.num_head = num_head
        print("num head:", self.num_head)
        self.encoder_query = TransformerEncoder(
            num_layers=num_layers-1, hidden_size=160, num_head=num_head, d_ff=4 * hidden_size, dropout=dropout,max_len=max_len
        )
        self.decoder_doc=TransformerDecoder(
            num_layers=1, hidden_size=160, num_head=8, d_ff=4 * hidden_size, dropout=dropout,max_len=max_len
        )
        for p in self.encoder_query.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

        self.time_counter=0
        self.linear_score_attention= nn.Linear(160, 1)
        self.score_final = nn.Linear(160+160 + max_subtopic+N_REL_FEATURES+50, 1)
        init.xavier_normal_(self.linear_score_attention.weight)
        init.xavier_normal_(self.score_final.weight)

        self.lstm_context=LSTM_select(100,50)
        self.lstm_hidden=None
        for p in self.lstm_context.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

        self.tags = 'new framework;original transformer;pos emb'
        #self.lambda_r=0.5
        print(self.tags)

    def forward(self, seq_doc, seq_query, mask_doc, mask_query, predict=False,predict_context=None):
        """

        :param seq_doc: [batch,doc_dim,length] ->[batch,length,doc_dim]
        :param seq_query: [batch,max_query]
        :param mask_doc: [batch,length]
        :param mask_query:[batch,max_query]
        :return:
        """
        import time
        start_time=time.time()
        hidden_doc = seq_doc[:, :, self.features_total:]
        feat_doc_query_single=seq_doc[:, :, :N_REL_FEATURES]
        feat_doc_querys = seq_doc[:, :, N_REL_FEATURES:self.features_total]
        feat_doc_querys = feat_doc_querys.view(feat_doc_querys.size()[0], feat_doc_querys.size()[1], max_subtopic,
                                               N_REL_FEATURES)
        feat_doc_query_scores = self.linear_features_single(feat_doc_querys)
        feat_doc_query_scores = feat_doc_query_scores.squeeze(3)
        #print(feat_doc_query_scores.shape)
        #print('mark')
        #hidden_feat_querys,_=self.encoder_subtopic(feat_doc_query_scores,mask_doc)
        #feat_doc_score=self.linear_querys(feat_doc_query_scores)
        seq_query = seq_query[:,1:]
        mask_query=mask_query[:,1:]
        hidden_query = self.query_embed(seq_query)
        

        
        # feat_doc_querys_lstm = seq_doc[:, :, self.embedding_length:self.embedding_length+100]
        # hidden_query=emb_query#self.input_query(emb_query)
        # hidden_doc=hidden_doc.permute(1,2)#[batch,length,hidden]#进来的时候已经是[batch,length,doc_dim]了  # [batch,length,hidden]
        # hidden_doc_lstm, _ = self.lstm_test(feat_doc_querys_lstm)
        hidden_doc_proj=self.projection_encoder(hidden_doc)
        hidden_doc_seq_attn = self.encoder_query(hidden_doc_proj, mask_doc)  # , hidden_query, mask_query)
        # hidden_doc_seq_attn = hidden_doc_seq_attn + hidden_doc_lstm
        # doc_score_seq=self.score(full_cat_query_attn[:,:doc_length,:])
        hidden_query_proj=self.projection_encoder(hidden_query)
        hidden_query_seq_attn=self.encoder_query(hidden_query_proj, mask_query,posenc=True,predict=predict)
        hidden_doc_query_attn=self.decoder_doc(hidden_doc_seq_attn,hidden_query_seq_attn,posenc=False)
        mask_doc_expand=mask_doc.unsqueeze(2).expand(hidden_doc_query_attn.shape)
        #print(mask_doc)
        #print(mask_doc_expand[:,:5,:])
        #print(hidden_doc_query_attn.shape)
        hidden_doc_query_attn=hidden_doc_query_attn.masked_fill_(mask_doc_expand==0,0)
        dot_scores = self.linear_score_attention(hidden_query_seq_attn)
        #print(dot_scores.shape)
        #dot_scores=dot_scores.masked_fill_(mask_query==0,-1e9)
        #print(feat_doc_query_scores)
        #print(dot_scores)
        
        #dot_scores=dot_scores.masked_fill_(mask_query==0,-1e9)
        dot_scores=F.softmax(dot_scores,dim=1)
        #print(dot_scores)
        #print(dot_scores.shape)
        #print(feat_doc_query_scores.shape)
        feat_doc_query_scores*=dot_scores.permute(0,2,1)
        #feat_doc_query_scores*=(1.0/max_subtopic)
        stat_end_time=time.time()
        if self.time_counter==0:
            self.time_counter+=(stat_end_time-start_time)
        
        if predict is True:
            hidden_doc_lstm, self.lstm_hidden = self.lstm_context.predict(hidden_doc,predict_context)
            hidden_doc_lstm=hidden_doc_lstm.permute(1,0,2)
        else:
            hidden_doc_lstm, self.lstm_hidden = self.lstm_context(hidden_doc)
        #attn_doc_score = self.linear_score_attention(hidden_doc_seq_attn)
        # print(attn_doc_score.shape)
        doc_score_seq = self.score_final(torch.cat([feat_doc_query_scores,hidden_doc_seq_attn,hidden_doc_query_attn,hidden_doc_lstm, feat_doc_query_single], 2))
        dyn_end_time=time.time()
        #self.time_counter+=(dyn_end_time-stat_end_time)
        if not predict:
            return torch.sum(doc_score_seq, 1)
        else:
            return torch.tanh(doc_score_seq)


def testmodel():
    fakeBATCH = 1
    fakeDOCdim = 298
    fakeQUERYdim = 100
    fakeQUERYlength = 11
    fakeQUERYdicLength = 100
    fakeHidden = 100
    LENGTH = 12
    fake_doc = torch.FloatTensor(fakeBATCH, LENGTH, fakeDOCdim)
    fake_query = torch.ones(fakeBATCH, fakeQUERYlength).long()
    pad_doc = torch.ones(fakeBATCH, LENGTH)
    pad_doc[:, 4:] = 0
    mask_doc = torch.ByteTensor(fakeBATCH,LENGTH)

    pad_query = torch.ones(fakeBATCH, fakeQUERYlength)
    pad_query[:, 2:] = 0
    mask_query = torch.ByteTensor(fakeBATCH,fakeQUERYlength)
    print('doc:', fake_doc.shape)
    print('query:', fake_query.shape)
    model = Query_Self_Attn(
        doc_emb_length=fakeDOCdim,
        query_dic_length=fakeQUERYdicLength,
        query_emb_length=fakeQUERYdim,
        hidden_size=fakeHidden,
        num_layers=1,
        num_head=8,
        max_len=LENGTH,
        dropout=0.1
    )
    fake_doc.requires_grad = False
    fake_query.requires_grad = False
    mask_doc.requires_grad = False
    mask_query.requires_grad = False
    score = model(fake_doc, fake_query, mask_doc, mask_query,predict=True)
    max_index=5
    print('hidden:',model.lstm_hidden[0].shape)
    selected_hidden_h=model.lstm_hidden[0][:,max_index,:]
    selected_hidden_c = model.lstm_hidden[1][:, max_index, :]
    score=model(fake_doc,fake_query,mask_doc,mask_query)
    print('score:', score.shape)
    print('dimension check passed!')

def test_RNN():
    fake_doc=torch.Tensor(1,10,100)
    lstm_selection=LSTM_select(100,80)
    fake_doc=fake_doc.permute(1,0,2)
    print("doc:",fake_doc.shape)
    doc_lstm,hidden_lstm=lstm_selection(fake_doc)
    print(doc_lstm.shape)
    selected_index_list=[i for i in range(10)]
    for j in selected_index_list:
        selected_index=0
        selected_doc_h=hidden_lstm[0][:,selected_index,:]
        selected_doc_c = hidden_lstm[1][:, selected_index, :]
        #print(selected_doc_h.shape)
        fake_doc = fake_doc[torch.arange(fake_doc.size(0)) != selected_index,:,:]
        print("selected",fake_doc.shape)
        doc_lstm_iter,hidden_lstm_iter=lstm_selection(fake_doc,(selected_doc_h,selected_doc_c))
        print(doc_lstm_iter.shape,hidden_lstm_iter[0].shape)
if __name__ == "__main__":
    #test_RNN()
    testmodel()



