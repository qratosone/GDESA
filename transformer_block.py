import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import copy
import numpy as np
from torch.autograd import Variable
def clones(module, N):
    "生成N个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query,key,value,mask=None,dropout=None):
    "计算'可缩放点乘注意力'"
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
    #print('querys:',query.shape)
    #print('scores:',scores.shape)
    #print('mask:',mask.shape)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    #print(scores)
    p_attn = F.softmax(scores,dim = -1)
    if dropout is  not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn
class SublayerConnection(nn.Module):
    """
    层归一化之后的残差连接。
    注意：为了简化代码，归一化是第一个，而不是最后一个。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        "将残差连接应用于相同大小的任何子层。"
        return x+self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "构建层归一化模块"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class Encoder(nn.Module):
    "核心编码器是N层堆叠"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "依次将输入的数据（及屏蔽数据）通过每个层"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
def subsequent_mask(size):
    "屏蔽后续位置"
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
def subsequent_mask_fake(size):
    attn_shape = (1,size,size)
    #print("fake mask")
    subsequent_mask = np.ones(attn_shape).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
class EncoderLayer(nn.Module):
    "编码器由以下的自注意力和前馈网络组成"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "按照论文中的图1（左）的方式进行连接"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PositionalEncoding(nn.Module):
    "实现位置编码函数"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2) * -math.log(10000) / d_model).float())

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
class PositionwiseFeedForward(nn.Module):
    "实现FFN方程"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "设置模型大小和注意力头部数量"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假设 d_v 等于 d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 对应 Q,K,V 3次线性变换 + 最终的1次线性变换
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "实现论文中的第2张图"
        if mask is not None:
            # 同样的屏蔽适用于所有h型头
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1)批量执行所有线性变换 d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2）将注意力集中在批量的所有投射向量上
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3)使用view方法做Concat然后做最终的线性变换。
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers , hidden_size,  num_head, d_ff,dropout,max_len=50):
        super(TransformerEncoder, self).__init__()
        c=copy.deepcopy
        attn = MultiHeadedAttention(num_head, hidden_size)
        ff = PositionwiseFeedForward(hidden_size, d_ff, dropout)
        self.position = nn.Embedding(max_len, hidden_size)
        self.encoder=Encoder(EncoderLayer(hidden_size, c(attn), c(ff), dropout), num_layers)


    def forward(self, input,input_mask,posenc=True,predict=False):
        if not predict:
            input_seq_mask=Batch.make_std_mask(input_mask,pad=0)
        else:
            #print("use predict")
            input_seq_mask=Batch.make_std_mask_fake(input_mask,pad=0)
        #print('emb pos')
        B, L, H = input.size()
        #print('mask:',input_seq_mask)
        if posenc:
            P = self.position(torch.arange(L, dtype=torch.long, device=input.device).view(1, L))
            #print(P.size())
            input=input+P
        return self.encoder(input,input_seq_mask)
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.src_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
 
    def forward(self, x, memory):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, None))
        return self.sublayer[1](x, self.feed_forward)
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, doc, query):
        for layer in self.layers:
            x = layer(doc, query)
        return self.norm(x)
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers = 1, hidden_size = 100,  num_head = 2, d_ff = 400,dropout = 0.1,max_len=50):
        super(TransformerDecoder, self).__init__()
        c=copy.deepcopy
        attn = MultiHeadedAttention(num_head, hidden_size)
        ff = PositionwiseFeedForward(hidden_size, d_ff, dropout)
        self.decoder=Decoder(DecoderLayer(hidden_size, c(attn), c(ff), dropout), num_layers)


    def forward(self, input_doc,input_query,posenc=True,for_subtopic=False):
        #input_seq_mask=Batch.make_std_mask(input_mask,pad=0)
        #print('emb pos')
        B, L, H = input_doc.size()
        #print(input.size())
        #print('mask:',input_seq_mask)
        return self.decoder(input_doc,input_query)
class Batch:
    "此对象用于在训练时进行已屏蔽的批数据处理"

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
            print('trg shape:', self.trg.shape)
            print('trg mask shape:', self.trg_mask.shape)

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个mask来隐藏填充和将来的单词"
        tgt_mask = (tgt == pad).unsqueeze(-2)
        #print('input mask:', tgt)
        #print('tgt_mask:',tgt_mask)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    @staticmethod
    def make_std_mask_fake(tgt, pad):
        "创建一个mask来隐藏填充和将来的单词"
        tgt_mask = (tgt == pad).unsqueeze(-2)
        #print('input mask:', tgt)
        #print('tgt_mask:',tgt_mask)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask_fake(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
def data_gen(V,batch,nbatches):
    "为src-tgt复制任务生成随机数据"
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1,V,size = (batch,10)))
        data[:,0] =1
        src = Variable(data,requires_grad = False)
        tgt = Variable(data,requires_grad = False)
        yield Batch(src,tgt,0)
if __name__=="__main__":
    V=100
    model =TransformerDecoder(num_layers=1,hidden_size=100)
    handler=data_gen(V=V,batch=32,nbatches=11)
    batch=next(handler)
    d_model=100
    emb=nn.Embedding(V,d_model)
    src_emb=emb(batch.src.long())

    #out = model.forward(src_emb,batch.src_mask)
    #print(out.shape)
    query_length=11
    fakequery=torch.ones([1,query_length,d_model])
    doc_length=40
    fakedoc=torch.ones([1,doc_length,d_model])
    result,_=attention(fakedoc,fakequery,fakequery)
    print(result.shape)
