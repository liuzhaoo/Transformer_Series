import torch
import torch.nn as nn
import math
from utils import masked_softmax
import numpy as np

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self,dropout):
        super(Scaled_Dot_Product_Attention,self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens=None):
        """
        :param queries: (batch_size,n_q,d)
        :param keys: (batch_size,n_k,d)
        :param values: (batch_size,n_v,d_v)
        :param valid_lens: 对当前步之后的数据进行mask (batch_size,) 或者(batch_size,n_q)
        :return:
        """
        d = queries.shapen[-1]
        scores = torch.bmm(queries,keys.transpose(1,2)) / math.sqrt(d)
        attention_weigths = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(attention_weigths),values)

class MutiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super(MutiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = Scaled_Dot_Product_Attention(dropout)

        # 线性层的作用是将qka的维度映射到低维，再在concat时组合起来（特征维度），但是为了并行化，这里的处理有点不一样
        # 假设在单个头的注意力中，fc的输出维度是fc_o，则fc_o*num_heads=num_hiddens.
        # 多头注意力的输出与输入一致，因此num_hiddens = query_size
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_fc = nn.Linear(num_hiddens,num_hiddens,bias=bias)

    def forward(self,queries,keys,values,valid_lens):
        """
        :param queries,key,valuess:(batch_size,n,d)
        :param valid_lens:
        :return:
        """

        queries = self.transpose_qkv(self.W_q(queries),self.num_heads)
        keys = self.transpose_qkv(self.W_(keys),self.num_heads)
        values = self.transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)
        # attention后的特征维度与之前相同，即(batch_size*num_heads,n,fc_o)
        output = self.attention(queries,keys,values,valid_lens)
        output_concat = self.transpose_output(output,self.num_heads)
        # 输出维度为(batch_size,n,num_hiddens) num_hiddens等于value_size
        return self.W_fc(output_concat)

    def transpose_qkv(self,X,num_heads):
        # 输入进来的X维度为(batch_size,n,num_hiddens)
        # 第一次reshape后维度变为 (batch_size,n,num_heads,num_hiddens/num_heads)
        # num_hiddens/num_heads 是单头的fc输出fc_o
        X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)

        # 调整位置后再融合前两个维度，输出的特征维度为(batch_size*num_heads,n,fc_o)
        X = X.permute(0,2,1,3)
        return X.reshape(-1,X.shape[2],X.shape[3])

    def transpose_output(self, X, num_heads):
        # 输入维度(batch_size*num_heads,n,fc_o)，先拆分前两个，再调换位置，变为(batch_size,n,num_heads，fc_o)
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        # 合并后两个维度(batch_size,n,num_heads*fc_o) = (batch_size,n,num_hiddens)
        return X.reshape(X.shape[0],X.shape[1],-1)




