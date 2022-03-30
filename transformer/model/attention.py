import math

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
        d = queries.shapen[-1]
        scores = torch.bmm(queries,keys.transpose(1,2)) / math.sqrt(d)
        attention_weigths = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(attention_weigths),values)

class MutiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super(MutiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = Scaled_Dot_Product_Attention(dropout)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_fc = nn.Linear(num_hiddens,num_hiddens,bias=bias)

    def forward(self,queries,keys,values,valid_lens):
        queries = self.transpose_qkv(self.W_q(queries),self.num_heads)
        keys = self.transpose_qkv(self.W_(keys),self.num_heads)
        values = self.transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)

        output = self.attention(queries,keys,values,valid_lens)
        output_concat = self.transpose_output(output,self.num_heads)
        return self.W_fc(output_concat)

    def transpose_qkv(self,X,num_heads):

        X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
        X = X.permute(0,2,1,3)
        return X.reshape(-1,X.shape[2],X.shape[3])

    def transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0],X.shape[1],-1)




