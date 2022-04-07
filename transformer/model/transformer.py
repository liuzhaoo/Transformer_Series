import math

import torch
import numpy as np
from torch import nn
from .attention import MultiHeadAttention
from .network import PositionalEncoding,AddNorm,PositionWiseFFN

class Encorder(nn.Module):
    def __init__(self,quiry_size,key_size,value_size,num_hiddens,
                 norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,
                 dropout,bias=False,**kwargs):
        # 实际上 num_hiddens 等于 value_size ···
        super(Encorder, self).__init__()

        self.attention = MultiHeadAttention(key_size,quiry_size,value_size,num_hiddens,
                                           num_heads,dropout,bias)
        self.addnorm1 = AddNorm(dropout,norm_shape)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm2 = AddNorm(dropout,norm_shape)

    def forward(self,X,valid_lens):
        Y = self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return self.addnorm2(Y,self.ffn(Y))


class TransformerEncoder(nn.Module):
    # 包含位置编码的n个attention
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,normshape,
                 ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
        self.blocks = nn.Sequential()

        for i in range(num_layers):
            self.blocks.add_module('block'+str(i),
                Encorder(query_size,key_size,value_size,num_hiddens,normshape,ffn_num_input,
                         ffn_num_hiddens,num_heads,dropout,bias))
    def forward(self,X,valid_lens,**kwargs):
        # 位置编码在-1到1之间，需要先对输入进行缩放，然后再相加
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights = [None]*len(self.blocks)
        for i, block in enumerate(self.blocks):
            X = block(X,valid_lens)
            self.attention_weights[i] = block.attention.attention.attention_weights
        return X


class Decoder(nn.Module):
    # 第i个块
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,
                 ffn_num_input,ffn_num_hiddens,num_heads,dropout,i,**kwargs):
        super(Decoder, self).__init__()
        self.attention_mask = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,
                                                 num_heads,dropout)
        self.addnorm1 = AddNorm(dropout,norm_shape)
        self.m_attention = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,
                                                 num_heads,dropout)
        self.addnorm2 = AddNorm(dropout,norm_shape)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm3 = AddNorm(dropout,norm_shape)
        self.i = i

    def forward(self,X,state):
        #
        encoder_outputs,encoder_validlens = state[0],state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i],X),axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size,num_steps,_ = X.shape
            decoder_validlens = torch.arange(
                1,num_steps+1,device=X.device).repeat(batch_size,1)
        else:
            decoder_validlens=None

        X2 = self.attention_mask(X,key_values,key_values,decoder_validlens)
        Y = self.addnorm1(X,X2)
        Y2 = self.m_attention(Y,encoder_outputs,encoder_outputs,encoder_validlens)
        Z = self.addnorm2(Y,Y2)
        return self.addnorm3(Z,self.ffn(Z)),state

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,normshape,
                 ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,**kwargs):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
        self.blocks = nn.Sequential()

        for i in range(num_layers):
            self.blocks.add_module('block'+str(i),
                Decoder(key_size,query_size,value_size,num_hiddens,normshape,ffn_num_input,
                        ffn_num_hiddens,num_heads,dropout,i))
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self,encoder_outputs,encoder_validlens,*args):
        return [encoder_outputs,encoder_validlens,[None]*self.num_layers]

    def forward(self,X,state):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))

        for i, block in enumerate(self.blocks):
            X,state = block(X,state)

        return self.dense(X),state

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,enc_in,dec_in,*args):
        enc_out = self.encoder(enc_in,*args)
        dec_state = self.decoder.init_state(enc_out,*args)
        return self.decoder(dec_in,dec_state)








