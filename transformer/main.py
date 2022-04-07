from dataset import load_data_nmt
from model.transformer import TransformerEncoder,TransformerDecoder,EncoderDecoder
from train import train_seq2seq
from d2l import torch as d2l


num_hiddens, num_layers, batch_size, num_steps = 32, 2, 64, 10
dropout = 0.1
lr, num_epochs, device = 0.005, 200, 'cpu'
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
fra_eng_dir = '/Users/zhaoliu/PROJECTS/models/Transformer_series/data/fra-eng/fra.txt'

train_iter,src_vocab,tgt_vocab = load_data_nmt(batch_size,num_steps,fra_eng_dir)
encoder = TransformerEncoder(
    len(src_vocab),key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,
    ffn_num_hiddens,num_heads,num_layers,dropout)
decoder = TransformerDecoder(
    len(tgt_vocab),key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,
    ffn_num_hiddens,num_heads,num_layers,dropout)

net = EncoderDecoder(encoder,decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)