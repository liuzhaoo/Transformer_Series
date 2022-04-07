import torch
from d2l import torch as d2l
from data_tools import Vocab
from torch.utils import data

def read_mnt_data(dir_path):
    with open(dir_path,'r',encoding='UTF-8') as f:
        return f.read()

def preprocess_nmt(text):
# 对英-法数据集进行预处理
    def no_space(char,prev_char):
        return char in set(',.!?') and prev_char != ' '
    # 使用空格代替不间断空格
    # 小写字母代替大写字母
    text =  text.replace('\u202f',' ').replace('\xa0', ' ').lower()
    # 在单词和标点之间插入空格
    out = [' ' + char if i > 0 and no_space(char,text[i-1]) else char
           for i, char in enumerate(text)]

    return ''.join(out)

# print(preprocess_nmt(raw_text)[:80])

def tokenize_nmt(text,num_examples=None):
    # 词元化
    source, target = [], []
    for i ,line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source,target

def truncate_pad(line,num_steps,padding_token):
    # 截断或填充文本，以保持num_steps相同
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array_nmt(lines,vocab,num_steps):
    # 将文本序列转换为小批量
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l,num_steps,vocab['<pad>'])
                         for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)

    return array,valid_len

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_data_nmt(batch_size,num_steps,data_dir,num_examples=600):
    text = preprocess_nmt(read_mnt_data(data_dir))
    # source, target 分别为原语言和目标语言的词元（list）
    source, target = tokenize_nmt(text, num_examples)

    # src_vocab,tgt_vocab 分别是原语言和目标语言的词表
    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # src_array 是词表内的所有词的索引值，维度为（n,num_steps）n是词表的长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array,src_valid_len,tgt_array,tgt_valid_len)
    data_iter = load_array(data_arrays,batch_size)  # dataloader

    return data_iter,src_vocab,tgt_vocab



fra_eng_dir = '/Users/zhaoliu/PROJECTS/models/Transformer_series/data/fra-eng/fra.txt'















