
from d2l import torch as d2l
from data_tools import count_corpus

# d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
#                            '94646ad1522d915e7b0f9296181140edcf86a4f5')
#
# data_dir = d2l.download_extract('fra-eng')

fra_eng_dir = '/Users/zhaoliu/PROJECTS/models/Transformer_series/data/fra-eng/fra.txt'
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

raw_text = read_mnt_data(fra_eng_dir)
text = preprocess_nmt(raw_text)
source,target = tokenize_nmt(text,1000)

# counter = count_corpus(source)
#
# _token_freqs = sorted(counter.items(), key=lambda x: x[1],
#                                    reverse=True)
#
# print(_token_freqs)

src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])

# print(src_vocab.idx_to_token[:100])
# print(src_vocab.token_to_idx)

print(src_vocab.to_tokens(5))
# print(src_vocab[1100])
# print(source[:6],target[:6])









