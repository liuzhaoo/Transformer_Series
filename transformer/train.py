import torch
from d2l import torch as d2l
from torch import nn


def sequence_mask(X,valid_len,value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen,dtype=torch.float32,
                        device=X.device)[None,:] < valid_len[:,None]

    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """mask 交叉熵损失"""
    # pred：（batch_size,num_steps,vocab_size）
    # label: (batch_size,num_steps)
    # valid (batch_size,)
    def forward(self, pred,label,valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights,valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,self).forward(
            pred.permute(0,2,1),label)
        weighted_loss = (unweighted_loss*weights).mean(dim=1)

        return weighted_loss

def train_seq2seq(net,data_iter,lr,num_epochs,tgt_vocab,device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weights' in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',xlim=[10,num_epochs])

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)

        for batch in data_iter:
            optimizer.zero_grad()
            src_data,src_validlens,tar_data,tar_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']]*tar_data.shape[0],device=device).reshape(-1,1)

            dec_input = torch.cat([bos,tar_data[:,:-1]],1)

            Y_hat,_ = net(src_data,dec_input,src_validlens)

            l = loss(Y_hat,tar_data,src_validlens)
            l.sum().backward()
            d2l.grad_clipping(net,1)
            num_tokens = tar_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(),num_tokens)
        if (epoch+1) %10 ==0:
            animator.add(epoch+1,(metric[0]/metric[1]))
        print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')