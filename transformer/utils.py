import torch


def sequence_mask(X, valid_len, value=0):
    # X 应该为tensor
    maxlen = X.size(1)
    mask = torch.arange(maxlen, device=X.device, dtype=torch.float32)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    # print('start')
    if valid_lens is None:
        return torch.nn.functional.softmax(X,dim=-1)
    else:
        shape = X.shape


        if valid_lens.dim() == 1:

            valid_lens = torch.repeat_interleave(valid_lens, shape[1])

        else:
            valid_lens = valid_lens.reshape(-1)

        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return torch.nn.functional.softmax(X.reshape(shape), dim=-1)


if __name__ == '__main__':
    # x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 0]])
    # print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([1, 3])))
    # num_steps = 6
    # batch_size =3
    # dec_valid_lens = torch.arange(
    #     1, num_steps + 1).repeat(batch_size, 1)
    #
    # print(dec_valid_lens)
    x = 10
    for i in range(x):
        print(i)
        x -=1