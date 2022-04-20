import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manual_seed',default='0',type=str)
    parser.add_argument('--num_classes',default=5,type=int)
    parser.add_argument('--image_size',default=224,type=int)
    parser.add_argument('--no_cuda',action='store_true')
    parser.add_argument('--result_path',default='/Users/zhaoliu/PROJECTS/models/Transformer_series/ViT/results')
    parser.add_argument('--train_data_path',
                        default='/Users/zhaoliu/PROJECTS/models/Transformer_series/data/flower_photos/train_map.txt')
    parser.add_argument('--val_data_path',
                        default='/Users/zhaoliu/PROJECTS/models/Transformer_series/data/flower_photos/val_map.txt')

    parser.add_argument('--num_worker', default=8, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--lrf', default=0.01, type=float)


    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--begin_epoch', default=1, type=int)

    parser.add_argument('--tensorboard', default=True)


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    opt = parse_opts()
    print(opt.no_cuda)