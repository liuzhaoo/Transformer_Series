import numpy as np
import torch
import random
import math

from vit import get_model
from val import val_one_epoch
from train import train_one_epoch
from utils import Logger, save_checkpoint
from dataset import VitDataset
from opt import parse_opts

from torch.optim import SGD, lr_scheduler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

def get_train_utils(opt, model_parameters):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(opt.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_data = VitDataset(file_path=opt.train_data_path, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_worker, pin_memory=True)
    train_logger = Logger(opt.result_path + '/train.log',
                          ['epoch', 'loss', 'acc'])
    train_batch_logger = Logger(opt.result_path + '/train_batch.log',
                                ['epoch', 'batch', 'iter', 'loss', 'acc'])

    optimizer = SGD(model_parameters, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return train_loader, train_logger, train_batch_logger, optimizer, scheduler


def get_val_utils(opt):
    transform_val = transforms.Compose([
        transforms.RandomResizedCrop(opt.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_data = VitDataset(file_path=opt.val_data_path, transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_worker, pin_memory=True)
    val_logger = Logger(opt.result_path + '/val.log',
                        ['epoch', 'loss', 'acc'])

    return val_logger, val_loader


def main(opt):


    # random.seed(opt.manual_seed)
    # np.random.seed(opt.manual_seed)
    # torch.manual_seed(opt.manual_seed)

    model = get_model(opt)
    loss_fn = torch.nn.CrossEntropyLoss().to(opt.device)
    parameters = model.parameters()

    train_loader, train_logger, \
    train_batch_logger, optimizer, scheduler = get_train_utils(opt, parameters)

    val_logger,val_loader = get_val_utils(opt)

    if opt.tensorboard:
        tb_writer = SummaryWriter(log_dir=opt.result_path)

    for i in range(opt.begin_epoch, opt.epochs + 1):
        loss,acc = train_one_epoch(i, model, train_loader, loss_fn, optimizer, opt.device, tb_writer, train_batch_logger,
                        train_logger)
        if i % opt.checkpoint == 0:
            save_file_path = opt.result_path + '/checkpoint/' + 'save_{}_loss_{}_acc_{}.pth'.format(i,loss,acc)
            save_checkpoint(save_file_path,i,model,optimizer,scheduler)

        val_loss = val_one_epoch(i,model,loss_fn,val_loader,opt.device,tb_writer,val_logger)

if __name__ == '__main__':
    opt = parse_opts()
    opt.device = torch.device('cpu')
    main(opt)


