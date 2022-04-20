import time

from utils import AverageMeter,calculate_accuracy
import torch

def val_one_epoch(epoch,model,loss_fn,dataloader,device,tb_writer,epoch_logger):
    print('validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    dataload_time = AverageMeter()
    Loss = AverageMeter()
    Acc = AverageMeter()
    end_time = time.time()

    # 后续加上tqdm
    with torch.no_grad():
        for i, (img,label) in enumerate(dataloader):

            dataload_time.update(time.time()-end_time)

            label = label.to(device,non_blocking=True)
            outputs = model(img)

            loss = loss_fn(outputs,label)
            acc = calculate_accuracy(outputs,label)

            Loss.update(loss.item(),img.size(0))
            Acc.update(acc,img.size(0))

            batch_time.update(time.time()-end_time)
            end_time = time.time()

            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time})\t'
                  'loss {loss.val:.4f} ({loss.avg:4f})\t'
                  'Acc {acc.val:3f} ({acc.avg:.3f})'.format(epoch,
                                                            i + 1,
                                                            len(dataloader),
                                                            batch_time=batch_time,
                                                            data_time=dataload_time,
                                                            loss=Loss,
                                                            acc=Acc))
    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': Loss.avg,
            'acc': Acc.avg
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', Loss.avg, epoch)
        tb_writer.add_scalar('train/acc', Acc.avg, epoch)

    return Loss.avg














