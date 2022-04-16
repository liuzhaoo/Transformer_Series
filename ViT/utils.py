import json
import csv
import torch
import torch.nn as nn
import random
import os


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def updata(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')  # 待查

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()  # 转置

        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def load_flower_data(root_path, val_ratio=0.2):
    assert os.path.exists(root_path), 'root_path not exists'

    # 遍历文件夹,得到类别
    classes = [cls for cls in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, cls))]
    classes.sort()
    cls2index = dict((v, k) for k, v in enumerate(classes))
    index2cls = dict((v, k) for k, v in cls2index.items())

    train_data = []
    val_data = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cls in classes:
        cls_path = os.path.join(root_path, cls)
        # 取出当前类别下所有图像的地址
        index = cls2index[cls]
        cur_img_info = [os.path.join(cls_path, i) + ' ' + str(index) for i in os.listdir(cls_path)
                        if os.path.splitext(i)[-1] in supported]
        random.shuffle(cur_img_info)
        offset = int(len(cur_img_info) * val_ratio)
        val_data.extend(cur_img_info[:offset])
        train_data.extend(cur_img_info[offset:])
    random.shuffle(val_data)
    random.shuffle(train_data)
    with open(os.path.join(root_path, 'train_map.txt'), 'w') as f1:
        f1.write('\n'.join(train_data))
    with open(os.path.join(root_path, 'val_map.txt'), 'w') as f2:
        f2.write('\n'.join(val_data))
    json_str = json.dumps(cls2index, indent=4)
    with open(os.path.join(root_path, 'classes.json'), 'w') as json_f:
        json_f.write(json_str)


if __name__ == '__main__':
    root_path = '/Users/zhaoliu/PROJECTS/models/Transformer_series/data/flower_photos'
    load_flower_data(root_path)

    # x = ['asdsadas','adasfasfs','fghdhhdhdf']
    # print('\n'.join(x))
