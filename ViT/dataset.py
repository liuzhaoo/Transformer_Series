from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

class VitDataset(Dataset):

    def __init__(self, file_path, transform):
        super(VitDataset, self).__init__()
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self.transform = transform
        self.path_list = []
        self.label_list = []
        for line in lines:
            line = line.strip().split(' ')
            self.path_list.append(line[0])
            self.label_list.append(int(line[1]))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        img = Image.open(self.path_list[index])
        img = self.transform(img)

        return img,self.label_list[index]



