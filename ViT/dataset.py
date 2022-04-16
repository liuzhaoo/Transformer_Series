from torch.utils.data import Dataset
from PIL import Image


class VitDataset(Dataset):

    def __init__(self, file_path, transform):
        super(VitDataset, self).__init__()
        with open(file_path, 'r') as f:
            self.img_info = f.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        img_idex = self.img_info[index]
        image_path = img_idex.split('\n')[0]
        image_label = img_idex.split('\n')[-1]

        img = Image.open(image_path)
        img = self.transform(img)

        return img,image_label

