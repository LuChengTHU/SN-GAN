#! /usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io as sio
from torch.utils.data import Dataset
import os
from PIL import Image


def label_loader_64x64_31t(folder_list, file_list):
    t_list = list(set([ i.split('_')[3][1:] for i in file_list ]))
    t_list.sort()
    img_list = []
    for folder in folder_list:
        for file in file_list:
            imgPath = os.path.join(folder, file)
            t = file.split('_')[3][1:]
            label = (t_list.index(t))
            img_list.append((imgPath, label))
            if t == -1:
                print("## Label Loader ERROR")
                exit(-1)
    return img_list

def label_loader_64x64_62tp(folder_list, file_list):
    t_list = list(set([ i.split('_')[3][1:] for i in file_list ]))
    t_list.sort()
    p_list = list(set([ i.split('_')[2][1:] for i in file_list ]))
    p_list.sort()
    img_list = []
    for folder in folder_list:
        for file in file_list:
            imgPath = os.path.join(folder, file)
            t = file.split('_')[3][1:]
            p = file.split('_')[2][1:]
            t_label = t_list.index(t)
            p_label = p_list.index(p)
            label = (t_label, p_label)
            img_list.append((imgPath, label))
            if t == -1:
                print("## Label Loader ERROR")
                exit(-1)
    return img_list

'''
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [0] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    return img, pad_label, lens
'''

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class custom_dataset(Dataset):
    def __init__(self,
                 root,
                 mat_path,
                 label_loader,
                 transform=None,
                 loader=default_loader):

        matfile = sio.loadmat(mat_path)
        filenames = matfile['folder_names'][0]
        instnames = matfile['instance_names'][0]

        folderPath_list = [
            os.path.join(
                os.path.join(root, str(i[0])),
                'renders') for i in filenames
        ]
        filename_list = [str(i[0]) for i in instnames]

        self.img_list = label_loader(folderPath_list, filename_list)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label = self.img_list[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)

