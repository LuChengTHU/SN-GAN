#! /usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io as sio
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import h5py

def label_loader_64x64_31t(folder_list, filename_list, img_dic):
    t_list = list(set([ i.split('_')[3][1:] for i in filename_list ]))
    t_list.sort()
    img_list = []
    for folder in folder_list:
        for filename in filename_list:
            img = img_dic[folder + '/' + filename]
            t = filename.split('_')[3][1:]
            label = [t_list.index(t)]
            img_list.append((img, label))
            if t == -1:
                print("## Label Loader ERROR")
                exit(-1)
    return img_list

def label_loader_64x64_62tp(folder_list, filename_list, img_dic):
    t_list = list(set([ i.split('_')[3][1:] for i in filename_list ]))
    t_list.sort()
    p_list = list(set([ i.split('_')[2][1:] for i in filename_list ]))
    p_list.sort()
    img_list = []
    for folder in folder_list:
        for filename in filename_list:
            img = img_dic[folder + '/' + filename]
            t = filename.split('_')[3][1:]
            p = filename.split('_')[2][1:]
            t_label = t_list.index(t)
            p_label = p_list.index(p)
            label = [t_label, p_label]
            img_list.append((img, label))
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
                 img_path,
                 transform=None,
                 loader=default_loader):

        self.imgs = np.load(img_path)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img = self.imgs[index]
        #img = img.astype(np.float) / 255.0
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)



def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        ans = {}
        recursively_load_dict_contents_from_group(h5file, '/', '', ans)
        return ans


def recursively_load_dict_contents_from_group(h5file, path, rawkey, ans):
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[str(rawkey + key)] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            recursively_load_dict_contents_from_group(h5file, path + key + '/', rawkey + key + '/', ans)


def gen_hdf5_256():
    mat_path = '../rendered_chairs/all_chair_names.mat'
    matfile = sio.loadmat(mat_path)
    filenames = matfile['folder_names'][0]
    instnames = matfile['instance_names'][0]
    folder_list = [
        os.path.join(
            os.path.join('../rendered_chairs', str(i[0])),
            'renders') for i in filenames
    ]
    filename_list = [str(i[0]) for i in instnames]
    img_dic = {}
    num = 0
    for folder in folder_list:
        folder_img_dic = {}
        for filename in filename_list:
            imgPath = os.path.join(folder, filename)
            img = default_loader(imgPath).resize((256, 256), Image.ANTIALIAS)
            img = np.asarray(img)
            folder_img_dic[filename] = img
        img_dic[folder] = folder_img_dic
        num += 1
        print("%d: %s" % (num, folder))

    save_dict_to_hdf5(img_dic, '../rendered_chairs/all_chair_img_256.h5')

def gen_hdf5_64():
    mat_path = '../rendered_chairs/all_chair_names.mat'
    matfile = sio.loadmat(mat_path)
    filenames = matfile['folder_names'][0]
    instnames = matfile['instance_names'][0]
    folder_list = [
        os.path.join(
            os.path.join('../rendered_chairs', str(i[0])),
            'renders') for i in filenames
    ]
    filename_list = [str(i[0]) for i in instnames]
    img_dic = {}
    num = 0
    for folder in folder_list:
        folder_img_dic = {}
        for filename in filename_list:
            imgPath = os.path.join(folder, filename)
            img = default_loader(imgPath).resize((64, 64), Image.ANTIALIAS)
            img = np.asarray(img)
            folder_img_dic[filename] = img
        img_dic[folder] = folder_img_dic
        num += 1
        print("%d: %s" % (num, folder))

    save_dict_to_hdf5(img_dic, '../rendered_chairs/all_chair_img.h5')


