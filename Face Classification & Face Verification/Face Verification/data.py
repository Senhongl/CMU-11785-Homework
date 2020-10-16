from torch.utils import data as Data
import numpy as np
import torch
from torch import nn
import os
from util import data_transform, data_transform_val
import torchvision

class train_datasets(Data.Dataset):
    def __init__(self,root, transform, images_per_person, start = 0):
        
        filenames = list(os.path.join(root, f.name) for f in os.scandir(root) if not f.name.startswith('.'))
        self.transform = transform
        self.labels = list(f.name for f in os.scandir(root) if not f.name.startswith('.'))
        self.images_per_person = images_per_person
        self.start = start
        imgs = []
        for f in filenames:
            imgs.append(list(os.path.join(f, sub_f.name) for sub_f in os.scandir(f) if not sub_f.name.startswith('.')))
        
        self.idx_map = {}
        for idx, label in enumerate(self.labels):
            arr = np.arange(len(imgs[idx]))
            np.random.shuffle(arr)
            self.idx_map[int(label)] = []
            for i in range(images_per_person):
                self.idx_map[int(label)].append(imgs[idx][arr[i]])
                
        
    def __getitem__(self, index):
        
        label = self.start + index
        img_paths = self.idx_map[self.start + index]
        data = []
        for path in img_paths:
            pil_img = Image.open(path)
            pil_img = self.transform(pil_img)
            data.append(np.asarray(pil_img))
            
        return np.array(data), label


    def __len__(self):
        return len(self.labels)


class val_datasets(Data.Dataset):
    def __init__(self, text_root, file_root, transform, task = 'validation'):
        self.transform = transform
        self.first_images = []
        self.second_images = []
        self.labels = []
        self.task = task
        self.file_root = file_root
        
        with open(text_root) as f:
            f = f.readlines()
            for row in f:
                row = row.strip()
                row = row.split(' ')
                self.first_images += [row[0]]
                self.second_images += [row[1]]
                if self.task == 'validation':
                    self.labels += [int(row[2])]
                
        
    def __getitem__(self, index):
        if self.task == 'validation':
            label = self.labels[index]
        else:
            label = 0
        first_img = Image.open(self.file_root + '/' + self.first_images[index])
        second_img = Image.open(self.file_root + '/' + self.second_images[index])

        first_img = self.transform(first_img)
        first_img = np.array(first_img)
        second_img = self.transform(second_img)
        second_img = np.array(second_img)
        outputs = np.concatenate((first_img, second_img), axis = 0)
        
        return outputs, label
            
    def __len__(self):
        return len(self.first_images)