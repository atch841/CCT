from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json

class LiTSDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2

        # self.palette = pallete.get_voc_pallete(self.num_classes)
        super(LiTSDataset, self).__init__(**kwargs)

    def _set_files(self):
        # self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
        # if self.split == "val":
        #     file_list = os.path.join("dataloaders/voc_splits", f"{self.split}" + ".txt")
        # elif self.split in ["train_supervised", "train_unsupervised"]:
        #     file_list = os.path.join("dataloaders/voc_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")
        # else:
        #     raise ValueError(f"Invalid split name {self.split}")
        self.ct_path = self.root + 'ct/'
        self.seg_path = self.root + 'seg/'
        if self.use_weak_lables:
            self.files = [f.replace('png', 'npy') for f in os.listdir(self.weak_labels_output)]
        else:
            self.files = os.listdir(self.ct_path)
        self.labels = os.listdir(self.seg_path)
        self.files.sort()
        self.labels.sort()

        # file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        # self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.ct_path, self.files[index])
        image = np.load(image_path)
        image_id = self.files[index].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            # label = (label == 255).astype(np.int32)
        else:
            assert self.files[index] == self.labels[index].replace('seg', 'ct'), (self.files[index], self.labels[index])
            label_path = os.path.join(self.seg_path, self.labels[index])
            label = np.load(label_path)
            label = (label == 2).astype(np.int32)
        return image, label, image_id

class LiTS(BaseDataLoader):
    def __init__(self, kwargs):
        
        # self.MEAN = [0.485, 0.456, 0.406]
        # self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        # kwargs['mean'] = self.MEAN
        # kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = LiTSDataset(**kwargs)

        super(LiTS, self).__init__(self.dataset, self.batch_size, shuffle, num_workers)
