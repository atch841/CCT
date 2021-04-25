import os
import random
# import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
        
class LiTS_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None, tumor_only=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list_ct = os.listdir(base_dir + 'ct/')
        self.sample_list_seg = os.listdir(base_dir + 'seg/')
        self.data_dir = base_dir
        self.tumor_only = tumor_only

    def __len__(self):
        return len(self.sample_list_ct)

    def __getitem__(self, idx):
        if self.split == "train":
            image = np.load(self.data_dir + 'ct/' +  self.sample_list_ct[idx])
            label = np.load(self.data_dir + 'seg/' +  self.sample_list_seg[idx])
        else:
            ct = sitk.ReadImage(self.data_dir + 'ct/' + self.sample_list_ct[idx], sitk.sitkInt16)
            seg = sitk.ReadImage(self.data_dir + 'seg/' + self.sample_list_seg[idx], sitk.sitkUInt8)
            image = sitk.GetArrayFromImage(ct)
            label = sitk.GetArrayFromImage(seg)

            image = image.astype(np.float32)
            image = image / 200

            image = ndimage.zoom(image, (1, 0.5, 0.5), order=3)
            label = ndimage.zoom(label, (1, 0.5, 0.5), order=0)

        if self.tumor_only:
            label = (label == 2).astype('float32')

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        else:
            image = ndimage.zoom(image, (0.5, 0.5), order=3)
            label = ndimage.zoom(label, (0.5, 0.5), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label}
        sample['label'] = sample['label'].max()
        sample['name'] = self.sample_list_ct[idx][:-4]
        return sample