import os
import random
# import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
from misc import imutils


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


def random_flip(image, label):
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
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
        

class RandomGenerator_flip(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_flip(image, label)
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
        self.sample_list_ct.sort()
        self.sample_list_seg.sort()
        self.data_dir = base_dir
        self.tumor_only = tumor_only

    def __len__(self):
        return len(self.sample_list_ct)

    def __getitem__(self, idx):
        if self.split == "train":
            image_path = self.data_dir + 'ct/' +  self.sample_list_ct[idx]
            seg_path = self.data_dir + 'seg/' +  self.sample_list_seg[idx]
            assert seg_path[seg_path.rfind('/') + 1:].replace('seg', 'ct') == image_path[image_path.rfind('/') + 1:], (image_path, seg_path)
            image = np.load(image_path)
            label = np.load(seg_path)
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
            # print(image.shape, label.shape)
            sample = {'image': image, 'label': label}
        sample['label'] = sample['label'].max()
        sample['name'] = self.sample_list_ct[idx][:-4]
        return sample


class LiTS_datasetMSF(LiTS_dataset):

    def __init__(self, base_dir, split,
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(base_dir, split, tumor_only=True)
        self.scales = scales

    def __getitem__(self, idx):
        # name = self.img_name_list[idx]
        name = self.sample_list_ct[idx][:-4]
        # name_str = decode_int_filename(name)

        # img = imageio.imread(get_img_path(name_str, self.voc12_root))
        img = np.load(self.data_dir + 'ct/' +  self.sample_list_ct[idx])
        label = np.load(self.data_dir + 'seg/' +  self.sample_list_seg[idx])
        img = ndimage.zoom(img, (0.5, 0.5), order=3)
        label = ndimage.zoom(label, (0.5, 0.5), order=0)

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            # s_img = self.img_normal(s_img)
            # s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(np.array([label.max()]))}
        return out


class KiTS_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None, tumor_only=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list_ct = os.listdir(base_dir + 'ct/')
        self.sample_list_seg = os.listdir(base_dir + 'seg/')
        self.sample_list_ct.sort()
        self.sample_list_seg.sort()
        self.data_dir = base_dir
        self.tumor_only = tumor_only

    def __len__(self):
        return len(self.sample_list_ct)

    def __getitem__(self, idx):
        if self.split == "train":
            image_path = self.data_dir + 'ct/' +  self.sample_list_ct[idx]
            seg_path = self.data_dir + 'seg/' +  self.sample_list_seg[idx]
            assert seg_path[seg_path.rfind('/') + 1:].replace('seg', 'ct') == image_path[image_path.rfind('/') + 1:], (image_path, seg_path)
            image = np.load(image_path)
            label = np.load(seg_path)
        else:
            ct = sitk.ReadImage(self.data_dir + 'ct/' + self.sample_list_ct[idx], sitk.sitkInt16)
            seg = sitk.ReadImage(self.data_dir + 'seg/' + self.sample_list_seg[idx], sitk.sitkUInt8)
            image = sitk.GetArrayFromImage(ct)
            label = sitk.GetArrayFromImage(seg)

            image = image.astype(np.float32) - 50
            image = image / 250

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
            # print(image.shape, label.shape)
            sample = {'image': image, 'label': label}
        sample['label'] = sample['label'].max()
        sample['name'] = self.sample_list_ct[idx][:-4]
        return sample


class KiTS_datasetMSF(KiTS_dataset):

    def __init__(self, base_dir, split,
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(base_dir, split, tumor_only=True)
        self.scales = scales

    def __getitem__(self, idx):
        # name = self.img_name_list[idx]
        name = self.sample_list_ct[idx][:-4]
        # name_str = decode_int_filename(name)

        # img = imageio.imread(get_img_path(name_str, self.voc12_root))
        img = np.load(self.data_dir + 'ct/' +  self.sample_list_ct[idx])
        label = np.load(self.data_dir + 'seg/' +  self.sample_list_seg[idx])
        img = ndimage.zoom(img, (0.5, 0.5), order=3)
        label = ndimage.zoom(label, (0.5, 0.5), order=0)

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            # s_img = self.img_normal(s_img)
            # s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(np.array([label.max()]))}
        return out