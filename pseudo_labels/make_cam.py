import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12#.dataloader
from misc import torchutils, imutils
from dataset import LiTS_datasetMSF


cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            # print(len(pack['label']))
            # print(label.shape, label)
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            # print(len(pack['img']))
            outputs = []
            for img in pack['img']:
                # print(img.shape)
                img = img.permute(1, 0, 2, 3)
                o = model(img.cuda(non_blocking=True))
                # print(o.shape)
                outputs.append(o)
            # outputs = [model(img.cuda(non_blocking=True))
            #            for img in pack['img']]

            temp = []
            for o in outputs:
                temp.append(F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0])
            strided_cam = torch.sum(torch.stack(temp), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]
            # print(valid_cat, bool(valid_cat))
            # print(strided_cam.shape, valid_cat, label)
            # raise EOFError()
            if valid_cat.nelement() != 0:
                strided_cam = strided_cam[valid_cat]
                strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

                highres_cam = highres_cam[valid_cat]
                highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            else:
                strided_cam = torch.zeros_like(strided_cam[0])
                highres_cam = torch.zeros_like(highres_cam[0])

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
            # print(process_id, n_gpus, iter, process_id == n_gpus - 1, iter % (len(databin) // 20) == 0, (len(databin) // 20))
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')
                print()


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    # dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
    #                                                          voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = LiTS_datasetMSF('/home/viplab/nas/train5/', 'train', scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()