import torch
import time, random, cv2, sys, os
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize1
from medpy import metric
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
import SimpleITK as sitk
from torch.utils.data import Dataset



class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                val_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch, train_logger)
        
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.save_dir = config['trainer']['save_dir']

        # self.ignore_index = self.val_loader.dataset.ignore_index
        self.ignore_index = 255
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize1(),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()



    def _train_epoch(self, epoch):
        self.html_results.save()
        
        self.logger.info('\n')
        self.model.train()

        print(self.mode)
        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=135)

        self._reset_metrics()
        for batch_idx in tbar:
            if self.mode == 'supervised':
                (input_l, target_l), (input_ul, target_ul) = next(dataloader), (None, None)
            else:
                (input_l, target_l), (input_ul, target_ul) = next(dataloader)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)

            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            self.optimizer.zero_grad()

            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                        curr_iter=batch_idx, target_ul=target_ul, epoch=epoch-1)
            # np.save(self.save_dir + 'test.npy', {
            #     'input_l': input_l.cpu().data.numpy(),
            #     'outputs': outputs['sup_pred'].cpu().data.numpy(),
            #     'target_l': target_l.cpu().data.numpy()
            # })
            # time.sleep(5)
            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch-1)
            logs = self._log_values(cur_losses)
            
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                self._write_scalars_tb(logs)

            if batch_idx % int(len(self.unsupervised_loader)*0.9) == 0:
                self._write_img_tb(input_l, target_l, input_ul, target_ul, outputs, epoch)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs
            
            tbar.set_description('T ({}) | Ls {:.5f} Lu {:.5f} Lw {:.5f} PW {:.5f} m1 {:.5f} m2 {:.5f} lr {:.5f}|'.format(
                epoch, self.loss_sup.average, self.loss_unsup.average, self.loss_weakly.average,
                self.pair_wise.average, self.mIoU_l, self.mIoU_ul, self.optimizer.param_groups[0]['lr']))

            self.lr_scheduler.step(epoch=epoch-1)

        return logs



    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        dsc = self.inference(2, self.model, epoch)

        # self.model.eval()
        self.wrt_mode = 'val'
        # total_loss_val = AverageMeter()
        # total_inter, total_union = 0, 0
        # total_correct, total_label = 0, 0

        # tbar = tqdm(self.val_loader, ncols=130)
        # with torch.no_grad():
        #     val_visual = []
        #     for batch_idx, (data, target) in enumerate(tbar):
        #         target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

        #         H, W = target.size(1), target.size(2)
        #         up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
        #         pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
        #         data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
        #         output = self.model(data)
        #         output = output[:, :, :H, :W]

        #         # LOSS
        #         loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
        #         total_loss_val.update(loss.item())

        #         print('val')
        #         correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
        #         total_inter, total_union = total_inter+inter, total_union+union
        #         total_correct, total_label = total_correct+correct, total_label+labeled

        #         # LIST OF IMAGE TO VIZ (15 images)
        #         if len(val_visual) < 15:
        #             if isinstance(data, list): data = data[0]
        #             target_np = target.data.cpu().numpy()
        #             output_np = output.data.max(1)[1].cpu().numpy()
        #             val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

        #         # PRINT INFO
        #         pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        #         IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        #         mIoU = IoU.mean()
        #         seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
        #                         "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

        #         tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
        #                                         total_loss_val.average, pixAcc, mIoU))

        #     self._add_img_tb(val_visual, 'val')

        seg_metrics = {'val_dsc_percase': dsc}

        # METRICS TO TENSORBOARD
        self.wrt_step = (epoch) * len(self.val_loader)
        # self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
        for k, v in list(seg_metrics.items()): #[:-1]: 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

        log = {
            # 'val_loss': total_loss_val.average,
            **seg_metrics
        }
        # self.html_results.add_results(epoch=epoch, seg_resuts=log)
        # self.html_results.save()

        if (time.time() - self.start_time) / 3600 > 22:
            self._save_checkpoint(epoch, save_best=self.improved)
        return log


    def calculate_metric_percase(self, pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum()>0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return dice, hd95
        elif pred.sum() > 0 and gt.sum()==0:
            return 1, 0
        else:
            return 0, 0


    def test_single_volume(self, image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
        image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
        if len(image.shape) == 3:
            prediction = np.zeros_like(label)
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                if x != patch_size[0] or y != patch_size[1]:
                    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
                net.eval()
                with torch.no_grad():
                    outputs = net(input)
                    # if type(outputs) == tuple: # for deeplab_resnest
                    #     outputs = outputs[0]
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    if x != patch_size[0] or y != patch_size[1]:
                        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    else:
                        pred = out
                    prediction[ind] = pred
        else:
            input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
                prediction = out.cpu().detach().numpy()
        metric_list = []
        for i in range(1, classes):
            metric_list.append(self.calculate_metric_percase(prediction == i, label == i))

        if test_save_path is not None:
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
            img_itk.SetSpacing((1, 1, z_spacing))
            prd_itk.SetSpacing((1, 1, z_spacing))
            lab_itk.SetSpacing((1, 1, z_spacing))
            sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        return metric_list

    def inference(self, num_classes, model, epoch, test_save_path=None):
        # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
        db_test = LiTS_dataset('/home/viplab/data/stage1/test/', split='test_vol', tumor_only=True)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        self.logger.info("{} test iterations per epoch".format(len(testloader)))
        model.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = self.test_single_volume(image, label, model, classes=num_classes, patch_size=[256, 256],
                                        test_save_path=test_save_path, case=case_name, z_spacing=1)
            metric_list += np.array(metric_i)
            self.logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(db_test)
        for i in range(1, num_classes):
            self.logger.info('%d Mean class %d mean_dice %f mean_hd95 %f' % (epoch, i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        self.logger.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        # return "Testing Finished!"
        return metric_list[-1][0]

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup  = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}



    def _update_losses(self, cur_losses):
        if "loss_sup" in cur_losses.keys():
            self.loss_sup.update(cur_losses['loss_sup'].mean().item())
        if "loss_unsup" in cur_losses.keys():
            self.loss_unsup.update(cur_losses['loss_unsup'].mean().item())
        if "loss_weakly" in cur_losses.keys():
            self.loss_weakly.update(cur_losses['loss_weakly'].mean().item())
        if "pair_wise" in cur_losses.keys():
            self.pair_wise.update(cur_losses['pair_wise'].mean().item())



    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        # print('sup')
        # predict_np = outputs['sup_pred'].cpu().data.numpy()
        # target_np = target_l.cpu().data.numpy()
        # np.save(self.save_dir + 'test_sup_p.npy', predict_np)
        # np.save(self.save_dir + 'test_sup_t.npy', target_np)
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)
        self._update_seg_metrics(*seg_metrics_l, True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if self.mode == 'semi':
            # print('unsup')
            # predict_np = outputs['unsup_pred'].cpu().data.numpy()
            # target_np = target_ul.cpu().data.numpy()
            # np.save(self.save_dir + 'test_unsup_p.npy', predict_np)
            # np.save(self.save_dir + 'test_unsup_t.npy', target_np)
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)
            self._update_seg_metrics(*seg_metrics_ul, False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()
            


    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union



    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }



    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs


    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        current_rampup = self.model.module.unsup_loss_w.current_rampup
        self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)



    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        # palette = self.val_loader.dataset.palette
        # print(len(val_visual))
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3) 
                        else Image.fromarray( i , 'L') for i in imgs]
            # imgs = []
            # for i in imgs:
            #     if (isinstance(i, torch.Tensor) and len(i.shape) == 3):
            #         imgs.append(self.restore_transform(i))
            #     else:
            #         imgs.append(Image.fromarray( i , 'L'))
            #         # imgs.append(colorize_mask(i, palette))
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        # print(len(val_img), val_img[0].shape)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0)//len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)



    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        if self.mode == 'semi':
            outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
            targets_ul_np = target_ul.data.cpu().numpy()
            imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul, outputs_ul_np, targets_ul_np)]
            self._add_img_tb(imgs, 'unsupervised')

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
            assert seg_path.replace('seg', 'ct') == image_path, (image_path, seg_path)
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
        sample['case_name'] = self.sample_list_ct[idx][:-4]
        return sample