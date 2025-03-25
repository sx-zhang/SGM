# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision import utils as vutils
import os

    

def cal_IOU(input, output):
    iou = 0
    for i in range(input.size(0)):
        tmp = input[i,2:,:,:] + output[i,2:,:,:]
        tmp = torch.where(tmp>0,1.0,0.0)
        if tmp.sum()==0:
            iou += 0
        else:
            iou += (input[i,2:,:,:]*output[i,2:,:,:]).sum()/tmp.sum()
    # iou /= input.size(0)
    return iou

def cal_Recall(input, output):
    n,c,h,w=input.size()
    recall = 0
    for i in range(n):
        a = torch.zeros(c)
        b = torch.zeros(c)
        for j in range(2, c):
            a[j] = input[i,j,:,:].max()
            b[j] = output[i,j,:,:].max()
        if a.sum()==0:
            recall += 0
        else:
            recall += (a*b).sum()/a.sum()
    # recall /= n
    return recall


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, bool_masked_pos = batch
        
        # bool_masked_pos = torch.from_numpy(new_masking_generator())

        samples = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        if args.bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, _ = model(samples, mask=bool_masked_pos)
        else:
            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask=bool_masked_pos)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    iou_all = 0
    recall_all= 0
    num_all= 0

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, bool_masked_pos = batch

        samples = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        if args.bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, outputs, mask = model(samples, mask=bool_masked_pos)
        else:
            with torch.cuda.amp.autocast():
                loss, outputs, mask = model(samples, mask=bool_masked_pos)
        
        # if (data_iter_step%10==0):
        h_1 = samples.size(2) // 16
        B = torch.tensor(range(0,h_1*h_1)).to(samples.device).unsqueeze(0)
        B = B.repeat(samples.size(0),1)
        order = mask.float()*1000+B
        ids = torch.argsort(order, dim=1)
        ids_ = torch.argsort(ids, dim=1)
        y = torch.gather(outputs, dim=1, index=ids_.unsqueeze(-1).expand(-1, -1, outputs.size(-1)))
        y = torch.where(y>0.7,1.0,0.0)
        y = model.unpatchify(y, c=samples.shape[1])
        y = y.detach()
        iou_all += cal_IOU(samples, y)
        recall_all += cal_Recall(samples, y)
        num_all += samples.size(0)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        # loss_scaler(loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}