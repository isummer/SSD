import os
import sys
import numpy as np
from collections import OrderedDict
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data

from libs.datasets import *
from libs.transforms import SSDAugmentation
from libs.layers.functions import PriorBox
from libs.layers.modules import RefineMultiBoxLoss, WarmupMultiStepLR
from libs.models import RefineDet
from libs.utils import Config
from libs.utils import Timer


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--cfg', default='', type=str,
                    help='train config file path')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
args = parser.parse_args()
args.num_gpus = 1

cfgs = Config.fromfile(args.cfg)

if not os.path.exists(cfgs.save_folder):
    os.mkdir(cfgs.save_folder)

dataset = VOCDetection(root=cfgs.data.dataset_root,
                       transform=SSDAugmentation(tuple(cfgs.coder.input_shape), cfgs.train.aug_params.bgr_mean))

net = RefineDet(cfgs.model)
# net.load_state_dict(torch.load('./model/refine320_vgg16.pth'))

use_gpu = torch.cuda.is_available()

if use_gpu:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    net.load_weights(args.resume)

if use_gpu:
    net = net.to('cuda')

lr = cfgs.train.optimizer.learning_rate * args.num_gpus  # scale by num gpus
optimizer = optim.SGD(net.parameters(),
                      lr=lr,
                      momentum=cfgs.train.optimizer.momentum,
                      weight_decay=cfgs.train.optimizer.weight_decay)

arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
                                   False)
odm_criterion = RefineMultiBoxLoss(cfgs.model.num_classes, 0.5, True, 0, True, 3, 0.5,
                                   False, 0.01)

milestones = [step // args.num_gpus for step in cfgs.train.lr_steps]
epoch_size = len(dataset) // cfgs.train.batch_size
print('epoch size:', epoch_size)
scheduler = WarmupMultiStepLR(optimizer=optimizer,
                              milestones=milestones,
                              gamma=cfgs.train.optimizer.gamma,
                              # warmup_factor=cfgs.train.warmup_factor,
                              warmup_epochs=cfgs.train.warmup_epochs,
                              epoch_size=epoch_size)

data_loader = data.DataLoader(dataset, cfgs.train.batch_size,
                              num_workers=cfgs.train.num_workers,
                              shuffle=True, collate_fn=detection_collate,
                              pin_memory=True)

priorbox = PriorBox(cfgs.coder)
priors = priorbox.forward()

def train():

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // cfgs.train.batch_size
    print('Training RefineDet on:', cfgs.data.dataset)
    print('Using the specified args:')
    print(args)

    epoch_step = 0

    for epoch in range(cfgs.train.max_epoches+1):
        print('epoch:', epoch)
        if epoch in cfgs.train.lr_steps:
            epoch_step += 1
            
        running_loss = 0.0
        _t = Timer()

        for iteration, (images, targets) in enumerate(data_loader):
            scheduler.step()

            if use_gpu:
                images = images.to('cuda')
                targets = [ann.to('cuda') for ann in targets]

            _t.tic()

            optimizer.zero_grad()
            arm_loc_preds, arm_cls_preds, odm_loc_preds, odm_cls_preds = net(images)
            arm_preds = (arm_loc_preds, arm_cls_preds)
            odm_preds = (odm_loc_preds, odm_cls_preds)
            arm_loss_l, arm_loss_c = arm_criterion(arm_preds, priors, targets)
            odm_loss_l, odm_loss_c = odm_criterion(odm_preds, priors, targets, arm_preds, False)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
            loss.backward()
            optimizer.step()

            _t.toc()
            running_loss += loss.item()

            if iteration % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print('[%3d / %3d] | loss: %.4f | arm_loc loss: %.4f | arm_cls loss: %.4f | odm_loc loss: %.4f | odm_cls loss: %.4f | lr: %.6f' % (_iter, epoch_size, loss.item(), arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item(), lr), end='\r')
        lr = optimizer.param_groups[0]['lr']
        print('\nTotal time: %.3fs | loss: %.4f | lr: %.6f' % (_t.total_time, running_loss/epoch_size, lr))

        if epoch != 0 and epoch % 10 == 0:
            # print('Saving state, epoch:', epoch)
            # torch.save(net.state_dict(), 'weights/ssd300_voc_epoch_' +
            #            repr(epoch) + '.pth')
            print('Save checkpoint, epoch:', epoch)
            save_checkpoint(epoch, net, optimizer)
    torch.save(net.state_dict(),
               os.path.join(cfgs.save_folder, cfgs.data.dataset + '.pth'))


def save_checkpoint(epoch, net, optimizer):
    checkpoint_path = 'checkpoints/refine320_voc.ckpt'
    model_path = 'weights/refine320_voc_epoch_' + repr(epoch) + '.pth'
    torch.save({
        'epoch': epoch,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, checkpoint_path)
    torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    train()
