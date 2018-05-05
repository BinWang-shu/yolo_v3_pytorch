# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import os
import random
import numpy as np 
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.voc_datasets import VOC2007
from models.DarkNet import DarkNet
import torch.nn.init as init
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='train hed model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--lr_decay', type=float, help='learning rate decay', default=0.1)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(2)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

lr_decay_epoch = {20， 30}
lr_decay = opt.lr_decay
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

mask = [0, 1, 2]
anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
mask1 = [6, 7, 8]
mask2 = [3, 4, 5]
mask3 = [0, 1, 2]
anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
anchors1 = [anchors[i] for i in mask1] # 52x52
anchors2 = [anchors[i] for i in mask2] # 26x26
anchors3 = [anchors[i] for i in mask3] # 13x13
anchors = [anchors1, anchors2, anchors3]

print (anchors)

##########   DATASET   ###########
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])

###########   DATASET   ###########
data = VOC2007(transform=transform, size=416)
train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=2)

net = DarkNet(anchors, 20)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

net.apply(weights_init)
net.cuda()

lr = opt.lr
optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))

########### Training   ###########
loss_step = 0
step = 0
net.train()
for epoch in range(1,opt.niter+1):
    for i, image in enumerate(train_loader):
        img = image[0]
        gt_boxes = image[1]
        gt_classes = image[2]
        # print ('gt_boxes:', gt_boxes[0])
        # print ('gt_classes:', gt_classes[0])
        img = Variable(img, requires_grad=True).cuda()
        gt_boxes = Variable(gt_boxes, requires_grad=True).cuda()
        gt_classes = Variable(gt_classes, requires_grad=True).cuda()
        # print ('boxes', gt_boxes)
        # print ('classes', gt_classes)
        loss_13, loss_26, loss_52 = net(img, gt_boxes, gt_classes)
        loss = loss_13 + loss_26 + loss_52
        loss.backward()
        optimizer.step()

        loss_step += loss.data.sum()
        step += 1
        loss_show = loss_step / float(step)

        if(i % 10 == 0):
            print('[%d/%d][%d/%d] loss_show: %.4f, Loss_13: %.4f, Loss_26: %.4f, Loss_52: %.4f, lr= %.g'
                      % (epoch, opt.niter, i, len(train_loader),
                         loss_show, loss_13.data.sum(), loss_26.data.sum(), loss_52.data.sum(), lr))
            loss_step = 0
            step = 0
        # if(i % 1000 == 0):
        #     vutils.save_image(final_output_F.data,
        #                'tmp/samples_i_%d_%03d.png' % (epoch, i),
        #                normalize=True)

    if epoch in lr_decay_epoch:
                lr *= lr_decay
                optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))

    torch.save(net.state_dict(), '%s/darknet_%d.pth' % (opt.outf, epoch))
