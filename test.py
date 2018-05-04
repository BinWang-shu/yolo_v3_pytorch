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
from utils.voc_datasets import VOC2007_TEST
from models.DarkNet import DarkNet
import torch.nn.init as init
import torch.nn.functional as F
from models.test_util import write_results
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser(description='train hed model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--lr_decay', type=float, help='learning rate decay', default=0.1)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--imgdir', default='/media/data2/wb/VOCdevkit/VOC2007/JPEGImages/', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(2)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

imgdir = opt.imgdir
lr_decay_epoch = {10, 12}
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

confidence = 0.5
num_classes = 20
nms_thesh = 0.4

##########   DATASET   ###########
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])

###########   DATASET   ###########
data = VOC2007_TEST(transform=transform, size=416)
train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=2)

net = DarkNet(anchors, num_classes)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

net.apply(weights_init)

net.load_state_dict(torch.load('checkpoints/darknet_14.pth'))
net.cuda()

lr = opt.lr
optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))

########### Testing   ###########

net.eval()
topilimage = transforms.ToPILImage()
for i, image in enumerate(train_loader):
    img = image[0]
    gt_boxes = image[1]
    gt_classes = image[2]
    name = image[3][0] + '.jpg'

    img = Variable(img, volatile=True).cuda()
    gt_boxes = Variable(gt_boxes).cuda()
    gt_classes = Variable(gt_classes).cuda()

    detections = net(img, 0, 0)

    prediction = write_results(detections, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    print (prediction)

    # resize to the original img
    # ori_im = topilimage(img.data[0].cpu())
    ori_im = Image.open(imgdir + name).convert('RGB')
    W, H = ori_im.size

    scale_H = float(H) / img.size(2)
    scale_W = float(W) / img.size(3)

    drawObject = ImageDraw.Draw(ori_im)

    prediction[:,1:5:2] = prediction[:,1:5:2] * scale_W
    prediction[:,2:5:2] = prediction[:,2:5:2] * scale_H
    bboxes = prediction[:,1:5].clamp(min=0.0, max=415.0)
    for l in range(len(bboxes)):

        bbox = list(bboxes[l])
        print (bbox)
    
        drawObject.rectangle(bbox, outline='red')
    ori_im.show()

    exit()



