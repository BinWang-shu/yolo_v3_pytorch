import torch
import torch.nn as nn
import torch.nn.functional as F 
from test_util import predict_transform
from train_util import build_target
import numpy as np 
from torch.autograd import Variable

class Conv(nn.Module):
	def __init__(self, inC, outC, ksize=3, stride=1, padding=1):
		super(Conv, self).__init__()
		self.conv = nn.Conv2d(inC, outC, ksize, stride, padding)
		self.batchnorm = nn.BatchNorm2d(outC)
		self.relu = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, x):
		out = self.relu(self.batchnorm(self.conv(x)))
		return out

class Residual(nn.Module):
	def __init__(self, inC):
		super(Residual, self).__init__()
		self.conv1 = Conv(inC, inC/2, 1, 1, 0)
		self.conv2 = Conv(inC/2, inC, 3, 1, 1)

	def forward(self, x):
		out = self.conv2(self.conv1(x))
		return x + out

class ConvS(nn.Module):
	def __init__(self, inC, outC, concat=False):
		super(ConvS, self).__init__()
		if concat:
			inp = inC + inC/2
		else:
			inp = inC
		self.conv1 = Conv(inp, inC/2, 1, 1, 0)
		self.conv2 = Conv(inC/2, inC, 3, 1, 1)
		self.conv3 = Conv(inC, inC/2, 1, 1, 0)
		self.conv4 = Conv(inC/2, inC, 3, 1, 1)
		self.conv5 = Conv(inC, inC/2, 1, 1, 0)
		self.conv6 = Conv(inC/2, inC, 3, 1, 1)
		self.conv7 = Conv(inC, outC, 1, 1, 0)

		self.conv_route = Conv(inC/2, inC/4, 1, 1, 0)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		conv6 = self.conv6(conv5)
		out = self.conv7(conv6)

		out_route = self.conv_route(conv5)

		return out_route, out

class DarkNet(nn.Module):
	def __init__(self, anchors, num_classes):
		super(DarkNet, self).__init__()

		self.anchors1 = anchors[0]
		self.anchors2 = anchors[1]
		self.anchors3 = anchors[2]
		self.num_anchors = len(self.anchors1)
		self.num_classes = num_classes

		self.object_scale=5
		self.noobject_scale=1
		self.class_scale=1
		self.coord_scale=1
		out_channels = (5 + num_classes) * len(anchors)
		# stage1 1/1
		self.conv0 = Conv(3, 32, 3, 1, 1)

		# downsample1 1/2 
		self.down1 = Conv(32, 64, 3, 2, 1)
		self.res1 = Residual(64)

		# downsample2 1/4
		self.down2 = Conv(64, 128, 3, 2, 1)
		self.res2 = nn.Sequential(
			Residual(128),
			Residual(128),
		)

		# downsample3 1/8
		self.down3 = Conv(128, 256, 3, 2, 1)
		self.res3 = nn.Sequential(
			Residual(256),
			Residual(256),
			Residual(256),
			Residual(256),
			Residual(256),
			Residual(256),
			Residual(256),
			Residual(256),
			)

		# downsample4 1/16
		self.down4 = Conv(256, 512, 3, 2, 1)
		self.res4 = nn.Sequential(
			Residual(512),
			Residual(512),
			Residual(512),
			Residual(512),
			Residual(512),
			Residual(512),
			Residual(512),
			Residual(512),
			)

		# downsample5 1/32
		self.down5 = Conv(512, 1024, 3, 2, 1)
		self.res5 = nn.Sequential(
			Residual(1024),
			Residual(1024),
			Residual(1024),
			Residual(1024),
			)

		self.convs_13 = ConvS(1024, out_channels)
		self.convs_26 = ConvS(512, out_channels, concat=True)
		self.convs_52 = ConvS(256, out_channels, concat=True)

		self.upsample_13_26 = nn.Upsample(scale_factor=2, mode="bilinear")
		self.upsample_26_52 = nn.Upsample(scale_factor=2, mode="bilinear")

	def forward(self, x, gt_boxes=None, gt_classes=None):
		bsize, _, H, W = x.size()
		assert H==W, ['Input size is not equal!']
		x = self.conv0(x)
		feature1 = self.res1(self.down1(x))
		feature2 = self.res2(self.down2(feature1))
		feature3 = self.res3(self.down3(feature2))
		feature4 = self.res4(self.down4(feature3))
		feature5 = self.res5(self.down5(feature4))

		## route
		route_13, detection_13 = self.convs_13(feature5)
		route_13_up = self.upsample_13_26(route_13)

		route_26, detection_26 = self.convs_26(torch.cat((route_13_up, feature4), 1))
		route_26_up = self.upsample_26_52(route_26)

		_, detection_52 = self.convs_52(torch.cat((route_26_up, feature3), 1))

		if self.training:
			## reshape the out (N,C,H,W) ==> (N,H*W,3,num_anchors+5)
			detection_13_reshape = \
				detection_13.permute(0, 2, 3, 1).contiguous().view(bsize,
                                             -1, self.num_anchors, self.num_classes + 5)
			detection_26_reshape = \
				detection_26.permute(0, 2, 3, 1).contiguous().view(bsize,
                                             -1, self.num_anchors, self.num_classes + 5)
			detection_52_reshape = \
				detection_52.permute(0, 2, 3, 1).contiguous().view(bsize,
                                             -1, self.num_anchors, self.num_classes + 5)

			### detection_13: ###
			_, _, h, w = detection_13.size()
			assert h == 13, ['output 13x13 size is not 13!']
			assert h==w, ['output 13x13 size is not equal!']

			xy_pred_13 = F.sigmoid(detection_13_reshape[:,:,:,0:2])
			wh_pred_13 = torch.exp(detection_13_reshape[:,:,:,2:4])
			bbox_pred_13 = torch.cat([xy_pred_13, wh_pred_13], 3) # b,h*w,a,4
			iou_pred_13 = F.sigmoid(detection_13_reshape[:,:,:,4:5]) # b,h*w,a,1
			classes_pred_13 = F.sigmoid(detection_13_reshape[:,:,:,5:]) # b,h*w,a,20

			bbox_pred_13_tensor = bbox_pred_13.data.cpu()
			iou_pred_13_tensor = iou_pred_13.data.cpu()

			inp_size = [H, W]
			out_size = [h, w]

			_boxes_13, _ious_13, _classes_13, _box_mask_13, _iou_mask_13, _class_mask_13 = \
					build_target(inp_size, out_size, bbox_pred_13_tensor, gt_boxes, 
							gt_classes, iou_pred_13_tensor, self.anchors3,
								self.object_scale, self.noobject_scale, 
									self.class_scale, self.coord_scale)

			## to Variable
			boxes_13 = Variable(_boxes_13).cuda() # b, hw, a, 4
			ious_13 = Variable(_ious_13).cuda() # b, hw, a, 1
			classes_13 = Variable(_classes_13).cuda() # b, hw, a, 20
			box_mask_13 = Variable(_box_mask_13).cuda() # b, hw, a, 1
			iou_mask_13 = Variable(_iou_mask_13.sqrt()).cuda() # b, hw, a, 1
			class_mask_13 = Variable(_class_mask_13).cuda() # b, hw, a, 1

			# compute loss
			box_mask_13 = box_mask_13.expand_as(_boxes_13)
			bbox_loss_13 = 5.0 * F.mse_loss(bbox_pred_13 * box_mask_13, 
					boxes_13 * box_mask_13, size_average=False)
			iou_loss_13 = F.mse_loss(iou_pred_13 * iou_mask_13, 
					ious_13 * iou_mask_13, size_average=False)

			class_mask_13 = class_mask_13.expand_as(_classes_13)
			cls_loss_13 = F.mse_loss(classes_pred_13 * class_mask_13, 
					classes_13 * class_mask_13, size_average=False)

			### detection_26: ###
			_, _, h, w = detection_26.size()
			assert h == 26, ['output 26x26 size is not 26!']
			assert h==w, ['output 26x26 size is not equal!']

			xy_pred_26 = F.sigmoid(detection_26_reshape[:,:,:,0:2])
			wh_pred_26 = torch.exp(detection_26_reshape[:,:,:,2:4])
			bbox_pred_26 = torch.cat([xy_pred_26, wh_pred_26], 3) # b,h*w,a,4
			iou_pred_26 = F.sigmoid(detection_26_reshape[:,:,:,4:5]) # b,h*w,a,1
			classes_pred_26 = F.sigmoid(detection_26_reshape[:,:,:,5:]) # b,h*w,a,20

			bbox_pred_26_tensor = bbox_pred_26.data.cpu()
			iou_pred_26_tensor = iou_pred_26.data.cpu()

			inp_size = [H, W]
			out_size = [h, w]

			_boxes_26, _ious_26, _classes_26, _box_mask_26, _iou_mask_26, _class_mask_26 = \
					build_target(inp_size, out_size, bbox_pred_26_tensor, gt_boxes, 
							gt_classes, iou_pred_26_tensor, self.anchors2,
								self.object_scale, self.noobject_scale, 
									self.class_scale, self.coord_scale)

			## to Variable
			boxes_26 = Variable(_boxes_26).cuda() # b, hw, a, 4
			ious_26 = Variable(_ious_26).cuda() # b, hw, a, 1
			classes_26 = Variable(_classes_26).cuda() # b, hw, a, 20
			box_mask_26 = Variable(_box_mask_26).cuda() # b, hw, a, 1
			iou_mask_26 = Variable(_iou_mask_26.sqrt()).cuda() # b, hw, a, 1
			class_mask_26 = Variable(_class_mask_26).cuda() # b, hw, a, 1

			# compute loss
			box_mask_26 = box_mask_26.expand_as(_boxes_26)
			bbox_loss_26 = 5.0 * F.mse_loss(bbox_pred_26 * box_mask_26, 
					boxes_26 * box_mask_26, size_average=False)
			iou_loss_26 = F.mse_loss(iou_pred_26 * iou_mask_26, 
					ious_26 * iou_mask_26, size_average=False)

			class_mask_26 = class_mask_26.expand_as(_classes_26)
			cls_loss_26 = F.mse_loss(classes_pred_26 * class_mask_26, 
					classes_26 * class_mask_26, size_average=False)

			### detection_52 ###
			_, _, h, w = detection_52.size()
			assert h == 52, ['output 52x52 size is not 52!']
			assert h==w, ['output 52x52 size is not equal!']

			xy_pred_52 = F.sigmoid(detection_52_reshape[:,:,:,0:2])
			wh_pred_52 = torch.exp(detection_52_reshape[:,:,:,2:4])
			bbox_pred_52 = torch.cat([xy_pred_52, wh_pred_52], 3) # b,h*w,a,4
			iou_pred_52 = F.sigmoid(detection_52_reshape[:,:,:,4:5]) # b,h*w,a,1
			classes_pred_52 = F.sigmoid(detection_52_reshape[:,:,:,5:]) # b,h*w,a,20

			bbox_pred_52_tensor = bbox_pred_52.data.cpu()
			iou_pred_52_tensor = iou_pred_52.data.cpu()

			inp_size = [H, W]
			out_size = [h, w]

			_boxes_52, _ious_52, _classes_52, _box_mask_52, _iou_mask_52, _class_mask_52 = \
					build_target(inp_size, out_size, bbox_pred_52_tensor, gt_boxes, 
							gt_classes, iou_pred_52_tensor, self.anchors1,
								self.object_scale, self.noobject_scale, 
									self.class_scale, self.coord_scale)

			## to Variable
			boxes_52 = Variable(_boxes_52).cuda() # b, hw, a, 4
			ious_52 = Variable(_ious_52).cuda() # b, hw, a, 1
			classes_52 = Variable(_classes_52).cuda() # b, hw, a, 20
			box_mask_52 = Variable(_box_mask_52).cuda() # b, hw, a, 1
			iou_mask_52 = Variable(_iou_mask_52.sqrt()).cuda() # b, hw, a, 1
			class_mask_52 = Variable(_class_mask_52).cuda() # b, hw, a, 1

			# compute loss
			box_mask_52 = box_mask_52.expand_as(_boxes_52)
			bbox_loss_52 = 5.0 * F.mse_loss(bbox_pred_52 * box_mask_52, 
					boxes_52 * box_mask_52, size_average=False)
			iou_loss_52 = F.mse_loss(iou_pred_52 * iou_mask_52, 
					ious_52 * iou_mask_52, size_average=False)

			class_mask_52 = class_mask_52.expand_as(_classes_52)
			cls_loss_52 = F.mse_loss(classes_pred_52 * class_mask_52, 
					classes_52 * class_mask_52, size_average=False)

			# print ('bbox_loss_13:', bbox_loss_13.data[0])
			# print ('iou_loss_13:', iou_loss_13.data[0])
			# print ('cls_loss_13:', cls_loss_13.data[0])
			# print ('bbox_loss_26:', bbox_loss_26.data[0])
			# print ('iou_loss_26:', iou_loss_26.data[0])
			# print ('cls_loss_26:', cls_loss_26.data[0])
			# print ('bbox_loss_52:', bbox_loss_52.data[0])
			# print ('iou_loss_52:', iou_loss_52.data[0])
			# print ('cls_loss_52:', cls_loss_52.data[0])
			loss_13 = bbox_loss_13 + iou_loss_13 + cls_loss_13
			loss_26 = bbox_loss_26 + iou_loss_26 + cls_loss_26
			loss_52 = bbox_loss_52 + iou_loss_52 + cls_loss_52

			return loss_13, loss_26, loss_52


		else:
			detection_13_trans = predict_transform(detection_13.data.cpu(), H, self.anchors3, self.num_classes)
			detection_26_trans = predict_transform(detection_26.data.cpu(), H, self.anchors2, self.num_classes)
			detection_52_trans = predict_transform(detection_52.data.cpu(), H, self.anchors1, self.num_classes)

			detections = torch.cat((detection_13_trans, detection_26_trans, detection_52_trans), 1)

			return detections






















