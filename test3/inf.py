import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tf
from torch.autograd import Variable
from torchvision import models

import cv2
import numpy as np
import time
from PIL import Image

import sys
sys.path.append('/home/storage3/traffic-sign/test3/')
from main3 import DetectionNet

home_dir = '/home/storage3/traffic-sign/test3/'
data_dir = home_dir + 'data/CUB200/CUB_200_2011/'


def get_classes(root=data_dir):
	class_file_name = root + 'classes.txt'
	classes_map = []
	classes_map.append('none') # dummy at classes_map[0]
	with open(class_file_name) as f:
		for line in f.read().splitlines():
			classes_map.append(line.split(' ')[1])
	return classes_map


def inf_transform(img, size=224): # img: PIL.Image object
	preprocess = tf.Compose([
		tf.Resize((size, size)),
		tf.ToTensor(),
		tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	return preprocess(img)


def bbox_inv_transform(bbox, w, h, size=224): # bbox: 1 * 4
	w_ratio = w / size
	h_ratio = h / size
	b = [int(bbox[0][0] * w_ratio), int(bbox[0][1] * h_ratio), int(bbox[0][2] * w_ratio), int(bbox[0][3] * h_ratio)]
	b[0] = max(b[0], 0)
	b[1] = max(b[1], 0)
	b[2] = min(w - 1 - b[0], b[2])
	b[3] = min(h - 1 - b[1], b[3])
	return b


def show(img, gt, pred, index):
	cv2.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 255, 0), thickness=2)
	cv2.rectangle(img, (pred[0], pred[1]), (pred[0] + pred[2], pred[1] + pred[3]), (255, 0, 0), thickness=2)
	cv2.imwrite(home_dir + 'output/result' + str(index) + '.jpg', img)


if __name__ == '__main__':
	images = [
		'images/001.Black_footed_Albatross/Black_Footed_Albatross_0061_796082.jpg',
		'images/002.Laysan_Albatross/Laysan_Albatross_0056_500.jpg',
		'images/003.Sooty_Albatross/Sooty_Albatross_0024_1161.jpg',
		'images/005.Crested_Auklet/Crested_Auklet_0070_785261.jpg',
		'images/017.Cardinal/Cardinal_0105_19045.jpg',
		'images/024.Red_faced_Cormorant/Red_Faced_Cormorant_0021_796265.jpg',
		'images/054.Blue_Grosbeak/Blue_Grosbeak_0075_36963.jpg',
		'images/070.Green_Violetear/Green_Violetear_0048_60789.jpg'
	]
	bboxes = [
		[50, 25, 156, 139],
		[92, 47, 331, 260],
		[98, 140, 181, 247],
		[133, 20, 309, 355],
		[138, 115, 280, 194],
		[91, 86, 193, 208],
		[196, 108, 156, 152],
		[54, 58, 273, 348]
	]
	classes_map = get_classes()
	sizes = [] # original sizes
	net = DetectionNet(models.resnet18(pretrained=True))
	net.cuda()

	# load parameters into the model:
	load_name = home_dir + 'load/' + 'DetectionNet_19.pth'
	print("loading checkpoint %s" % (load_name))
	checkpoint = torch.load(load_name)
	net.load_state_dict(checkpoint['model'])
	print("loaded checkpoint %s" % (load_name))

	for i, j in enumerate(images):
		img = Image.open(data_dir + j).convert('RGB')
		sizes.append(img.size) # [w, h]
		inputs = inf_transform(img).unsqueeze_(0)
		net.eval()
		inputs = Variable(inputs.cuda())
		outputs1, outputs2 = net(inputs)
		outputs2 = nn.functional.log_softmax(outputs2, dim=1)
		#print('outputs1:', outputs1.cpu().data) # FloatTensor of size 1(row)*4(column)
		#print('outputs2:', outputs2.cpu().data) # FloatTensor of size 1*200
		index = outputs2.data.cpu().max(1, keepdim=True)[1] # pred
		#print('index:', index)
		print('class:', classes_map[index[0][0]])
		pred = bbox_inv_transform(outputs1.data.cpu(), sizes[i][0], sizes[i][1])
		show(cv2.imread(data_dir + j), bboxes[i], pred, i + 1)

"""
tests:
24   001.Black_footed_Albatross/Black_Footed_Albatross_0061_796082.jpg
24   50.0 25.0 156.0 139.0
24   1
102  002.Laysan_Albatross/Laysan_Albatross_0056_500.jpg
102  92.0 47.0 331.0 260.0
102  2
139  003.Sooty_Albatross/Sooty_Albatross_0024_1161.jpg
139  98.0 140.0 181.0 247.0
139  3
277  005.Crested_Auklet/Crested_Auklet_0070_785261.jpg
277  133.0 20.0 309.0 355.0
277  5
921  017.Cardinal/Cardinal_0105_19045.jpg
921  138.0 115.0 280.0 194.0
921  17
1293 024.Red_faced_Cormorant/Red_Faced_Cormorant_0021_796265.jpg
1293 91.0 86.0 193.0 208.0
1293 24
3090 054.Blue_Grosbeak/Blue_Grosbeak_0075_36963.jpg
3090 196.0 108.0 156.0 152.0
3090 54
4053 070.Green_Violetear/Green_Violetear_0048_60789.jpg
4053 54.0 58.0 273.0 348.0
4053 70
"""
