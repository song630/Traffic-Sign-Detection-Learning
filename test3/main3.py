import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tf
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, models

import numpy as np
import time, os, random, argparse
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('/home/storage3/traffic-sign/test2/')
# self-implemented, not pretrained,
# the pretrained one in rotchvison.models is recommended
from ResNet import ResNet18

home_dir = '/home/storage3/traffic-sign/test3/'
data_dir = home_dir + 'data/CUB200/CUB_200_2011/'
"""
about the dataset:
1.altogether 11788 samples, 200 classes of birds
2.training set : test set = 8:2
"""

def parse_args():
	parser = argparse.ArgumentParser(description='birds classification & localization')
	# it should achieve an accuracy of more than 90% after 35 epoches
	# original: 20
	parser.add_argument('--epochs', dest='max_epochs',
						help='number of epochs to train',
						default=20, type=int)
	parser.add_argument('--nw', dest='num_workers',
						help='number of worker to load data', # ??
						default=0, type=int)
	# original: 224
	parser.add_argument('--size', dest='resize',
						help='resized before entering',
						default=224, type=int)
	parser.add_argument('--cuda', dest='cuda',
						help='whether use CUDA',
						action='store_true')
	# original: 0.001
	parser.add_argument('--lr', dest='lr',
						help='learnign rate',
						default=1e-3, type=float)
	# original: 5
	parser.add_argument('--decay', dest='decay_step',
						help='decay step',
						default=5, type=int)
	parser.add_argument('--net', dest='use_self',
						help='whether use self-implemented ResNet',
						default=False, type=bool)
	# original: 32
	parser.add_argument('--bs', dest='batch_size',
						help='batch_size',
						default=32, type=int)
	parser.add_argument('--r', dest='resume',
						help='resume checkpoint or not',
						default=False, type=bool)
	parser.add_argument('--checkepoch', dest='checkepoch',
						help='checkepoch to load model',
						default=1, type=int)
	parser.add_argument('--save_dir', dest='save_dir',
						help='directory to save models', default="/home/storage3/traffic-sign/test3/load/",
						type=str)
	return parser.parse_args()

n_images = 11788
n_classes = 200


def get_train_test(root=data_dir):
	file_name = root + 'train_test_split.txt'
	# train_indexes, test_indexes: stores the indexes of the images
	train_indexes, test_indexes = [], []
	with open(file_name) as f:
		for line in f.read().split():
			# every line in the file is like: <image_id> <is_train_set>,
			# if <is_train_set> == 1, then it belongs to training set.
			# notice: index starts from 1.
			index, is_train = line.split(' ')
			if bool(is_train):
				train_indexes.append(int(index))
			else:
				test_indexes.append(int(index))
	return train_indexes, test_indexes


def get_imgs(root=data_dir, train_indexes, test_indexes):
	file_name = data_dir + 'images.txt'
	paths = [] # stores all relative paths of images
	paths.append('none') # paths[0] is dummy
	with open(file_name) as f:
		for line in f.read().split():
			paths.append(line.split(' ')[1])
	# absolute paths of training images
	train_images = [(i, data_dir + 'images' + paths[i]) for i in train_indexes]
	test_images = [(i, data_dir + 'images' + paths[i]) for i in test_indexes]
	return train_images, test_images


def get_bboxes(root=data_dir, train_indexes, test_indexes):
	file_name = root + 'bounding_boxes.txt'
	train_bbox, test_bbox = [], []
	# construct a 2-dimentional array, it has (11788 + 1) * 4 float elements
	all_bbox.append([0.0, 0.0, 0.0, 0.0]) # the first row [0] is dummy
	with open(file_name) as f:
		for line in f.read().split():
			_, x, y, w, h = line.split(' ')
			all_bbox.append([float(x), float(y), float(w), float(h)])
	# elements in train bbox: (index, [x, y, w, h])
	train_bbox = [(i, all_bbox[i]) for i in train_indexes]
	test_bbox = [(i, all_bbox[i]) for i in test_indexes]
	return train_bbox, test_bbox # convert to Tensor later in class Dataset's __getitem__()


def get_lbls(root=data_dir, train_indexes, test_indexes): # ge the classification labels
	class_file_name = root + 'classes.txt'
	label_file_name = root + 'image_class_labels.txt'
	classes_map = []
	classes_map.append('none') # dummy at classes_map[0]
	all_labels = []
	all_labels.append('none') # dummy at all_labels[0]
	with open(class_file_name) as f:
		for line in f.read().split():
			classes_map.append(line.split(' ')[1])
	with open(label_file_name) as f:
		for line in f.read().split():
			all_labels.append(int(line.split(' ')[1]))
	train_labels = [(i, classes_map[all_labels[i]]) for i in train_indexes]
	test_labels = [(i, classes_map[all_labels[i]]) for i in test_indexes]
	return train_labels, test_labels


train_indexes, test_indexes = get_train_test(data_dir)
# element format: (index, absolute path)
train_images, test_images = get_imgs(data_dir, train_indexes, test_indexes)
# element format: (index, [x, y, w, h])
train_bbox, test_bbox = get_bboxes(data_dir, train_indexes, test_indexes)
# element format: (index, class)
train_labels, test_labels = get_lbls(data_dir, train_indexes, test_indexes)


class CUBDataset(Dataset):
	def __init__(self, size, train=True, trans=None, root=data_dir):
		self.size = size
		global train_images, test_images
		global train_bbox, test_bbox
		global train_labels, test_labels
		if train:
			self.imgs, self.bboxes, self.lbls = train_images, train_bbox, train_labels
		else:
			self.imgs, self.bboxes, self.lbls = test_images, test_bbox, test_labels
		self.transform = trans
		print('Dataset loaded')


	def __getitem__(self, index):
		img, bbox, lbl = self.imgs[index][1], self.bboxes[index][1], self.lbls[index][1]
		return self.transform(Image.open(img), bbox, lbl, self.size)


	def __len__(self):
		return len(self.imgs)


if __name__ = '__main__':

