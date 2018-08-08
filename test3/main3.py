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
data_dir = 'data/CUB200/CUB_200_2011/'
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


def get_images(root = home_dir + data_dir):
	file_name = root + 'train_test_split.txt'
	# train_images, test_images: stores the indexes of the images
	train_images, test_images = [], []
	with open(file_name) as f:
		for line in f.read().split():
			# every line in the file is like: <image_id> <is_train_set>,
			# if <is_train_set> == 1, then it belongs to training set.
			# notice: index starts from 1.
			index, is_train = line.split(' ')
			if is_train:
				train_images.append(index)
			else:
				test_images.append(index)
	return train_images, test_images


def get_bboxes(root = home_dir + data_dir):
	file_name = root + 'bounding_boxes.txt'
	train_bbox, test_bbox = [], []
	all_bbox = np.array(11788 + 1, 4) # construct a 2-dimentional array
	all_bbox[0] = [0.0, 0.0, 0.0, 0.0] # the first row [0] is dummy
	with open(file_name) as f:
		for i, line in enumerate(f.read().split()): # i starts from 0
			index, x, y, w, h = line.split()
			all_bbox[i + 1] = [x, y, w, h]


if __name__ = '__main__':

