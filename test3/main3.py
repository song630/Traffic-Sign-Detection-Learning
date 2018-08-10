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


def get_train_test(root=data_dir):
	file_name = root + 'train_test_split.txt'
	# train_indexes, test_indexes: stores the indexes of the images
	train_indexes, test_indexes = [], []
	with open(file_name) as f:
		for line in f.read().splitlines():
			# every line in the file is like: <image_id> <is_train_set>,
			# if <is_train_set> == 1, then it belongs to training set.
			# notice: index starts from 1.
			index, is_train = line.split(' ')
			# === drop images with index larger then 11729:
			# === some images of class 200 seem to have unknown problems.
			if int(index) >= 11729: break
			if is_train == '1':
				train_indexes.append(int(index))
			else:
				test_indexes.append(int(index))
	return train_indexes, test_indexes


def get_imgs(train_indexes, test_indexes, root=data_dir):
	file_name = data_dir + 'images.txt'
	paths = [] # stores all relative paths of images
	paths.append('none') # paths[0] is dummy
	with open(file_name) as f:
		for line in f.read().splitlines():
			paths.append(line.split(' ')[1])
	# absolute paths of training images
	train_images = [(i, data_dir + 'images/' + paths[i]) for i in train_indexes]
	test_images = [(i, data_dir + 'images/' + paths[i]) for i in test_indexes]
	return train_images, test_images


def get_bboxes(train_indexes, test_indexes, root=data_dir):
	file_name = root + 'bounding_boxes.txt'
	train_bbox, test_bbox = [], []
	# construct a 2-dimentional array, it has (11788 + 1) * 4 float elements
	all_bbox = []
	all_bbox.append([0.0, 0.0, 0.0, 0.0]) # the first row [0] is dummy
	with open(file_name) as f:
		for line in f.read().splitlines():
			_, x, y, w, h = line.split(' ')
			all_bbox.append([float(x), float(y), float(w), float(h)])
	# elements in train bbox: (index, [x, y, w, h])
	train_bbox = [(i, all_bbox[i]) for i in train_indexes]
	test_bbox = [(i, all_bbox[i]) for i in test_indexes]
	return train_bbox, test_bbox # convert to Tensor later in class Dataset's __getitem__()


def get_lbls(train_indexes, test_indexes, root=data_dir): # ge the classification labels
	# class_file_name = root + 'classes.txt'
	label_file_name = root + 'image_class_labels.txt'
	# classes_map = []
	# classes_map.append('none') # dummy at classes_map[0]
	all_labels = []
	all_labels.append('none') # dummy at all_labels[0]
	"""
	with open(class_file_name) as f:
		for line in f.read().split():
			classes_map.append(line.split(' ')[1])
	"""
	with open(label_file_name) as f:
		for line in f.read().splitlines():
			all_labels.append(int(line.split(' ')[1]))
	train_labels = [(i, all_labels[i]) for i in train_indexes]
	test_labels = [(i, all_labels[i]) for i in test_indexes]
	return train_labels, test_labels


train_indexes, test_indexes = get_train_test(data_dir)
# element format: (index, absolute path)
train_images, test_images = get_imgs(train_indexes, test_indexes, data_dir)
# element format: (index, [x, y, w, h])
train_bbox, test_bbox = get_bboxes(train_indexes, test_indexes, data_dir)
# element format: (index, class)
train_labels, test_labels = get_lbls(train_indexes, test_indexes, data_dir)


def data_transform(img, bbox, lbl, size=224):
	# first deal with img:
	#print('in data_transform:')
	width, height = img.size
	preprocess = tf.Compose([
		tf.Resize((size, size)), # ?? the ratio of original image has been modified
		tf.ToTensor(),
		tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	img = preprocess(img)
	# then deal with bbox:
	w_ratio = size / width # ?? use float to improve accuracy (=== /: get float)
	h_ratio = size / height
	bbox = [bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio]
	# after the transformation, bbox may be beyond the preprocessed img
	bbox[0] = max(bbox[0], 0)
	bbox[1] = max(bbox[1], 0)
	bbox[2] = min(size - 1 - bbox[0], bbox[2])
	bbox[3] = min(size - 1 - bbox[1], bbox[3])
	bbox = torch.FloatTensor(bbox) # convert to float tensor
	# === this line has been deleted: lbl = torch.LongTensor([lbl]),
	# === since targets2 will be LongTensor of 32*1, but it should be 32.
	return img, bbox, lbl


def compute_acc(pred_bbox, target_bbox, size=224, thres=0.75):
	# first limit pred_bbox (must be within the image)
	# ?? should be converted to int
	for i in pred_bbox: #  traverse every bbox in tensor pred_bbox
		i[0], i[1] = max(i[0], 0), max(i[1], 0)
		i[2], i[3] = min(size - 1 - i[0], i[2]), min(size - 1 - i[1], i[3])
	max_x = torch.max(pred_bbox[:, 0], target_bbox[:, 0])
	max_y = torch.max(pred_bbox[:, 1], target_bbox[:, 1])
	min_x1 = torch.min(pred_bbox[:, 0] + pred_bbox[:, 2], target_bbox[:, 0] + target_bbox[:, 2])
	min_y1 = torch.min(pred_bbox[:, 1] + pred_bbox[:, 3], target_bbox[:, 1] + target_bbox[:, 3])
	intersec = (min_x1 - max_x + 1) * (min_y1 - max_y + 1)
	union = pred_bbox[:, 2] * pred_bbox[:, 3] + target_bbox[:, 2] * target_bbox[:, 3] - intersec
	IoU = intersec / union
	count = pred_bbox.size()[0]
	return (IoU >= thres).sum() / count


class CUBDataset(Dataset):
	def __init__(self, size=224, train=True, trans=data_transform, root=data_dir):
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
		# aside from img, also convert bbox and lbl to tensors
		# === must convert to RGB, since some pics contains only 1 channel.
		return self.transform(Image.open(img).convert('RGB'), bbox, lbl, self.size)


	def __len__(self):
		return len(self.imgs)


class DetectionNet(nn.Module):
	def __init__(self, pre_model, num_classes=200): # pre_model: resnet18
		super(DetectionNet, self).__init__()
		self.head = nn.Sequential(*list(pre_model.children())[:-1]) # discard the final fc layer
		self.n_features = pre_model.fc.in_features
		# ?? convert the fc layers to conv layers later
		self.fc1 = nn.Linear(self.n_features, 4) # predict the 4 coordinates of bbox
		self.fc2 = nn.Linear(self.n_features, num_classes) # predict which class the input belongs to


	def forward(self, x):
		x = self.head(x) # first get features after resnet18
		x = x.view(-1, self.n_features) # flattended before entering fc layers
		return self.fc1(x), self.fc2(x) # return bbox, class distribution


if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)

	in_size = args.resize # default: 224
	train_set = CUBDataset(in_size, True, data_transform, data_dir)
	test_set = CUBDataset(in_size, False, data_transform, data_dir)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
		shuffle=True, num_workers=args.num_workers)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
		shuffle=True, num_workers=args.num_workers)

	n_images = 11788
	n_classes = 200

	if args.use_self:
		model = ResNet18()
	else: # default
		model = models.resnet18(pretrained=True)
	net = DetectionNet(model, n_classes)
	criterion1 = nn.SmoothL1Loss() # predicted bbox
	criterion2 = nn.NLLLoss() # === updated
	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

	if args.cuda:
		net.cuda()
	else:
		raise Exception('Should use cuda here.')

	start_epoch = 0

	if args.resume:
		load_name = args.save_dir + 'DetectionNet_{}.pth'.format(args.checkepoch)
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		start_epoch = checkpoint['epoch']
		net.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		# load lr of last time:
		adjust_learning_rate(optimizer, optimizer.param_groups[0]['lr'])
		print("loaded checkpoint %s" % (load_name))

	best_acc = 0.0 # online, update

	for epoch in range(start_epoch, args.max_epochs): # in every epoch, do training and test
		train_loss, train_acc = 0.0, 0.0
		start = time.time()
		net.train()
		for img, bbox, lbl in train_loader:
			if args.cuda:
				inputs = Variable(img.cuda())
				targets1 = Variable(bbox.cuda())
				targets2 = Variable(lbl.cuda())
			else:
				raise Exception('No cuda')
			outputs1, outputs2 = net(inputs)
			outputs2 = nn.functional.log_softmax(outputs2, dim=1) # === updated
			# loss1, loss2: FloatTensor of size 1, so are loss1.mean() and loss2.mean()
			loss1 = criterion1(outputs1, targets1) # bbox
			loss2 = criterion2(outputs2, targets2) # classification
			loss = loss1.mean() + loss2.mean()
			# backwards:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss = loss.data[0]
			train_acc = compute_acc(outputs1.cpu().data, targets1.cpu().data, args.resize)

		test_loss, test_acc = 0.0, 0.0
		net.eval()
		for img, bbox, lbl in test_loader:
			if args.cuda:
				inputs = Variable(img.cuda())
				targets1 = Variable(bbox.cuda())
				targets2 = Variable(lbl.cuda())
			else:
				raise Exception('No cuda')
			outputs1, outputs2 = net(inputs)
			outputs2 = nn.functional.log_softmax(outputs2, dim=1)
			loss1 = criterion1(outputs1, targets1) # bbox
			loss2 = criterion2(outputs2, targets2) # classification
			loss = loss1.mean() + loss2.mean()
			test_loss = loss.data[0]
			test_acc = compute_acc(outputs1.cpu().data, targets1.cpu().data, args.resize)

		#if test_acc > best_acc:
			#best_acc = test_acc
			# save current state:
		save_name = args.save_dir + 'DetectionNet_{}.pth'.format(epoch)
		torch.save({
			'epoch': epoch + 1,
			'model': net.state_dict(),
			'optimizer': optimizer.state_dict()},
			save_name
		)
		print('updated, epoch {} saved'.format(epoch + 1))

		end = time.time()
		print('after epoch {}/{}, train_loss: {:.3f}, train_acc: {:.2f}, test_loss: {:.3f}, test_acc: {:.2f}, time: {:.2f}'.format(epoch + 1, args.max_epochs, train_loss, train_acc, test_loss, test_acc, end - start))
