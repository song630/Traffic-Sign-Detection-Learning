import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tf
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, models

import numpy as np
import time, os, random, math, argparse
import matplotlib.pyplot as plt
from PIL import Image

dataset_dir = '/home/storage3/traffic-sign/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/'


def parse_args():
	parser = argparse.ArgumentParser(description='Train a network')
	parser.add_argument('--epochs', dest='max_epochs',
						help='number of epochs to train',
						default=40, type=int)
	parser.add_argument('--nw', dest='num_workers',
						help='number of worker to load data', # ??
						default=0, type=int)
	parser.add_argument('--width', dest='width',
						help='crop size',
						default=480, type=int)
	parser.add_argument('--height', dest='height',
						help='crop size',
						default=320, type=int)
	parser.add_argument('--cuda', dest='cuda',
						help='whether use CUDA',
						action='store_true')
	"""
	parser.add_argument('--mGPUs', dest='mGPUs', # === ?? try this later
						help='whether use multiple GPUs',
						action='store_true')
	"""
	parser.add_argument('--bs', dest='batch_size',
						help='batch_size',
						default=32, type=int)
	parser.add_argument('--lr', dest='lr',
						help='starting learning rate',
						default=1e-2, type=float) # 1e-3 ??
	parser.add_argument('--r', dest='resume',
						help='resume checkpoint or not',
						default=False, type=bool)
	parser.add_argument('--checkepoch', dest='checkepoch',
						help='checkepoch to load model',
						default=1, type=int)
	parser.add_argument('--save_dir', dest='save_dir',
						help='directory to save models', default="/home/storage3/traffic-sign/test1/load/",
						type=str)
	return parser.parse_args()


def get_images(root=dataset_dir, train=True):
	file_name = root + 'ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
	images, labels = [], []
	with open(file_name) as f:
		for line in f.read().split():
			images.append(dataset_dir + 'JPEGImages/' + line + '.jpg')
			labels.append(dataset_dir + 'SegmentationClass/' + line + '.png')
	if len(images) != len(labels):
		raise Exception('get_images(): Invalid input.')
	return images, labels


class RandomCrop:
	def __init__(self, output_size):
		"""
		output_size: width first, then height.
		"""
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size


	def __call__(self, sample_img, sample_lbl):
		"""
		sample_img, sample_lbl: PIL.Image objects,
		should be cropped in the same way
		"""
		w, h = sample_img.size
		new_w, new_h = self.output_size
		if new_h >= h or new_w >= w:
			raise Exception('Invalid crop outputs.')
		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		sample_img = sample_img.crop((left, top, left + new_w, top + new_h))
		sample_lbl = sample_lbl.crop((left, top, left + new_w, top + new_h))
		# print('crop result:', sample_img.size, sample_lbl.size)
		return sample_img, sample_lbl


def random_crop(the_image, the_label, width, height):  # preprocess
	"""
	image, label: PIL.Image objects
	height, width: crop_sizes[0], [1]
	crop image and label together syntheticly
	"""
	the_image, rect = tf.RandomCrop((height, width))(the_image)
	the_label = tf.FixedCrop(*rect)(the_label)
	return the_image, the_label


"""
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
			'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
			'dog', 'horse', 'motorbike', 'person', 'potted plant',
			'sheep', 'sofa', 'train', 'tv/monitor']
"""
class_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
				[128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
				[64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
				[64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
				[0, 192, 0], [128, 192, 0], [0, 64, 128]]

color_map = np.zeros(256 ** 3)
for i, j in enumerate(class_colors):
	color_map[j[0] * 256 * 256 + j[1] * 256 + j[2]] = i  # 0-20


def image2label(_image):  # image: PIL.Image object
	temp = np.array(_image, dtype='int32')  # int32: RGB
	index = temp[:, :, 0] * 256 * 256 + temp[:, :, 1] * 256 + temp[:, :, 2]
	return np.array(color_map[index], dtype='int64')  # int64: after computing *255*255


def transform(_image, _label, crop_sizes):
	# image, label: PIL.Image objects
	# first crop:
	# _image, _label = random_crop(_image, _label, crop_sizes[0], crop_sizes[1])
	# RandomCrop: a callable class
	rc = RandomCrop((crop_sizes[0], crop_sizes[1]))
	_image, _label = rc(_image, _label)
	# then use tf.Compose:
	preprocess = tf.Compose([
		tf.ToTensor(),
		# normalize using stdev and mean from ImageNet:
		tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	_image = preprocess(_image)
	_label = image2label(_label)
	_label = torch.from_numpy(_label)
	return _image, _label


class VOCDataset(Dataset):
	# must redefine the 3 functions below:
	def __init__(self, crop_sizes, train=True, trans=None, root=dataset_dir):
		images, labels = get_images(root, train)
		self.crop_sizes = crop_sizes
		self.imgs, self.lbls = [], []
		# Image.size: width, height
		for k in range(len(images)):
			size = Image.open(images[k]).size  # get width & height of the image
			# filter the one that is smaller than crop_sizes
			if size[0] > crop_sizes[0] and size[1] > crop_sizes[1]:
				self.imgs.append(images[k])
				self.lbls.append(labels[k])
		if trans is None:
			self.transform = tf.Compose([  # combines all pre-processings
				tf.ToTensor(),
				# normalize using stdev and mean from ImageNet:
				tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
			print('Warning: no transformation function assigned, use default instead.')
		else:
			self.transform = transform
		print('Dataset imported successfully.')

	def __getitem__(self, index):
		_image, _label = self.imgs[index], self.lbls[index]
		# get PIL.Image objects first, then send them through self-defined transform function
		return self.transform(Image.open(_image), Image.open(_label).convert('RGB'), self.crop_sizes)

	def __len__(self):
		return len(self.imgs)


def bilinear_kernel(in_channels, out_channels, kernel_size):
	# code from Internet
	factor = (kernel_size + 1) // 2
	if kernel_size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:kernel_size, :kernel_size]
	filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
	weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
	weight[range(in_channels), range(out_channels), :, :] = filt
	return torch.from_numpy(weight)


class MyFCN(nn.Module):
	# self.defined network. must redefine the 2 funcitons below:
	def __init__(self, num_classes, pretrained_model):
		super(MyFCN, self).__init__()
		# e.g. Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
		# e.g. Conv2d(1, 10, kernel_size=5)
		# children(): return outer components
		# Sequential: quickly establish the network, given several layers
		# *: pass multi parameters
		# [:-4]: from start to the last but 5 layer
		self.stage1 = nn.Sequential(*list(pretrained_model.children())[:-4])
		# [-4]: the last but 4 layer
		self.stage2 = list(pretrained_model.children())[-4]
		# [-3]: the last but 3 layer
		# discard the last 2 layers: pooling and FC layer.
		self.stage3 = list(pretrained_model.children())[-3]

		self.conv1 = nn.Conv2d(512, num_classes, 1)  # out from layer [-3]
		self.conv2 = nn.Conv2d(256, num_classes, 1)  # out from layer [-4]
		self.conv3 = nn.Conv2d(128, num_classes, 1)  # out from layer [-5]

		# e.g. ConvTranspose2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
		# in_channels must == out_channels, since they will be added
		self.deconv8 = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
		self.deconv8.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 16: kernel size
		self.deconv4 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
		self.deconv4.weight.data = bilinear_kernel(num_classes, num_classes, 4)
		self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
		self.deconv2.weight.data = bilinear_kernel(num_classes, num_classes, 4)

		# architecture of ResNet34: 7 outer components:
		# [], [64], [128] | [256] | [512] | [pool] | [FC]
		# stage1          | stage2| stage3| discarded

	def forward(self, x):  # x: input
		s1_out = self.stage1(x)
		s2_out = self.stage2(s1_out)
		s3_out = self.stage3(s2_out)
		# 1.deconv output of stage3(it doubles)
		# 2.then add to output of stage2, deconv(it doubles)
		# 3.add to output of stage1, deconv(it doubles)
		return self.deconv8(self.conv3(s1_out) + self.deconv4(self.conv2(s2_out)
			+ self.deconv2(self.conv1(s3_out))))


def fast_hist(true_label, pred_label, n_class):
	# code from Internet
	mask = (true_label >= 0) & (true_label < n_class)
	hist = np.bincount(
		n_class * true_label[mask].astype(int) +
		pred_label[mask], minlength=n_class ** 2).reshape(n_class, n_class)
	return hist


def label_accuracy_score(label_trues, label_preds, n_class):
	"""
	code from Internet
	Returns segmentation evaluation result.
	overall accuracy, mean accuracy, mean IU, fwavacc
	"""
	hist = np.zeros((n_class, n_class))
	for lt, lp in zip(label_trues, label_preds):
		hist += fast_hist(lt.flatten(), lp.flatten(), n_class)
	temp_acc = np.diag(hist).sum() / hist.sum()
	temp_acc_cls = np.diag(hist) / hist.sum(axis=1)
	temp_acc_cls = np.nanmean(temp_acc_cls)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	temp_mean_iu = np.nanmean(iu)
	freq = hist.sum(axis=1) / hist.sum()
	temp_fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
	return temp_acc, temp_acc_cls, temp_mean_iu, temp_fwavacc


def adjust_learning_rate(optimizer, new_lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr


if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)

	input_size = [args.width, args.height]  # crop_sizes
	# init:
	datasets = [VOCDataset(input_size, True, transform, dataset_dir),
				VOCDataset(input_size, False, transform, dataset_dir)]
	# improve: change batch_sizes to 64, 128, respectively;
	# also change num_workers.
	dataloaders = [torch.utils.data.DataLoader(
		datasets[0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory='True'),
		torch.utils.data.DataLoader(
			datasets[1], batch_size=2*args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory='True')]

	imported_model = models.resnet34(pretrained=True)
	n_classes = len(class_colors)  # 21

	model = MyFCN(n_classes, imported_model)
	criterion = nn.NLLLoss2d()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
	# after every 'step_size' epoches, lr = lr * gamma
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
	if args.cuda:
		model.cuda()
	else:
		raise Exception('Should use cuda here.')

	start_epoch = 0

	if args.resume: # load state of last time:
		load_name = args.save_dir + 'fcn_{}.pth'.format(args.checkepoch)
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		# load lr of last time:
		adjust_learning_rate(optimizer, optimizer.param_groups[0]['lr'])
		print("loaded checkpoint %s" % (load_name))

	for epoch in range(start_epoch, args.max_epochs):
		train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc = 0, 0, 0, 0, 0
		start = time.time()
		# always add model.train() before training, add model.eval() before test
		# these 2 methods deal with the case where training/test are different.
		model.train()
		for data in dataloaders[0]:  # i: iteration
			inputs = Variable(data[0].cuda())
			targets = Variable(data[1].cuda())
			# forward:
			outputs = model(inputs)
			# log_softmax: output a probability distribution
			# after softmax: batch * n_classes * height * width
			outputs = torch.nn.functional.log_softmax(outputs, dim=1)  # predict (21 classes)
			loss = criterion(outputs, targets)
			# backward:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.data[0]
			# decide the class by selecting the maximum from the distribution
			# code from Internet:
			label_pred = outputs.max(dim=1)[1].data.cpu().numpy()
			label_true = targets.data.cpu().numpy()
			for lbt, lbp in zip(label_true, label_pred):
				acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, n_classes)
				train_acc += acc
				train_acc_cls += acc_cls
				train_mean_iu += mean_iu
				train_fwavacc += fwavacc
		model.eval()
		eval_loss, eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc = 0, 0, 0, 0, 0
		for data in dataloaders[1]:  # i: iteration
			inputs = Variable(data[0].cuda(), volatile=True)
			targets = Variable(data[1].cuda(), volatile=True)
			# forward:
			outputs = model(inputs)
			outputs = torch.nn.functional.log_softmax(outputs, dim=1)
			loss = criterion(outputs, targets)
			# backward:
			# optimizer.zero_grad()
			# loss.backward()
			# optimizer.step()
			eval_loss += loss.data[0]

			label_pred = outputs.max(dim=1)[1].data.cpu().numpy()
			label_true = targets.data.cpu().numpy()
			for lbt, lbp in zip(label_true, label_pred):
				acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, n_classes)
				eval_acc += acc
				eval_acc_cls += acc_cls
				eval_mean_iu += mean_iu
				eval_fwavacc += fwavacc
		# print info of epoch:
		end = time.time()
		print('After epoch {}/{}, \
			train loss: {:.5f}, train acc: {:.5f}, train mean iu: {:.5f}, \
			valid loss: {:.5f}, valid acc: {:.5f}, valid mean iu: {:.5f}, time: {:.2f}'.format(
			epoch + 1, args.max_epochs,
			train_loss / len(dataloaders[0]),
			train_acc / len(datasets[0]),
			train_mean_iu / len(datasets[0]),
			eval_loss / len(dataloaders[1]),
			eval_acc / len(datasets[1]),
			eval_mean_iu / len(datasets[1]),
			end - start))
		# save state:
		if (epoch + 1) % 5 == 0:
			save_name = args.save_dir + 'fcn_{}.pth'.format(epoch)
			torch.save({
				'epoch': epoch + 1,
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict()},
				save_name
			)
			print('epoch {} saved'.format(epoch + 1))
