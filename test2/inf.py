"""
input: several images from Internet of size 224
output: predict the class
10 classes: 0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer,
			5-dog, 6-frog, 7-horse, 8-ship, 9-truck
pre-trained model: torchvision.models.squeezenet1_1
sizes of the given images:
500*241, 500*313, 500*375, 533*300, 500*333, 511*300, 533*300
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tf
from torch.autograd import Variable
from torchvision import models

import numpy as np
from PIL import Image

import sys
sys.path.append('/home/storage3/traffic-sign/test2/')
from main2 import SimpleNet

home_dir = '/home/storage3/traffic-sign/test2/'


def inf_transform(image):
	# image: PIL.Image object
	w, h = image.size
	preprocess = tf.Compose([
		tf.CenterCrop(min(w, h)),
		tf.Resize(32),
		tf.ToTensor(),
		tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
	])
	return preprocess(image)


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == '__main__':
	net = SimpleNet() # num_classes=10
	"""
	w/o the line below:
	RuntimeError: Input type (CUDAFloatTensor) and weight type (CPUFloatTensor) should be the same
	"""
	net.cuda()

	# load parameters into the model:
	load_name = home_dir + 'SimpleNet_38.pth'
	print("loading checkpoint %s" % (load_name))
	checkpoint = torch.load(load_name)
	net.load_state_dict(checkpoint['model'])
	print("loaded checkpoint %s" % (load_name))

	num_pics = 7 # 7 pictures to infer
	images = [home_dir + 'images/image' + str(i) + '.jpg' for i in range(1, num_pics + 1)]

	for j in images: # traverse every images
		inputs = inf_transform(Image.open(j)).float() # open image, transform, and convert to tensor
		inputs = inputs.unsqueeze_(0) # === add one dimension, since net treats inputs as batch
		net.eval()
		inputs = Variable(inputs.cuda())
		outputs = net(inputs) # expect a 1*10 vector
		"""
		At first: index = outputs.data.numpy().argmax()
		can't convert CUDA tensor to numpy (it doesn't support GPU arrays).
		Use .cpu() to move the tensor to host memory first.
		"""
		index = outputs.data.cpu().numpy().argmax()
		print('predicted class:', classes[index])

"""
run:
CUDA_VISIBLE_DEVICES=3 python inf.py
loading checkpoint /home/storage3/traffic-sign/test2/SimpleNet_38.pth
loaded checkpoint /home/storage3/traffic-sign/test2/SimpleNet_38.pth
predicted class: 1
predicted class: 3
predicted class: 9
predicted class: 0
predicted class: 6
predicted class: 8
predicted class: 4
acc: 100%
"""
