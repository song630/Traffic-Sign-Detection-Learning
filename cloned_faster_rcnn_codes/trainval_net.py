# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

"""
总结:
1.首先处理若干参数
2.定义smapler取样器类 分割batches 经过一系列张量处理得到样本中所有元素的对应下标组成的矩阵
3.训练前的配置:
  决定使用的数据集
  计算roidb中所有roi的宽高比并确定序号
  构造一个取样器实例
  导入训练集
  ship to cuda
  确定faster-rcnn使用的网络模型
  建立整个模型
  确定优化方法
  加载之前没训练完的数据
4.训练
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
	  adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  # dest: 这个参数相当于把位置或者选项关联到一个特定的名字
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset', # 使用哪个数据集
					  help='training dataset',
					  default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net', # 使用哪个模型
					help='vgg16, res101',
					default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
					  help='starting epoch',
					  default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs', # 多少个epoch
					  help='number of epochs to train',
					  default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
					  help='number of iterations to display',
					  default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
					  help='number of iterations to display',
					  default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
					  help='directory to save models', default="/srv/share/jyang375/models",
					  type=str)
  parser.add_argument('--nw', dest='num_workers', # 快速加载数据 和GPU内存有关
					  help='number of worker to load data',
					  default=0, type=int)
  # action参数: 6种动作可以在解析到一个参数时进行触发
  # store_ture/store_false 保存相应的布尔值 这两个动作被用于实现布尔开关
  parser.add_argument('--cuda', dest='cuda',
					  help='whether use CUDA',
					  action='store_true')
  parser.add_argument('--ls', dest='large_scale',
					  help='whether use large imag scale',
					  action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
					  help='whether use multiple GPUs',
					  action='store_true')
  parser.add_argument('--bs', dest='batch_size', # 和GPU内存有关
					  help='batch_size',
					  default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
					  help='whether perform class_agnostic bbox regression',
					  action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer', # 优化器
					  help='training optimizer',
					  default="sgd", type=str)
  parser.add_argument('--lr', dest='lr', # 初始学习率
					  help='starting learning rate',
					  default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step', # 学习率过多少个epoch衰减
					  help='step to do learning rate decay, unit is epoch',
					  default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', # 学习率衰减的倍数
					  help='learning rate decay ratio',
					  default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
					  help='training session',
					  default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
					  help='resume checkpoint or not', # 继续运行之前没运行完的model
					  default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
					  help='checksession to load model',
					  default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
					  help='checkepoch to load model',
					  default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
					  help='checkpoint to load model',
					  default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
					  help='whether use tensorflow tensorboard',
					  default=False, type=bool)

  args = parser.parse_args()
  return args

# ============== 上面是解析参数 ===============

class sampler(Sampler): # 取样器?
  def __init__(self, train_size, batch_size):
	self.num_data = train_size
	self.num_per_batch = int(train_size / batch_size) # batch数?
	self.batch_size = batch_size
	# arange: 返回一个1维张量 长度为floor((end − start)/step) 为一个数阶
	# view(x, y): 改变数组形状为x * y 下面的代码: [1, 2, 3]->[[1, 2, 3]]
	# long(): 浮点数转换为整数
	self.range = torch.arange(0, batch_size).view(1, batch_size).long()
	self.leftover_flag = False # 余数 训练集分割成batches最后剩下的
	if train_size % batch_size: # 没有剩下的
	  # lower bound: batch数 * batch大小
	  # upper bound: train_size
	  # 若两者相等 则返回[]
	  self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
	  self.leftover_flag = True # 没有剩下的

  def __iter__(self): # 迭代器?
  	# randperm: 返回一个随机排列 在这里是把所有的batch打乱后变为列向量
  	# view中有-1时会变为列向量:
  	"""
  	tensor([ 3,  0,  1,  2,  4])
  	tensor([[ 3],
        	[ 0],
        	[ 1],
        	[ 2],
        	[ 4]])
	"""
	# 乘上batch_size后变为每一个batch中的起始元素索引
	rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
	# expand: e.g. 上面的列向量中每个元素变为100个:
	# tensor([[ 3, 3, 3...],
	# 		  [ 0, 0, 0...],
	# 加上range后就得到了样本中所有元素的对应下标组成的矩阵
	self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
	# 类似于列向量转换回行向量
	self.rand_num_view = self.rand_num.view(-1)

	if self.leftover_flag:
	  self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0) # 第二个参数表示在第几个维度上连接

	return iter(self.rand_num_view)

  def __len__(self):
	return self.num_data

# ============= main ===============

if __name__ == '__main__':

  args = parse_args() # 处理参数

  print('Called with args:')
  print(args)

  if args.use_tfboard: # 可视化
	from model.utils.logger import Logger
	# Set the logger
	logger = Logger('./logs')

  # 决定使用的数据集
  # CFG: 网络配置脚本文件 .yml
  if args.dataset == "pascal_voc":
	  args.imdb_name = "voc_2007_trainval" # 训练集
	  args.imdbval_name = "voc_2007_test" # 测试集
	  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
	  args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
	  args.imdbval_name = "voc_2007_test"
	  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
	  args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
	  args.imdbval_name = "coco_2014_minival"
	  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
	  args.imdb_name = "imagenet_train"
	  args.imdbval_name = "imagenet_val"
	  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
	  # train sizes: train, smalltrain, minitrain
	  # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
	  args.imdb_name = "vg_150-50-50_minitrain"
	  args.imdbval_name = "vg_150-50-50_minival"
	  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  # .yml: yet another markup language 其中的冒号表示键值对
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
	cfg_from_file(args.cfg_file) # 做某些配置操作
  if args.set_cfgs is not None:
	cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg) # 输出配置情况
  np.random.seed(cfg.RNG_SEED) # (For reproducibility) ??

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading. 加速

  # cfg: configuration
  # whether Use horizontally-flipped images during training ??
  # 使用图像翻转来增加训练集
  cfg.TRAIN.USE_FLIPPED = True
  # Use GPU implementation of non-maximum suppression
  # 用GPU实现非极大值抑制(类似局部最大搜索)
  # 在应用滑动窗口确定bbox时使用
  cfg.USE_GPU_NMS = args.cuda
  # roidb: Region of Interest database
  # ratio_list, ratio_index: 计算roidb中所有roi的宽高比并确定序号
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
	os.makedirs(output_dir) # 创建目录

  # 构造一个取样器实例
  sampler_batch = sampler(train_size, args.batch_size)
  # 导入roi的训练集
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
						   imdb.num_classes, training=True)
  # 常规操作 导入训练集
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
							sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
	im_data = im_data.cuda()
	im_info = im_info.cuda()
	num_boxes = num_boxes.cuda()
	gt_boxes = gt_boxes.cuda()

  # make variable
  # 读取到数据后将数据从Tensor转换成Variable格式
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
	cfg.CUDA = True

  # initilize the network here.
  # 确定faster-rcnn使用的网络模型
  # 几个参数的含义??
  if args.net == 'vgg16':
	fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
	fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
	fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
	fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
	print("network is not defined")
	pdb.set_trace() # Program Data Base 交互的源代码调试功能
	# 运行时会停留在这里 此时可以使用一些命令

  fasterRCNN.create_architecture()  # 建立整个模型
  # 学习率
  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  # named_parameters??
  # items(): 返回字典的键值元组数组
  for key, value in dict(fasterRCNN.named_parameters()).items():
	if value.requires_grad: # 需要梯度??
	  if 'bias' in key:
	  	# DOUBLE_BIAS: Whether to double the learning rate for bias
	  	# BIAS_DECAY: Whether to have weight decay on bias as well
	  	# WEIGHT_DECAY: Weight decay, for regularization
	  	# 权重衰减: 过拟合 给误差函数添加惩罚项
		params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
				'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
	  else:
		params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  # 优化器
  if args.optimizer == "adam":
	lr = lr * 0.1
	# Adam: Adaptive Moment Estimation
	# 利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率
	optimizer = torch.optim.Adam(params)
  elif args.optimizer == "sgd":
  	# MOMENTUM: 动量
  	# 损失函数在训练过程中可能陷入局部最小值 用动量消除不确定性
  	# SGD指mini-batch gradient descent,
  	# 每次迭代计算mini-batch的梯度 然后对参数更新,
  	# 完全依赖于当前batch的梯度 可理解为允许当前batch的梯度多大程度影响参数更新
	optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  # 加载之前没训练完的数据
  if args.resume:
  	# 加载
  	# session??
	load_name = os.path.join(output_dir,
	  'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
	print("loading checkpoint %s" % (load_name))
	checkpoint = torch.load(load_name)
	args.session = checkpoint['session'] # ??
	args.start_epoch = checkpoint['epoch'] # 从哪个epoch开始
	# checkpoint中用不同字段保存了上一次的状态和配置
	fasterRCNN.load_state_dict(checkpoint['model']) # 上一次使用的网络模型
	optimizer.load_state_dict(checkpoint['optimizer']) # 上一次使用的优化方法
	# 设置参数组的学习率
	lr = optimizer.param_groups[0]['lr']
	if 'pooling_mode' in checkpoint.keys(): # 默认为crop??
	  cfg.POOLING_MODE = checkpoint['pooling_mode']
	print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
	fasterRCNN = nn.DataParallel(fasterRCNN) # ??

  if args.cuda:
	fasterRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size) # 每个epoch有多少次迭代

# ============ configuration done. begin training =============

  for epoch in range(args.start_epoch, args.max_epochs + 1): # 开始训练
	# setting to train mode
	fasterRCNN.train() # ?? 再看
	loss_temp = 0
	start = time.time()

	if epoch % (args.lr_decay_step + 1) == 0:  # 经过若干epoch后 学习率衰减
		adjust_learning_rate(optimizer, args.lr_decay_gamma)
		lr *= args.lr_decay_gamma

	# num_classes: (对bbox中物体分类)有多少类别
	# roidb: rigion of interest的数据库 有bbox的ground-truth和对应的label
	"""
	dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
						   imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
							sampler=sampler_batch, num_workers=args.num_workers)
	"""

	data_iter = iter(dataloader)  # 遍历之前的sampler (有一个包含了每个batch中所有元素索引的矩阵)
	for step in range(iters_per_epoch):  # 一个epoch中的每次迭代
	  data = next(data_iter) # 取下次迭代的数据
	  # ===== Tensor基本操作
	  # 任何要原地改动tensor的操作都需要加_ 比如x.copy_(y), x.t_(), x.add_(y) will change x
	  # =====
	  im_data.data.resize_(data[0].size()).copy_(data[0])
	  im_info.data.resize_(data[1].size()).copy_(data[1])
	  gt_boxes.data.resize_(data[2].size()).copy_(data[2]) # gt是梯度??
	  num_boxes.data.resize_(data[3].size()).copy_(data[3])

	  fasterRCNN.zero_grad() # 把模型的参数梯度清0 ??
	  # cls: class
	  # 7个参数
	  # 把4条数据输入模型之后 得到对应的7个输出
	  # 1.rois: regions
	  # 2.cls_prob: 属于不同类别的概率
	  # 3.bbox_pred: 预测的bbox
	  # 4.rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox: 模型共有4个loss函数
	  # 5.rois_label: rigions都属于哪个label
	  # ?? 再看 函数在lib/model/faster_rcnn/faster_rcnn.py
	  rois, cls_prob, bbox_pred, \
	  rpn_loss_cls, rpn_loss_box, \
	  RCNN_loss_cls, RCNN_loss_bbox, \
	  rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

	  # 由4个loss函数计算出的loss
	  loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
		   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
	  loss_temp += loss.data[0] # ??

	  # backward
	  optimizer.zero_grad() # 把模型的参数梯度清0
	  loss.backward() # 反向传递
	  if args.net == "vgg16":
		  clip_gradient(fasterRCNN, 10.)
	  optimizer.step() # 固定步骤

	  if step % args.disp_interval == 0: # 该显示出训练情况
		end = time.time()
		if step > 0:
		  loss_temp /= args.disp_interval

		if args.mGPUs:
		  # 4个loss function的值
		  loss_rpn_cls = rpn_loss_cls.mean().data[0]
		  loss_rpn_box = rpn_loss_box.mean().data[0]
		  loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
		  loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
		  # foreground, background??
		  fg_cnt = torch.sum(rois_label.data.ne(0))
		  bg_cnt = rois_label.data.numel() - fg_cnt
		else:
		  loss_rpn_cls = rpn_loss_cls.data[0]
		  loss_rpn_box = rpn_loss_box.data[0]
		  loss_rcnn_cls = RCNN_loss_cls.data[0]
		  loss_rcnn_box = RCNN_loss_bbox.data[0]
		  fg_cnt = torch.sum(rois_label.data.ne(0))
		  bg_cnt = rois_label.data.numel() - fg_cnt

		print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
								% (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
		print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
		# 4个loss function的值
		print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
					  % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
		if args.use_tfboard:
		  info = {
			'loss': loss_temp,
			'loss_rpn_cls': loss_rpn_cls,
			'loss_rpn_box': loss_rpn_box,
			'loss_rcnn_cls': loss_rcnn_cls,
			'loss_rcnn_box': loss_rcnn_box
		  }
		  for tag, value in info.items():
			logger.scalar_summary(tag, value, step)

		loss_temp = 0
		start = time.time()

	# 保存这次epoch训练完后的结果
	if args.mGPUs:
	  save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
	  save_checkpoint({
		'session': args.session,
		'epoch': epoch + 1,
		'model': fasterRCNN.module.state_dict(),
		'optimizer': optimizer.state_dict(),
		'pooling_mode': cfg.POOLING_MODE,
		'class_agnostic': args.class_agnostic,
	  }, save_name)
	else:
	  save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
	  save_checkpoint({
		'session': args.session,
		'epoch': epoch + 1,
		'model': fasterRCNN.state_dict(),
		'optimizer': optimizer.state_dict(),
		'pooling_mode': cfg.POOLING_MODE,
		'class_agnostic': args.class_agnostic,
	  }, save_name)
	print('save model: {}'.format(save_name))

	end = time.time()
	print(end - start)
