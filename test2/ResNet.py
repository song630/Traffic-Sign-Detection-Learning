import torch
import torch.nn as nn

class ResModule(nn.Module):
	def __init__(self, in_chan, out_chan, down_scale=False):
		super(ResModule, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=in_chan,
			kernel_size=3, stride=2 if down_scale else 1, padding=1)
		self.norm1 = nn.BatchNorm2d(in_chan)
		# inplace: override input with output, save grad when in back propagation
		self.relu1 = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
			kernel_size=3, stride=1, padding=1)
		self.norm2 = nn.BatchNorm2d(out_chan)
		self.relu2 = nn.ReLU()
		self.shortcut = nn.Sequential()
		if down_scale or in_chan != out_chan:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3,
					stride=2 if down_scale else 1, padding=1),
				nn.BatchNorm2d(out_chan)
			)

	def forward(self, x):
		out1 = self.relu1(self.norm1(self.conv1(x)))
		# add x before the second relu
		out2 = self.relu2(self.norm2(self.conv2(out1)) + self.shortcut(x))
		return out2


class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18, self).__init__()
		# ResNet-18 structure: https://blog.csdn.net/sunqiande88/article/details/80100891
		self.head = nn.Sequential(
			# 3: 3 channels(RGB)
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU()
		)
		self.block1_1 = ResModule(64, 64)
		self.block1_2 = ResModule(64, 128)
		self.block2_1 = ResModule(128, 128, down_scale=True)
		self.block2_2 = ResModule(128, 256)
		self.block3_1 = ResModule(256, 256, down_scale=True)
		self.block3_2 = ResModule(256, 512)
		self.block4_1 = ResModule(512, 512, down_scale=True)
		self.block4_2 = ResModule(512, 512)
		self.pool = nn.AvgPool2d(kernel_size=4) # 4*4 -> 1*1
		self.fc = nn.Linear(512, 10) # output: 10 classes

	def forward(self, x):
		#print('1.', x.size())
		out = self.head(x)
		#print('2.', out.size())
		out = self.block1_2(self.block1_1(out))
		#print('3.', out.size())
		out = self.block2_2(self.block2_1(out))
		#print('4.', out.size())
		out = self.block3_2(self.block3_1(out))
		#print('5.', out.size())
		out = self.block4_2(self.block4_1(out))
		#print('6.', out.size())
		# after avg: 1*1*512, should be flattened
		return self.fc(self.pool(out).view(-1, 512))
