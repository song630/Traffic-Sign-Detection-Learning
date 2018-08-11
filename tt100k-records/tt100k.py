from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg

import sys
sys.path.append('/.../lib/datasets/')
from datasets.anno_func import eval_annos # evaluation function

import os
import numpy as np
import scipy.sparse
import scipy.io
import pickle
import json
import uuid

class tt100k(imdb):
	def __init__(self, image_set, year='2016'):
		imdb.__init__(self, 'tt100k_' + image_set)
		self._year = year
		self._image_set = image_set
		# cfg.DATA_DIR: .../faster-rcnn.pytorch/data
		self._data_path = os.path.join(cfg.DATA_DIR, 'tt100k/data')
		self._image_ext = '.jpg'
		self._classes = ('__background__',
			'i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i2', 'i3', 'i4', 'i5',
			'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'io', 'ip',
			'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18',
			'p19', 'p2', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27',
			'p28', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pa10', 'pa12', 'pa13',
			'pa14', 'pa8', 'pb', 'pc', 'pg', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4',
			'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph3.8', 'ph4', 'ph4.2',
			'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'pl10', 'pl100', 'pl110',
			'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50',
			'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm1.5',
			'pm2', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55',
			'pm8', 'pn', 'pne', 'po', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45',
			'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5',
			'pw4', 'pw4.2', 'pw4.5', 'w1', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20',
			'w21', 'w22', 'w24', 'w28', 'w3', 'w30', 'w31', 'w32', 'w34', 'w35', 'w37',
			'w38', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w5',
			'w50', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w62', 'w63', 'w66', 'w8',
			'wo', 'i6', 'i7', 'i8', 'i9', 'ilx', 'p29', 'w29', 'w33', 'w36', 'w39', 'w4',
			'w40', 'w51', 'w52', 'w53', 'w54', 'w6', 'w61', 'w64', 'w65', 'w67', 'w7',
			'w9', 'pax', 'pd', 'pe', 'phx', 'plx', 'pmx', 'pnl', 'prx', 'pwx', 'w11',
			'w14', 'w15', 'w17', 'w19', 'w2', 'w23', 'w25', 'w26', 'w27', 'pl0', 'pl4',
			'pl3', 'pm2.5', 'ph4.4', 'pn40', 'ph3.3', 'ph2.6'
		)
		# construct a class map
		self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
		self._image_index = self._load_image_set_index() # to be defined later
		self._roidb_handler = self.gt_roidb
		# cleanup: will remove result files
		self.config = {'use_salt': False, 'cleanup': True}
		assert os.path.exists(self._data_path), \
			'Path does not exist: {}'.format(self._data_path)


	# get a list of indexes of all train/test images:
	def _load_image_set_index(self):
		# Load the indexes listed in this dataset's image set file
		# _image_set: 'train' or 'test'
		image_set_file = os.path.join(self._data_path, self._image_set, 'ids.txt')
		assert os.path.exists(image_set_file), \
			'Path does not exist: {}'.format(image_set_file)
		with open(image_set_file) as f:
			image_index = [x.strip() for x in f.readlines()]
		return image_index

	"""
	def _get_widths(self):
		return 2048
	"""

	# get path of annotation.json:
	def _get_gt_anno_file(self):
		return os.path.join(self._data_path, 'annotations.json')


	def _get_res_anno_file(self):
		return os.path.join(self._data_path, 'res.json') # detection result


	# get path of an image whose index is specified:
	def image_path_from_index(self, index):
		# get image path from the image's index
		image_name = index + '.jpg'
		image_path = os.path.join(self._data_path, self._image_set, image_name)
		assert os.path.exists(image_path), \
			'Path does not exist: {}'.format(image_path)
		return image_path


	def image_path_at(self, i):
		# Return the absolute path to image i in the image sequence
		return self.image_path_from_index(self._image_index[i])


	def image_id_at(self, i):
		# get index of an image
		return self._image_index[i]


	def gt_roidb(self):
		# get the database of gt RoIs
		# cache_path: defined in class imdb, which is
		# .../faster-rcnn.pytorch/data.
		# self.name: returns self._name
		cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
		if os.path.exists(cache_file): # if the cache file exists:
			with open(cache_file, 'rb') as fid:
				roidb = pickle.load(fid)
			print('{} gt roidb loaded from {}'.format(self.name, cache_file))
			return roidb

		gt_roidb = self._load_tt100k_annotation(self._image_index) # get all RoIs
		with open(cache_file, 'wb') as fid:
			pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote gt roidb to {}'.format(cache_file))
		return gt_roidb


	def _load_tt100k_annotation(self, indexes):
		results = []
		filename = self._get_gt_anno_file()
		annos = json.loads(open(filename).read())
		for ind in indexes: # traverse every indexes in train/test
			bboxes = [] # bboxes within an image
			classes = [] # classes within an image, relatively for each bbox

			image_info = annos['imgs'][ind] # get the subtree of an image
			objects = image_info['objects'] # get the subtree of objects within an image
			if objects:
				for obj in objects: # traverse every object within an image
					# convert class string to index
					classes.append(self._class_to_ind[obj['category']])
					bbox = obj['bbox']
					x1 = bbox['xmin'] - 1
					y1 = bbox['ymin'] - 1
					# === notice:
					# some coordinates is equal to 2048,
					# which get a 65535 in function append_flipped_images()'s (in imdb.py)
					# boxes[:, 2]=widths[i]-oldx1-1;
					# which will cause an abortion within the same function:
					# assert(boxes[:, 2] >= boxes[:, 0]).all()
					x2 = bbox['xmax'] - 1
					if x2 >= 2048:
						x2 = 2047
					y2 = bbox['ymax'] - 1
					if y2 >= 2048:
						y2 = 2047
					bboxes.append([x1, y1, x2, y2])
			n_objs = len(objects)

			boxes = np.array(bboxes, dtype=np.uint16)
			gt_classes = np.array(classes, dtype=np.int32)
			overlaps = np.zeros((n_objs, self.num_classes), dtype=np.float32)
			for i in range(n_objs):
				overlaps[i, classes[i]] = 1.0
			overlaps = scipy.sparse.csr_matrix(overlaps)
			seg_areas = np.zeros((n_objs), dtype=np.float32)
			results.append({ # returns an object
				'boxes': boxes,
				'gt_classes': gt_classes,
				'gt_overlaps': overlaps,
				'flipped': False,
				'seg_areas': seg_areas
			})
		return results


	# all_boxes: from test_net.py, dimension of num_classes * num_images * ??
	# all_boxes = [ [[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes) ]
	# test samples are not shuffled
	def _write_tt100k_results_file(self, all_boxes):
		results = {'imgs': {}} # the root object to be converted to json
		for cls_ind in len(all_boxes):
			for img_ind in len(all_boxes[cls_ind]):
				dets = all_boxes[cls_ind][img_ind] # detections
				# dets: output example:
				"""
				index: 009963
				dets[k, -1]: 6.887781e-05
				dets[k, 0]: 1.1987748
				dets[k, 1]: 399.96793
				dets[k, 2]: 65.66404
				dets[k, 3]: 498.92355
				index: 009963
				dets[k, -1]: 2.4918429e-05
				dets[k, 0]: 88.10372
				dets[k, 1]: 24.69757
				dets[k, 2]: 370.1334
				dets[k, 3]: 141.53722
				"""
				# terms with the same 'index' must be adjacent.
				# the same 'index' appears in every class.
				# every 'dets' is a 5-element list, it may be an empty [].
				if dets == []:
					continue
				for k in xrange(dets.shape[0]):
					index, score, xmin, ymin, xmax, ymax = \
						str(self._image_index[img_ind]), \
						dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]
					category = self._class_to_ind[cls_ind]
					obj = {'category': category, 'score': score, \
						'bbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}}
					if not results['imgs'].has_key(index):
						results['imgs'][index]['objects'] = [] # init
					results['img'][index]['objects'].append(obj) # an image has multiple objects
		filename = self._get_res_anno_file()
		with open(filename, 'w') as f:
			json.dump(results, f)
		print('writing results json file: done.')


	def evaluate_detections(self, all_boxes, output_dir):
		self._write_tt100k_results_file(all_boxes)
		gt_anno_file = self._get_anno_file()
		res_anno_file = self._get_res_anno_file()
		gt_annos = json.loads(open(gt_anno_file).read())
		res_annos = json.loads(open(res_anno_file).read())
		eval_annos(gt_annos, res_annos, imgids=self._image_index) # evaluation


	def competition_mode(self, on):
		pass
