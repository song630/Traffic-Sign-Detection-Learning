import json
import sys
sys.path.append('.../lib/datasets/')
from anno_func import eval_annos # evaluation function
from anno_func import type45

if __name__ == '__main__':
	gt_annos = json.loads(open('.../tt100k/data/annotations.json').read())
	res_annos = json.loads(open('.../tt100k/data/res3.json').read())
	image_set_file = '.../tt100k/data/test/ids.txt'
	with open(image_set_file) as f:
		image_index = [x.strip() for x in f.readlines()]
	eval_annos(gt_annos, res_annos, iou=0.5, types=type45, imgids=image_index)
