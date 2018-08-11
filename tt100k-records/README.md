### Run Faster-RCNN on TT100K
---
* Dataset: TT100K
* The tests are based on the source code from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).
* To import a specified dataset, the calling stack is like:
1. In trainval_net.py, `args.imdb_name`, `args.imdbval_name`, `args.set_cfgs` are defined.
2. In trainval_net.py, `combined_roidb(args.imdb_name)` is called.
3. In function `get_roidb` in roidb.py, function `get_imdb` is called.
4. In factory.py, the ctor of dataset is called, and datasets are imported globally (into `__sets`).
5. In imdb.py, `imdb` is a base dataset class. It gets `roidb` of specified dataset class.
6. In tt100k.py, class `tt100k`, inherited `imdb`.
* tt100k.py is in parallel with pascal_voc.py, coco.py, etc.
* How to get start on TT100K:
1. tt100k.py.
2. Insert the following lines into trainval_net.py:
```
elif args.dataset == 'tt100k':
	args.imdb_name = 'tt100k_train'
	args.imdb_val_name = 'tt100k_test'
	args.set_cfgs = ['ANCHOR_SCALES', '[16, 32, 64, 128]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
```
3. Insert the following lines into factory.py:
```
for year in ['2016']:
	for split in ['train', 'test']:
		name = 'tt100k_{}'.format(split)
		__sets[name] = (lambda split=split, year=year: tt100k(split, year))
```
* If tt100k.py has been modified, remember to delete the `.pkl` file in cache.
* The evaluation method `eval_annos` is from code provided on index page of TT100K.
* The anchor parameters must be adjusted, or the loss will be NA.
