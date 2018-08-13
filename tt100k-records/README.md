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

### How to get start on TT100K
---
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

## Deal with the case where loss became nan
1. This is mainly due to some coordinates provided by annotations.json being out of bound.
2. It is like:
```
[epoch  1][iter  700/6105] loss: nan, lr: 4.00e-03
fg/bg=(7/249), rpn_cls: 0.4264, rpn_box: nan, rcnn_cls: 0.2279, rcnn_box 0.0000
```
3. 5 possible reasons:
* `xmin` / `ymin` is less than 0.0; `xmax` / `ymax` is greater than 2047.0 (in case 2047.x becomes 2048.0) (the size of images in tt100k is 2048);
* `xmin == xmax` or `ymin == ymax` (not likely);
* `lr` should be smaller;
* Inappropriate anchor scales;
* Box sizes should be restricted.
4. Steps:
* In function `_load_tt100k_annotation` in tt100k.py, insert following lines:
```
if x1 < 0: x1 = 0.0
if y1 < 0: y1 = 0.0
if x2 >= 2047: x2 = 2047.0
if y2 >= 2047: y2 = 2047.0
```
* In function `append_flipped_images` in imdb.py, insert following lines:
```
for j in range(len(boxes)):
    if boxes[j][2] > 2047:
        boxes[j][2] = 2047
    if boxes[j][3] > 2047:
        boxes[j][3] = 2047

```
* (Optional) Replace line 190 of roibatchLoader.py with:
```
not_keep = ((gt_boxes[:,2] - gt_boxes[:,0]) < 15) & ((gt_boxes[:,3] - gt_boxes[:,1]) < 15)
```

## Results:
No. | lr | ep | box limit | 0.1 | 0.2 | 0.3 | 0.4 | 0.5
----|----|----| ---- | ---- | ---- | ---- | ---- | ----
run2 | 5e-4 | 8  | False | ac: 28.43<br>re: 52.29 | ac: 40.37<br>re: 46.62 | ac: 49.57<br>re: 42.35 | ac: 57.57<br>re: 38.55 | ac: 64.36<br>re: 35.54
run3 | 1e-4 | 8  | True | ac: 27.24<br>re: 23.87 | ac: 38.28<br>re: 20.72 | ac: 47.46<br>re: 18.07 | ac: 56.70<br>re: 16.21 | ac: 65.28<br>re: 14.25
run4 | 5e-4 | 10 | True | ac: 39.75<br>re: 24.89 | ac: 48.39<br>re: 22.86 | ac: 55.60<br>re: 21.33 | ac: 61.63<br>re: 20.12 | ac: 66.44<br>re: 19.02

1. Notice:
* **Dataset: TT100K**
* No.: Number of run. Run1 has been ruled out since loss became nan.
* lr: Learning rate.
* ep: Max epoch.
* box limit: (in line #190, lib/roi_data_layer/roibatchLoader.py) Whether to limit the size of box less than 15.
* 0.1-0.5: (in anno_func.py) Value of `minscore` as a parameter of function `eval_annos`.
* ac, re: Accuracy and Recall, respectively.

2. Other default parameters or runtime situations:
* `ANCHOR_SCALES` and `'MAX_NUM_GT_BOXES` are all `[16, 32, 64, 128, 256]` and 50.
* `atch_size` is 2, `net` is `res101`, and `lr_decay_step` is 6.
* Image size is 2048 * 2048.
* Memory occupied is between 3000MB and 4000MB.
* Detection time is around 0.056s, and NMS time is around 0.087s.
