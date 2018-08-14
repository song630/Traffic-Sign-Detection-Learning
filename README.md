### Traffic-Sign-Detection-Learning
---
1. Choose `Faster RCNN` or `Mask RCNN` (rewrite and run)
2. Understand `RoIAlign` & `RoIPooling`
3. Read two papers:

[Co-domain Embedding using Deep Quadruplet Networks for Unseen Traffic Sign Recognition](https://arxiv.org/pdf/1712.01907.pdf)

[Perceptual Generative Adversarial Networks for Small Object Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Perceptual_Generative_Adversarial_CVPR_2017_paper.pdf)

### Under Directory Test1 - Test3:
---
1. Three DL programs to quickly warm up and get familiar with `PyTorch`; more importantly, to gain experience on how to adjust parameters and architecture.
2. Test1: A simple `FCN` implementation on PASCAL-VOC-2007.
3. Test2: A classification network using `CNN` & `ResNet` on CIFAR-10.
4. Test3: A classfication & localization network on CUB-200-2011.

### Under Directory Faster-RCNN-records:
---
1. Do multiple tests on PASCAL-VOC-2007 with `Faster-RCNN` from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).
2. Records all the runtime output.

### Under Directory tt100k-records:
---
1. Do tests based on the same source code.
2. Modify the pretreatment and evaluation parts of the code.
3. Records all the runtime output.

### Under Directory cloned_faster_rcnn_codes:
---
1. Comment on the `Faster-RCNN` source code.

### Under Directory pooling:
---
1. Read and comment on the code of `RoIAlign` & `RoIPooling`.

### About the Notes:
---
1. note1.txt: The progress of object detection, starts from `RCNN`, through `SPP` and `Fast-RCNN`, ends with `Faster-RCNN`.
2. note2.txt: About `RPN`, etc.
3. note3.txt: Notes on [Tutorial](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/) provided by [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).
4. note4.txt: Details on `RoI-Pooling`, `RoI-Align` and `RoI-Warping`.
5. note5.txt: Study PASCAL-VOC-2007 and TT100K datasets.
