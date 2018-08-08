## Running Summary:
* Dataset: PASCAL VOC 2007
* Model: ResNet 101
* Number of GPUs: 1
* Default Pooling Mode: RoI-Crop

No. | bs | lr | step | ep | other | mem | time | mAP
----|----|----|----|----|----|----|----|----
run1 | 4 | 4e-3 | 8 | 10 |      |      | 23 | 75.28
run2 | 1 | 1e-3 | 5 | 7  |      | 3440 | 28 | 74.71
run3 | 4 | 4e-3 | 8 | 10 | adam | 9618 | 21 | 00.47
run4 | 4 | 4e-3 | 8 | 10 | no flip | 10546 | 11 | 00.00
run5 | 1 | 5e-4 | 4 | 10 |     | 3340  | 26 | 73.90
run6 | 4 | 4e-3 | 8 | 10 | align | 9390 | 24 | 75.43
run7 | 4 | 4e-3 | 8 | 10 | pool | 9700 | 22 | 74.85

1. No.: Number of run
2. bs: Batch size
3. lr: Learning rate
4. step: Decay step
5. ep: Max epochs
6. other: Other changes
7. mem: Memory(MB) / GPU
8. time: Minutes / epoch
9. mAP: %
