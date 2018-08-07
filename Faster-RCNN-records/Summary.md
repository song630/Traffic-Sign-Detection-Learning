## Running Summary:
---
* Dataset: PASCAL VOC 2007
* Model: ResNet 101
* Number of GPUs: 1
* Default Pooling Mode: RoI-Align

No. | bs | lr | step | ep | other | mem | time | mAP
----|----|----|----|----|----|----|----|----
1 | 4 | 4e-3 | 8 | 10 |      |      | 23 | 75.28
2 | 1 | 1e-3 | 5 | 7  |      | 3440 | 28 | 74.71
3 | 4 | 4e-3 | 8 | 10 | adam | 9618 | 21 | 00.47
4 | 4 | 4e-3 | 8 | 10 | no flip | 10546 |   |
5 | 1 | 5e-4 | 4 | 10 |     | 3340  |   |

* No.: number of run
* bs: batch size
* lr: learning rate
* step: decay step
* ep: max epochs
* other: other changes
* mem: memory/GPU
* time: min/epoch
* mAP: %
