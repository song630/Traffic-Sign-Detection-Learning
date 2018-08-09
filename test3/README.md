### Classification & Localization on Birds Images
---
* Dataset: CUB-200-2011 (with 200 classes and 11,788 images)
* Architecture: Add two branches after `resnet18`
* Firstly implement with FC layers at both ends, then replace them with conv layers
* Adjust the way in which images are pre-processed
