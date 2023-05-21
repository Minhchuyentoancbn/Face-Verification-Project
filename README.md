# Building a Life-long Face Verification System

![Face Recognition illustration](data/face_verification.jpg)

## Project Description

__Goal__: Build a life-long face verification system that can learn new faces without forgetting the old ones and generalize well on new faces.

__Base architecture__: Inception Resnet V1

__Dataset__: LFW, CASIA-Webface

__Preprocessing__: MTCNN face detection and alignment [1]

 __Evaluation__: Accuracy, VAL, FAR on LFW dataset using the unrestricted with labeled outside data protocol

 __Methods__: Softmax, Label smoothing, Triplet loss, Center loss, Adversarial training


## Setup environment

```
pip install -r requirements.txt
```

## Dataset Download
- [__LFW__](http://vis-www.cs.umass.edu/lfw/)
- __CASIA-Webface__

## Training

__Arguments:__

```

```


__Baseline__

```
python main.py --num_tasks 1 --batch_size 128 --epochs 3 --lr 0.1 --momentum 0.9 --weight_decay 3e-4 --eval_cycle 1
```


 __Baseline + Label smoothing__

```
python main.py --num_tasks 1 --batch_size 128 --epochs 3 --lr 0.1 --momentum 0.9 --weight_decay 3e-4 --eval_cycle 1 --smooth 1e-3
```


 __Baseline + Label smoothing + Triplet Loss + Center Loss__

```
python main.py --num_tasks 1 --batch_size 128 --epochs 24 --lr 0.1 --momentum 0.9 --weight_decay 3e-4 --eval_cycle 1 --smooth 1e-3 --center True --beta 1e-3 --triplet True
```



## Results on LFW

| Method | Accuracy | VAL | FAR |
| :---: | :---: | :---: | :---: |
| Baseline | 0.96816 | 0.755 | 0.00133 |
| Baseline + Label smoothing | 0.97349 | 0.82733 | 0.00133 |


## Run Server
```
python run_server.py
```


## References

__Papers:__

_[1] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks,” IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, Aug. 2016, doi: 10.1109/lsp.2016.2603342._

_[2] F. Schroff, D. Kalenichenko, and J. Philbin, FaceNet: A unified embedding for face recognition and clustering. 2015. doi: 10.1109/cvpr.2015.7298682._

_[3] B. Zhao, S. Tang, D. Chen, H. Bilen, and R. Zhao, Continual Representation Learning for Biometric Identification. 2021. doi: 10.1109/wacv48630.2021.00124._

_[4] G. Guo and N. Zhang, “A survey on deep learning based face recognition,” Computer Vision and Image Understanding, vol. 189, p. 102805, Dec. 2019, doi: 10.1016/j.cviu.2019.102805._

_[5] I. Goodfellow, J. Shlens, and C. Szegedy, Explaining and Harnessing Adversarial Examples. 2015. [Online]. Available: https://ai.google/research/pubs/pub43405_

_[6] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007._  



__Github repositories:__

[faceNet-pytorch](https://github.com/timesler/facenet-pytorch)