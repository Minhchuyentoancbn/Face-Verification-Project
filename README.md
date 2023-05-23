# Building a Life-long Face Verification System

![Face Recognition illustration](data/face_verification.jpg)

## Project Description

__Goal__: Build a life-long face verification system that can learn new faces without forgetting the old ones and generalize well on new faces.

__Base architecture__: Inception Resnet V1

__Dataset__: LFW, CASIA-Webface

__Preprocessing__: MTCNN face detection and alignment

__Evaluation__: Accuracy, VAL, FAR on LFW dataset using the unrestricted with labeled outside data protocol

__Methods__: Softmax loss, Label smoothing, Triplet loss, Center loss + Distillation loss, Neighborhood Selection, Consistency Relaxation for continual learning


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
usage: main.py [-h] [--seed SEED] [--preprocess PREPROCESS] [--num_tasks NUM_TASKS] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--optimizer OPTIMIZER]
               [--lr LR] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT] [--smooth SMOOTH] [--triplet TRIPLET] [--margin MARGIN]
               [--alpha ALPHA] [--center CENTER] [--beta BETA] [--center_lr CENTER_LR] [--finetune FINETUNE] [--distill DISTILL] [--ns NS] [--cr CR]
               [--lambda_old LAMBDA_OLD] [--T T] [--K K] [--beta0 BETA0] [--clip CLIP] [--clip_value CLIP_VALUE] [--eval_cycle EVAL_CYCLE]
               [--step_size STEP_SIZE] [--exp_name EXP_NAME]

options:
  -h, --help            show this help message and exit
  --seed SEED           Random seed
  --preprocess PREPROCESS
                        Preprocess CASIA-Webface dataset
  --num_tasks NUM_TASKS
                        Number of tasks to split the dataset
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Number of epochs
  --optimizer OPTIMIZER
                        Optimizer types: {sgd, adam}
  --margin MARGIN       Margin for triplet loss
  --alpha ALPHA         Alpha for triplet loss
  --center CENTER       Use center loss
  --beta BETA           Beta for center loss
  --center_lr CENTER_LR
                        Learning rate for center loss
  --finetune FINETUNE   Finetune the model
  --distill DISTILL     Use distillation loss
  --ns NS               Use Neighborhood Selection
  --cr CR               Use Consistency Relaxation
  --lambda_old LAMBDA_OLD
                        Lambda for old loss
  --T T                 Temperature for new loss
  --K K                 Number of selected neighbors
  --beta0 BETA0         Beta0, margin for Consistency Relaxation
  --clip CLIP           Whether to clip gradients
  --clip_value CLIP_VALUE
                        Value to clip gradients
  --eval_cycle EVAL_CYCLE
                        Evaluate every n epochs
  --step_size STEP_SIZE
                        Step size for LR scheduler
  --exp_name EXP_NAME   Experiment name
```


__Baseline__

```
python main.py --num_tasks 1 --batch_size 128 --epochs 3 --lr 0.1 --momentum 0.9 --weight_decay 3e-4 --eval_cycle 1
```


 __Baseline + Label smoothing__

```
python main.py --num_tasks 1 --batch_size 128 --epochs 3 --lr 0.1 --momentum 0.9 --weight_decay 3e-4 --eval_cycle 1 --smooth 1e-3
```


 __Baseline + Triplet Loss + Center Loss__

```
python main.py --num_tasks 1 --batch_size 128 --epochs 24 --lr 0.1 --momentum 0.9 --weight_decay 3e-4 --eval_cycle 1 --center True --beta 1e-3 --triplet True
```


 __Baseline + Label smoothing + Triplet Loss + Center Loss__

```
python main.py --num_tasks 1 --batch_size 128 --epochs 24 --lr 0.1 --momentum 0.9 --weight_decay 3e-4 --eval_cycle 1 --smooth 1e-3 --center True --beta 1e-3 --triplet True
```



## Experimental Results

### Training 1 task - LFW

| Method | Accuracy | VAL | FAR |
| --- | --- | --- | --- |
| Baseline | 0.96816 | 0.75500 | 0.00133 |
| Baseline + Label smoothing | 0.97316 | 0.82600 | 0.00133 |
| Baseline + Triplet Loss + Center Loss | 0.97483 | 0.77700 | 0.00100 |
| Baseline + Label smoothing + Triplet Loss + Center Loss | 0.97349 | 0.85300 | 0.00133 |

![LFW - Training 1 task](results/1task.png)

### Traning 5 tasks


## Run Server
```
python run_server.py
```


## References

__Papers:__

K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks,” IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, Aug. 2016, doi: 10.1109/lsp.2016.2603342.

F. Schroff, D. Kalenichenko, and J. Philbin, FaceNet: A unified embedding for face recognition and clustering. 2015. doi: 10.1109/cvpr.2015.7298682.

B. Zhao, S. Tang, D. Chen, H. Bilen, and R. Zhao, Continual Representation Learning for Biometric Identification. 2021. doi: 10.1109/wacv48630.2021.00124.

G. Guo and N. Zhang, “A survey on deep learning based face recognition,” Computer Vision and Image Understanding, vol. 189, p. 102805, Dec. 2019, doi: 10.1016/j.cviu.2019.102805.

G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007.  



__Github repositories:__

- [faceNet-pytorch](https://github.com/timesler/facenet-pytorch)