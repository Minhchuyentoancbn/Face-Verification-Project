import argparse
import sys
import os
import numpy as np

from models import get_mtcnn
from evaluate import evaluate_lfw
from config import *
from utils import get_device, seed_everything, collate_pil, weights_init, accuracy, BatchTimer, pass_epoch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, OneCycleLR
from torchvision import models, datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization


device = get_device()
casia_cropped_path = os.path.join(DATA_PATH, 'CASIA-WebFace-cropped/')
# casia_cropped_path = '/kaggle/input/casia-webface-cropped-with-mtcnn/CASIA-WebFace-cropped'


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer types: {sgd, adam}')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks to split the dataset')
    # parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    # parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    args = parser.parse_args(argv)
    return args


def preprocess_data(args):
    batch_size = args.batch_size
    # Get MTCNN for face detection
    mtcnn = get_mtcnn()
    data = datasets.ImageFolder(CASIA_PATH, transform=transforms.Resize((512, 512)))
    data.samples = [
        (p, p.replace(CASIA_PATH, casia_cropped_path))
        for p, _ in data.samples
    ]

    loader = DataLoader(
        data,
        batch_size=batch_size,
        collate_fn=collate_pil,
    )

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print(f'\rBatch {i+1}/{len(loader)}', end='')

    print('\nFinish Preprocessing')
    print('='*20)

    del mtcnn


def train(args):
    # Define hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr_init = args.lr
    device = get_device()

    # Define model
    num_classes = len(os.listdir(casia_cropped_path))
    resnet = InceptionResnetV1(classify=True, num_classes=num_classes)
    resnet = weights_init(resnet).to(device)

    # Define dataset, and dataloader
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(casia_cropped_path, transform=trans)
    img_inds = np.arange(len(dataset))
    targets = np.array(dataset.targets)
    
    # Some classes have only one sample, so we need to put them into training set
    # and validation set separately. We also split to 5 tasks, each task has
    # disjoint classes.
    num_tasks = args.num_tasks
    num_classes_per_task = num_classes // num_tasks
    classes = np.arange(num_classes)
    np.random.shuffle(classes)

    train_loaders = dict()
    val_loaders = dict()

    for task in range(num_tasks):
        # Get classes for this task
        task_classes = classes[task * num_classes_per_task: (task + 1) * num_classes_per_task]
        train_inds = []
        val_inds = []

        for i in task_classes:
            inds = img_inds[targets == i]
            if len(inds) == 1:
                train_inds.append(inds[0])
            elif len(inds) >= 4:
                train_inds.extend(inds[:-2])
                val_inds.append(inds[-2])
            else:
                train_inds.extend(inds[:-1])
                val_inds.extend(inds[-1:])

        train_inds = np.array(train_inds)
        val_inds = np.array(val_inds)
        np.random.shuffle(train_inds)
        np.random.shuffle(val_inds)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_inds)
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(val_inds)
        )
        train_loaders[task] = train_loader
        val_loaders[task] = val_loader


    # Define loss function and evaluation metrics
    loss_fn = nn.CrossEntropyLoss()
    metrics = {
        'accuracy': accuracy,
        'fps': BatchTimer()
    }

    # Train
    for task in range(num_tasks):
        print('=' * 10)
        print(f'Task {task} starts')

        # Define optimizer, scheduler
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(resnet.parameters(), lr=lr_init, momentum=0.9, weight_decay=0.0005)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(resnet.parameters(), lr=lr_init)
        scheduler = MultiStepLR(optimizer, [5, 10])

        writer = SummaryWriter(LOG_DIR + 'exp1', comment=f'task{task}_{args.optimizer}_lr{lr_init}_bs{batch_size}_epochs{epochs}_momentum0.9_wd0.0005')
        writer.iteration, writer.interval = 0, 10

        print('Initial')
        print('=' * 10)
        resnet.eval()
        pass_epoch(
            resnet, loss_fn, val_loaders[task],
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('=' * 10)

            resnet.train()
            pass_epoch(
                resnet, loss_fn, train_loaders[task], optimizer, scheduler,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )

            resnet.eval()
            pass_epoch(
                resnet, loss_fn, val_loaders[task],
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )

        writer.close()

        print('Validate on LFW')
        lfw_accuracy = evaluate_lfw(resnet)

        print(f'Task {task + 1} / {num_tasks} finished.')
        print('=' * 20)

        break

    del resnet


if __name__ == '__main__':
    seed_everything(SEED)
    args = parse_arguments(sys.argv[1:])

    # # Create data folders
    # if not os.path.exists(casia_cropped_path):
    #     os.makedirs(casia_cropped_path)
    #     preprocess_data(args)

    # Train the model
    train(args)

    torch.cuda.empty_cache()