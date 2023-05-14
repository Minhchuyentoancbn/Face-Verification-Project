import argparse
import sys
import os
import numpy as np

from models import get_mtcnn
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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer types: sgd, adam')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    # parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')


    # parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to dataset')
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

    # Define optimizer, scheduler
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(resnet.parameters(), lr=lr_init, momentum=0.9, weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(resnet.parameters(), lr=lr_init)
    scheduler = MultiStepLR(optimizer, [5, 10])

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
    # and validation set separately
    train_inds = []
    val_inds = []
    for i in range(num_classes):
        inds = img_inds[targets==i]
        if len(inds) == 1:
            train_inds.append(inds[0])
        elif len(inds) == 2:
            train_inds.extend(inds[:-1])
            val_inds.append(inds[-1])
        else:
            train_inds.extend(inds[:-2])
            val_inds.extend(inds[-2:])

    train_inds = np.array(train_inds)
    val_inds = np.array(val_inds)
    # Shuffle the indices
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

    # Define loss function and evaluation metrics
    loss_fn = nn.CrossEntropyLoss()
    metrics = {
        'accuracy': accuracy,
        'fps': BatchTimer()
    }

    # Train the model
    writer = SummaryWriter(LOG_DIR + 'exp1', comment=f'{args.optimizer}_lr{lr_init}_bs{batch_size}_epochs{epochs}_momentum0.9_wd0.0005')
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('=' * 10)
    resnet.eval()
    pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('=' * 10)

        resnet.train()
        pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()



if __name__ == '__main__':
    seed_everything(SEED)
    args = parse_arguments(sys.argv[1:])

    # Create data folders
    if not os.path.exists(casia_cropped_path):
        os.makedirs(casia_cropped_path)
        preprocess_data(args)

    # Train the model
    train(args)