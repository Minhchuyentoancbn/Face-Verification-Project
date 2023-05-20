import argparse
import sys
import os
import numpy as np

from preprocess import preprocess_data
from addition_loss import CenterLoss
from evaluate import evaluate_lfw
from config import *
from utils import get_device, seed_everything, weights_init, accuracy, BatchTimer, pass_epoch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks to split the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer types: {sgd, adam}')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability for last fully connected layer')
    # parser.add_argument('--min_lr', type=float, default=0.0, help='Minimum learning rate for LearningRateRangeTest')
    # parser.add_argument('--max_lr', type=float, default=0.0, help='Maximum learning rate for LearningRateRangeTest')
    parser.add_argument('--smooth', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--triplet', type=bool, default=False, help='Use triplet loss')
    parser.add_argument('--margin', type=float, default=0.3, help='Margin for triplet loss')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha for triplet loss')
    parser.add_argument('--center', type=bool, default=False, help='Use center loss')
    parser.add_argument('--beta', type=float, default= 0.0005, help='Beta for center loss')
    parser.add_argument('--center_lr', type=float, default=0.5, help='Learning rate for center loss')

    parser.add_argument('--eval_cycle', type=int, default=20, help='Evaluate every n epochs')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for LR scheduler')
    parser.add_argument('--clip', type=bool, default=False, help='Whether to clip gradients')
    parser.add_argument('--clip_value', type=float, default=0.0, help='Value to clip gradients')
    args = parser.parse_args(argv)
    return args



def main(args):
    # Set hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr_init = args.lr
    label_smoothing = args.smooth
    dropout_prob = args.dropout
    device = get_device()

    # Define model
    num_classes = len(os.listdir(casia_cropped_path))
    resnet = InceptionResnetV1(classify=True, num_classes=num_classes, dropout_prob=dropout_prob)
    resnet = weights_init(resnet).to(device)

    # Define classes and tasks
    num_tasks = args.num_tasks
    num_classes_per_task = num_classes // num_tasks
    if os.path.exists('./data/classes.npy'):
        classes = np.load('./data/classes.npy')
    else:
        classes = np.arange(num_classes)
        np.random.shuffle(classes)
        np.save('./data/classes.npy', classes)

    # Define loss function and evaluation metrics
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    metrics = {
        'accuracy': accuracy,
        'fps': BatchTimer()
    }

    if args.center:
        center_loss_fn = CenterLoss(num_classes, num_classes_per_task).to(device)
        optimizer_center = optim.SGD(center_loss_fn.parameters(), lr=args.center_lr)
    else:
        center_loss_fn = None
        optimizer_center = None

    #######################################
    # Define dataset, and dataloader
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(casia_cropped_path, transform=trans)
    img_inds = np.arange(len(dataset))
    targets = np.array(dataset.targets)

    train_loaders = dict()
    val_loaders = dict()

    # Train and validation loaders for each task
    for task in range(num_tasks):
        # Get classes for this task
        task_classes = classes[task * num_classes_per_task: (task + 1) * num_classes_per_task]
        train_inds = []
        val_inds = []

        for i in task_classes:
            inds = img_inds[targets == i]
            train_inds.extend(inds)
            if len(inds) <= 3:
                val_inds.extend(inds)
            else:
                val_inds.extend(inds[-3:])

        train_inds = np.array(train_inds)
        val_inds = np.array(val_inds)
        np.random.shuffle(train_inds)
        np.random.shuffle(val_inds)

        train_set = Subset(dataset, train_inds)
        val_set = Subset(dataset, val_inds)

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False
        )
        train_loaders[task] = train_loader
        val_loaders[task] = val_loader

    # Save dataset
    for task in range(num_tasks):
        if num_tasks > 1:
            torch.save(train_loaders[task], f'./data/train_loader_{task}.pth')
            torch.save(val_loaders[task], f'./data/val_loader_{task}.pth')
        else:
            torch.save(train_loaders[task], f'./data/train_loader.pth')
            torch.save(val_loaders[task], f'./data/val_loader.pth')
    #######################################

    # Train
    for task in range(num_tasks):
        train_loader = train_loaders[task]
        val_loader = val_loaders[task]
        # if num_tasks > 1:
        #     train_loader = torch.load(f'./data/train_loader_{task}.pth')
        #     val_loader = torch.load(f'./data/val_loader_{task}.pth')
        # else:
        #     train_loader = torch.load(f'./data/train_loader.pth')
        #     val_loader = torch.load(f'./data/val_loader.pth')
        print('=' * 10)
        print(f'Task {task} starts')

        # Define optimizer, scheduler
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(resnet.parameters(), lr=lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(resnet.parameters(), lr=lr_init, weight_decay=args.weight_decay, eps=0.1)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_update_rule)

        writer = SummaryWriter(LOG_DIR + '1task', comment=f'task{task}_{args.optimizer}_lr{lr_init}_bs{batch_size}_epochs{epochs}_momentum{args.momentum}_weight_decay{args.weight_decay}')
        writer.iteration = 0
        print('Initial')
        print('=' * 10)

        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('=' * 10)

            # Train
            pass_epoch(resnet, loss_fn, train_loader, val_loader, optimizer,
                       batch_metrics=metrics, device=device, writer=writer, args=args,
                       center_loss_fn=center_loss_fn, optimizer_center=optimizer_center
                       )

            if (epoch + 1) % args.step_size == 0:
                scheduler.step()

            # Evaluate on LFW
            if (epoch + 1) % args.eval_cycle == 0:
                print('Validate on LFW')
                lfw_accuracy = evaluate_lfw(resnet)

        writer.close()
        print(f'Task {task + 1} / {num_tasks} finished.')
        print('=' * 20)

        # Evaluate on LFW
        if epochs % args.eval_cycle != 0:
            print('Validate on LFW')
            lfw_accuracy = evaluate_lfw(resnet)

        # Save model
        if num_tasks > 1:
            torch.save(resnet.state_dict(), f'./trained_models/task{task + 1}_resnet.pth')
        else:
            torch.save(resnet.state_dict(), f'./trained_models/resnet.pth')

        break



if __name__ == '__main__':
    seed_everything(SEED)
    args = parse_arguments(sys.argv[1:])

    # # Create data folders
    # if not os.path.exists(casia_cropped_path):
    #     os.makedirs(casia_cropped_path)
    #     preprocess_data(args)

    # Train the model
    main(args)

    torch.cuda.empty_cache()