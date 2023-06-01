import argparse
import sys
import os
import numpy as np
import copy

from preprocess import preprocess_data
from addition_loss import CenterLoss
from evaluate import evaluate_lfw
from config import *
from utils import get_device, seed_everything, weights_init, accuracy, BatchTimer
from train_utils import pass_epoch, validate

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
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--preprocess', type=bool, default=False, help='Preprocess CASIA-Webface dataset')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks to split the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer types: {sgd, adam}')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability for last fully connected layer')

    parser.add_argument('--smooth', type=float, default=0.0, help='Label smoothing')

    parser.add_argument('--triplet', type=bool, default=False, help='Use triplet loss')
    parser.add_argument('--margin', type=float, default=0.3, help='Margin for triplet loss')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha for triplet loss')

    parser.add_argument('--center', type=bool, default=False, help='Use center loss')
    parser.add_argument('--beta', type=float, default= 0.0005, help='Beta for center loss')
    parser.add_argument('--center_lr', type=float, default=0.5, help='Learning rate for center loss')

    parser.add_argument('--finetune', type=bool, default=True, help='Finetune the model')
    parser.add_argument('--distill', type=bool, default=False, help='Use distillation loss')
    parser.add_argument('--ns', type=bool, default=False, help='Use Neighborhood Selection')
    parser.add_argument('--cr', type=bool, default=False, help='Use Consistency Relaxation')
    parser.add_argument('--lambda_old', type=float, default=1.0, help='Lambda for old loss')
    parser.add_argument('--T', type=float, default=1.0, help='Temperature for new loss')
    parser.add_argument('--K', type=int, default=100, help='Number of selected neighbors')
    parser.add_argument('--beta0', type=float, default=0.1, help='Beta0, margin for Consistency Relaxation')

    parser.add_argument('--clip', type=bool, default=False, help='Whether to clip gradients')
    parser.add_argument('--clip_value', type=float, default=0.0, help='Value to clip gradients')
    parser.add_argument('--eval_cycle', type=int, default=20, help='Evaluate every n epochs')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for LR scheduler')
    parser.add_argument('--exp_name', type=str, default='1task', help='Experiment name')
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
    resnet_old = None

    # Define classes and tasks
    num_tasks = args.num_tasks
    num_classes_per_task = np.ceil(num_classes / num_tasks).astype(int)
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
        center_loss_fn = CenterLoss(num_classes).to(device)
        optimizer_center = optim.SGD(center_loss_fn.parameters(), lr=args.center_lr)
    else:
        center_loss_fn = None
        optimizer_center = None

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

    # Experiment name
    if not os.path.exists(LOG_DIR + args.exp_name):
        os.makedirs(LOG_DIR + args.exp_name)

    # Initialize accuracy matrix
    # tasks_accuracy[i, j] is the accuracy of task j after observing task i
    tasks_accuracy = np.zeros((num_tasks, num_tasks))
    tasks_lfw_accuracy = np.zeros(num_tasks)
    tasks_lfw_val = np.zeros(num_tasks)
    tasks_lfw_far = np.zeros(num_tasks)

    # Train
    for task in range(0, num_tasks):
        old_classes = classes[:task * num_classes_per_task]
        train_loader = train_loaders[task]
        val_loader = val_loaders[task]

        if not args.finetune and task > 0:
            # Recreate model
            resnet = InceptionResnetV1(classify=True, num_classes=num_classes, dropout_prob=dropout_prob)
            resnet = weights_init(resnet).to(device)
        #######################################
        # NOTE: Change number of tasks in the for loop
        # else:
        #     if task > 0:
        #         # Load model
        #         resnet.load_state_dict(torch.load(f'./trained_models/task{task}_resnet.pth'))
        #         resnet_old = InceptionResnetV1(classify=True, num_classes=num_classes, dropout_prob=dropout_prob, device=device)
        #         resnet_old.load_state_dict(torch.load(f'./trained_models/task{task}_resnet.pth'))
        #         resnet_old.eval()
        #######################################
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

        writer = SummaryWriter(LOG_DIR + args.exp_name, comment=f'task{task}_{args.optimizer}_lr{lr_init}_bs{batch_size}_epochs{epochs}_momentum{args.momentum}_weight_decay{args.weight_decay}')
        writer.iteration = 0
        print('Initial')
        print('=' * 10)

        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('=' * 10)

            # Train
            pass_epoch(resnet, loss_fn, train_loader, val_loader, optimizer,
                       batch_metrics=metrics, device=device, writer=writer, args=args,
                       center_loss_fn=center_loss_fn, optimizer_center=optimizer_center,
                       old_classes=old_classes, model_old=resnet_old
                       )

            if (epoch + 1) % args.step_size == 0:
                scheduler.step()

            # Evaluate on LFW
            if (epoch + 1) % args.eval_cycle == 0:
                print('Validate on LFW')
                lfw_accuracy, lfw_val, lfw_far = evaluate_lfw(resnet)

        writer.close()
        print(f'Task {task + 1} / {num_tasks} finished.')
        print('=' * 20)

        # Evaluate on LFW
        if epochs % args.eval_cycle != 0:
            print('Validate on LFW')
            lfw_accuracy, lfw_val, lfw_far = evaluate_lfw(resnet)

        tasks_lfw_accuracy[task] = lfw_accuracy
        tasks_lfw_val[task] = lfw_val
        tasks_lfw_far[task] = lfw_far

        # Save model
        if num_tasks > 1:
            torch.save(resnet.state_dict(), f'./trained_models/task{task + 1}_resnet.pth')
        else:
            torch.save(resnet.state_dict(), f'./trained_models/resnet.pth')

        for tid in range(num_tasks):
            loss, task_metrics = validate(
                resnet, loss_fn, val_loaders[tid], metrics, 
                device=device, args=args, optimizer=optimizer, 
                center_loss_fn=center_loss_fn, model_old=resnet_old, 
                old_classes=old_classes
            )
            tasks_accuracy[task, tid] = task_metrics['accuracy']

        # Save accuracy matrix
        np.save('results/accuracy_matrix.npy', tasks_accuracy)
        np.save('results/lfw_accuracy.npy', tasks_lfw_accuracy)
        np.save('results/lfw_val.npy', tasks_lfw_val)
        np.save('results/lfw_far.npy', tasks_lfw_far)

        if args.finetune:
            if task == 0:
                resnet_old = InceptionResnetV1(classify=True, num_classes=num_classes, dropout_prob=dropout_prob, device=device)
            # Save model for distillation
            resnet_old.load_state_dict(copy.deepcopy(resnet.state_dict()))
            resnet_old.eval()

        #############
        # if task == 0:
        #     break



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    seed_everything(args.seed)

    if args.preprocess:
        # Create data folders
        if not os.path.exists(casia_cropped_path):
            os.makedirs(casia_cropped_path)
        
        preprocess_data(args)

    # Train the model
    main(args)

    torch.cuda.empty_cache()