import os
import time
import numpy as np
from addition_loss import triplet_loss

import torch
import torch.nn as nn
from torchvision.transforms.functional import hflip


def empty_folder(folder_path: str) -> None:
    """
    Empties the folder

    Parameters
    ----------
    folder_path : str
        Path to the folder to be emptied
    """
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))
    
    return


def get_device() -> torch.device:
    """
    Returns the device

    Returns
    -------
    torch.device
        Device
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    return device


def seed_everything(seed=42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def collate_pil(x):
    out_x, out_y = [], [] 
    for xx, yy in x: 
        out_x.append(xx) 
        out_y.append(yy) 
    return out_x, out_y


def weights_init(models: nn.Module):
    # Initialize weights
    for m in models.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)

    return models


class Logger(object):
    """Logging object for training and validation."""

    def __init__(self, mode, length, calculate_mean=False):
        """
        Parameters
        ----------
        mode : str
            'Train' or 'Valid'

        length : int
            Number of batches per epoch

        calculate_mean : bool, optional
            Whether to calculate the mean of the logged values, by default False
        """
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x


    def __call__(self, loss, metrics, i):
        """
        Logs the loss and metrics

        Parameters
        ----------
        loss : torch.Tensor
            Loss value

        metrics : dict
            Dictionary of metric values

        i : int
            Batch index
        """
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')


class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.
    """
    def __init__(self, rate=True, per_sample=True):
        """
        Parameters
        ----------
        rate : bool, optional
            Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample), by default True

        per_sample : bool, optional
            Whether to report times or rates per sample or per batch, by default True
        """
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample


    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


def accuracy(logits, y):
    """
    Computes the accuracy for multiple binary predictions

    Parameters
    ----------
    logits : torch.Tensor
        Logits

    y : torch.Tensor
        Labels

    Returns
    -------
    torch.Tensor
        Accuracy
    """
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def pass_epoch(
    model: nn.Module, 
    loss_fn: callable, 
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer=None, 
    batch_metrics: dict={'time': BatchTimer()},
    device='gpu', 
    writer=None,
    args=None,
    center_loss_fn:nn.Module=None, 
    optimizer_center: torch.optim.Optimizer=None
):
    """
    rain over a data epoch.
    
    Parameters:
    ----------
    model: torch.nn.Module
        Pytorch model.

    loss_fn: callable
        A function to compute (scalar) loss.

    train_loader: torch.utils.data.DataLoader
        DataLoader for training data.

    valid_loader: torch.utils.data.DataLoader
        DataLoader for validation data.
    
    optimizer: torch.optim.Optimizer
        A pytorch optimizer.

    batch_metrics: dict 
        Dictionary of metric functions to call on each batch. The default
        is a simple timer. A progressive average of these metrics, along with the average
        loss, is printed every batch. (default: {{'time': iter_timer()}})

    device: str or torch.device
        Device for pytorch to use. (default: {'cpu'})
        
    writer: torch.utils.tensorboard.SummaryWriter 
        Tensorboard SummaryWriter. (default: {None})
    
    args: argparse.ArgumentParser
        Command line arguments. (default: {None})
    
    center_loss_fn: nn.Module
        Center loss function. (default: {None})

    optimizer_center: torch.optim.Optimizer
        Optimizer for center loss. (default: {None})
    
    Returns:
    -------
    tuple(torch.Tensor, dict) 
        A tuple of the average loss and a dictionary of average metric values across the epoch.
    """
    
    # Set model to train or eval mode
    mode = 'Train' if model.training else 'Valid'
    if mode == 'Valid':
        model.train()
        mode = 'Train'
        if args.center:
            center_loss_fn.train()

    logger = Logger(mode, length=len(train_loader), calculate_mean=True)
    loss = 0
    metrics = {}

    for i_batch, (x, y) in enumerate(train_loader):
        # Forward pass
        x = x.to(device)
        y = y.to(device)
        model.train()
        y_pred, linear = model(x)
        loss_batch = loss_fn(y_pred, y)

        # Triplet loss
        if args.triplet:
            loss_batch += triplet_loss(linear, y, margin=args.margin) * args.alpha

        # Center loss
        if args.center:
            loss_batch += args.beta * center_loss_fn(linear, y)
            optimizer_center.zero_grad()

        # Backward pass
        optimizer.zero_grad()
        loss_batch.backward()
        # Clip gradients
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
        optimizer.step()
        if args.center:
            for param in center_loss_fn.parameters():
                param.grad.data *= (1. / args.beta)
            optimizer_center.step()

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch.item()

        # Update metrics and print values
        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(y_pred, y).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
        logger(loss, metrics, i_batch)

    # Validation per epoch
    print(f'Lr: {optimizer.param_groups[0]["lr"]:0.6f}')
    validate(model, loss_fn, valid_loader, batch_metrics, device, writer, optimizer, args, center_loss_fn)

    # Log to tensorboard
    if writer is not None:
        writer.add_scalars('loss', {mode: loss / (i_batch + 1)}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric / (i_batch + 1)})
        if optimizer is not None:
            writer.add_scalars('lr', {mode: optimizer.param_groups[0]['lr']}, writer.iteration)

    # Free intermediate variables
    del x, y, y_pred, loss_batch, metrics_batch, linear
    return loss, metrics


def validate(
    model: nn.Module, 
    loss_fn: callable, 
    loader: torch.utils.data.DataLoader, 
    batch_metrics: dict={'time': BatchTimer()}, 
    device='gpu', 
    writer=None,
    optimizer=None,
    args=None,
    center_loss_fn:nn.Module=None
):
    """
    Evaluate over a data loader
    
    Parameters:
    ----------
    model: torch.nn.Module
        Pytorch model.

    loss_fn: callable
        A function to compute (scalar) loss.

    loader: torch.utils.data.DataLoader
        A pytorch data loader.

    batch_metrics: dict 
        Dictionary of metric functions to call on each batch. The default
        is a simple timer. A progressive average of these metrics, along with the average
        loss, is printed every batch. (default: {{'time': iter_timer()}})

    device: str or torch.device
        Device for pytorch to use. (default: {'cpu'})
        
    writer: torch.utils.tensorboard.SummaryWriter 
        Tensorboard SummaryWriter. (default: {None})

    optimizer: torch.optim.Optimizer
        Pytorch optimizer. (default: {None})
    
    args: argparse.ArgumentParser
        Command line arguments. (default: {None})

    center_loss_fn: nn.Module
        Center loss function. (default: {None})

    Returns:
    -------
    tuple(torch.Tensor, dict) 
        A tuple of the average loss and a dictionary of average metric values across the epoch.
    """
    
    mode = 'Train' if model.training else 'Valid'
    if mode == 'Train':
        model.eval()
        if args.center:
            center_loss_fn.eval()
        mode = 'Valid'

    loss = 0
    metrics = {}
    logger = Logger(mode, length=len(loader), calculate_mean=True)

    for i_batch, (x, y) in enumerate(loader):
        # Flip images horizontally
        x = hflip(x).to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred, linear = model(x)

        loss_batch = loss_fn(y_pred, y)

        # Triplet loss
        if args.triplet:
            loss_batch += triplet_loss(linear, y, margin=args.margin)

        # Center loss
        if args.center:
            with torch.no_grad():
                loss_batch += args.beta * center_loss_fn(linear, y)

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch.item()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(y_pred, y).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

    logger(loss, metrics, i_batch)

    loss = loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}
            
    if writer is not None:
        writer.add_scalars('loss', {mode: loss}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})
        if optimizer is not None:
            writer.add_scalars('lr', {mode: optimizer.param_groups[0]['lr']}, writer.iteration)
        writer.iteration += 1

    # Free intermediate variables
    del x, y, y_pred, loss_batch, metrics_batch, linear

    return loss, metrics