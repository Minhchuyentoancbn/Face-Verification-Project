import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import hflip

import numpy as np
from addition_loss import triplet_loss
from utils import Logger, BatchTimer


def calculate_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    loss_fn: callable,
    center_loss_fn: nn.Module=None,
    model_old: nn.Module=None,
    current_classes: np.ndarray=None,
    old_classes: np.ndarray=None,
    args=None,
):
    """
    Calculates the loss of a batch

    Parameters
    ----------
    x : torch.Tensor
        Input tensor

    y : torch.Tensor
        Labels

    model : nn.Module
        PyTorch model

    loss_fn : callable
        Loss function

    center_loss_fn : nn.Module, optional
        Center loss function, by default None

    model_old : nn.Module, optional
        Old model, by default None

    current_classes : np.ndarray, optional
        Current classes, by default None

    old_classes : np.ndarray, optional
        Old classes, by default None

    args : argparse.ArgumentParser, optional
        Command line arguments, by default None

    Returns
    -------
    loss_batch: torch.Tensor
        Loss

    distill_loss: torch.Tensor
        Distillation loss

    y_pred_origin: torch.Tensor
        Predictions
    """
    if not model.training:
        with torch.no_grad():
            y_pred, linear = model(x)
    else:
        y_pred, linear = model(x)
    
    y_pred_origin = y_pred.clone()

    if args.num_tasks == 1:
        loss_batch = loss_fn(y_pred, y)
    else:
        loss_batch = F.log_softmax(y_pred[:, current_classes], dim=1)[torch.arange(y.size(0)), y].mean()

    # Triplet loss
    if args.triplet:
        loss_batch += triplet_loss(linear, y, margin=args.margin) * args.alpha

    # Center loss
    if args.center:
        if not model.training:
            with torch.no_grad():
                loss_batch += args.beta * center_loss_fn(linear, y)
        else:
            loss_batch += args.beta * center_loss_fn(linear, y)

    distill_loss = None

    # Distillation loss
    if args.distill and len(old_classes) > 0:

        # Old model prediction
        with torch.no_grad():
            y_pred_old, _ = model_old(x)

        y_pred_old = y_pred_old[:, old_classes]
        y_pred = y_pred[:, old_classes]

        # Neighborhood selection
        if args.ns and args.K > 0:
            # Select top args.K classes for each sample
            y_pred_old, selected_indices = y_pred_old.topk(args.K, dim=1)
            y_pred = y_pred.gather(1, selected_indices)

        # Distillation loss
        distill_loss = (
            F.log_softmax(y_pred_old / args.T, dim=1) * F.softmax(y_pred_old / args.T, dim=1)
            - F.log_softmax(y_pred / args.T, dim=1) * F.softmax(y_pred_old / args.T, dim=1)
        ).sum(dim=1)

        # Consistency relaxation
        if args.cr:
            margin = -args.beta0 * (F.softmax(y_pred_old / args.T, dim=1) * F.log_softmax(y_pred_old / args.T, dim=1)).sum(dim=1)
            distill_loss -= margin

        distill_loss = args.lambda_old * F.relu(distill_loss).mean()


    return loss_batch, distill_loss, y_pred_origin 


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
    optimizer_center: torch.optim.Optimizer=None,
    model_old: nn.Module=None,
    current_classes: np.ndarray=None,
    old_classes: np.ndarray=None
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

    model_old: nn.Module
        Old model. (default: {None})

    current_classes: np.ndarray
        Current classes. (default: {None})

    old_classes: np.ndarray
        Old classes. (default: {None})
    
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

        loss_batch, distill_loss, y_pred = calculate_loss(x, y, model, loss_fn, center_loss_fn, model_old=model_old, current_classes=current_classes, old_classes=old_classes, args=args)

        if args.center:
            optimizer_center.zero_grad()

        # Backward pass
        optimizer.zero_grad()
        loss_batch.backward()

        optimizer.step()
        if args.center:
            for param in center_loss_fn.parameters():
                param.grad.data *= (1. / args.beta)
            optimizer_center.step()

        if distill_loss is not None:
            optimizer.zero_grad()
            distill_loss.backward()
            # Zero grad for model.logits layer, which is a Linear layer
            for param in model.logits.parameters():
                param.grad.data *= 0.0
            optimizer.step()

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
    validate(model, loss_fn, valid_loader, batch_metrics, device, writer, optimizer, args, center_loss_fn, model_old, old_classes)

    # Log to tensorboard
    if writer is not None:
        writer.add_scalars('loss', {mode: loss / (i_batch + 1)}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric / (i_batch + 1)})
        if optimizer is not None:
            writer.add_scalars('lr', {mode: optimizer.param_groups[0]['lr']}, writer.iteration)

    # Free intermediate variables
    del x, y, y_pred, loss_batch, metrics_batch
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
    center_loss_fn:nn.Module=None,
    model_old: nn.Module=None,
    current_classes: np.ndarray=None,
    old_classes: np.ndarray=None
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

    model_old: nn.Module
        Old model. (default: {None})

    current_classes: np.ndarray
        Current classes. (default: {None})

    old_classes: np.ndarray
        Old classes. (default: {None})

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

        loss_batch, distill_loss, y_pred = calculate_loss(x, y, model, loss_fn, center_loss_fn, model_old=model_old, current_classes=current_classes, old_classes=old_classes, args=args)

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
    del x, y, y_pred, loss_batch, metrics_batch

    return loss, metrics