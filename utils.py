import os
import time
import numpy as np

import torch
import torch.nn as nn


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
            nn.init.constant_(m.bias, 0)

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