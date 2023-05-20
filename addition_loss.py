import torch
from torch import nn

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    
    Parameters
    ----------
    x: torch.Tensor
        PyTorch Tensor to be normalized.

    axis: int
        dimension to normalize. Default is -1, which corresponds to the last dimension.

    Returns
    -------
    x: torch.Tensor
        Normalized PyTorch Tensor.
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Parameters
    ----------
    x: torch.Tensor
        PyTorch Tensor, with shape [m, d]
    
    y: torch.Tensor
        PyTorch Tensor, with shape [n, d]
    Returns
    -------
    dist: torch.Tensor
        PyTorch Tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12)  # for numerical stability
    return dist



def triplet_loss(embeddings, labels, margin):
    """
    Compute triplet loss.
    
    Parameters
    ----------
    embeddings: torch.Tensor
        PyTorch Tensor, with shape [batch_size, embedding_size]

    labels: torch.LongTensor
        PyTorch LongTensor, with shape [batch_size]

    margin: float
        margin for triplet loss

    Returns
    -------
    loss: torch.Tensor
        PyTorch Tensor with shape [1]
    """
    
    # Get the pairwise distance matrix
    pairwise_dist = euclidean_dist(embeddings, embeddings)
    
    # For each anchor, find the hardest positive and negative
    # (Note that the pair (i, i) is excluded from the loss)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
    
    # shape [batch_size, 1]
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    # shape [batch_size, 1]
    hardest_positive_dist = anchor_positive_dist.max(1, keepdim=True)[0]
    # shape [batch_size]
    anchor_negative_dist = mask_anchor_negative * pairwise_dist
    # shape [batch_size]
    hardest_negative_dist = anchor_negative_dist.min(1, keepdim=True)[0]

    # Filter out samples that have no positive pairs
    having_positive = mask_anchor_positive.sum(1) > 0
    hardest_positive_dist = hardest_positive_dist[having_positive]
    hardest_negative_dist = hardest_negative_dist[having_positive]
    
    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = (hardest_positive_dist - hardest_negative_dist + margin).clamp(min=0)
    
    # Get final mean triplet loss
    triplet_loss = triplet_loss.mean()
    
    return triplet_loss


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    
    Parameters
    ----------
    labels: torch.LongTensor
        PyTorch LongTensor with shape [batch_size]
    
    Returns
    -------
    mask: torch.BoolTensor
        PyTorch BoolTensor with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
    
    # Check if labels[i] == labels[j]
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # Combine the two masks
    mask = indices_equal & labels_equal
    
    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    
    Parameters
    ----------
    labels: torch.LongTensor
        PyTorch LongTensor with shape [batch_size]
    
    Returns
    -------
    mask: torch.BoolTensor
        PyTorch BoolTensor with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = ~labels_equal
    
    return mask


class CenterLoss(nn.Module):
    """
    Center Loss
    """

    def __init__(self, num_classes:int=10544, feat_dim:int=512):
        """
        Parameters
        ----------
        num_classes: int
            number of classes

        feat_dim: int
            feature dimension
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))


    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_classes).long().to(labels.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss