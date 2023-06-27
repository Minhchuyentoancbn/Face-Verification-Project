import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms
from torchvision.transforms.functional import hflip
from facenet_pytorch import fixed_image_standardization, InceptionResnetV1

import numpy as np
import os
import math
from config import *
from utils import get_device
from sklearn.model_selection import KFold
from scipy import interpolate
from collections import defaultdict


# LFW functions taken from David Sandberg's FaceNet implementation
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far



def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=1, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)



def evaluate_lfw(model: nn.Module, 
             batch_size: int=32
             ):
    """
    Evaluate the model on the LFW dataset

    Parameters
    ----------
    model: nn.Module
        The model to evaluate

    batch_size: int
        The batch size to use for evaluation

    Returns
    -------
    accuracy: float
        The accuracy of the model on the LFW dataset

    val: float
        The validation rate of the model on the LFW dataset

    far: float
        The false acceptance rate of the model on the LFW dataset
    """
    crop_paths = np.load('data/lfw/crop_paths.npy')
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(LFW_PATH, transform=trans)

    embed_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    classify = model.classify
    model.classify = False
    device = get_device()
    resnet = model.to(device)

    classes = []
    embeddings = []
    resnet.eval()
    with torch.no_grad():
        for xb, yb in embed_loader:
            xb = xb.to(device)
            b_embeddings, _ = resnet(xb)
            b_embeddings = b_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)
    
    embeddings_dict = dict(zip(crop_paths, embeddings))
    pairs = read_pairs(LFW_PAIRS_PATH)

    path_list = np.load('data/lfw/path_list.npy')
    issame_list = np.load('data/lfw/issame_list.npy')
    # path_list, issame_list = get_paths(LFW_PATH, pairs)
    embeddings = np.array([embeddings_dict[path] for path in path_list])

    tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)
    print(f'Mean Accuracy: {np.mean(accuracy)}')
    print(f'VAL: {val}')
    print(f'FAR: {far}')


    model.classify = classify

    return np.mean(accuracy), val, far



def validate_first_task(
    num_tasks: int=5,
    num_pairs: int=3000,
    model_name: str='finetune',
    task: int=0
):
    classes = np.load('./data/classes.npy')
    num_classes = len(classes)
    val_loader = torch.load(f'./data/val_loader_{task}.pth')

    images = []
    labels = []

    for x, y in val_loader:
        images.append(x)
        labels.append(y)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    image_embeddings = defaultdict(list)
    for i in range(num_tasks):
        # Load models LWF
        resnet = InceptionResnetV1(classify=True, num_classes=num_classes, dropout_prob=0.2)
        resnet.to('cuda')
        resnet.load_state_dict(torch.load(f'trained_models/{model_name}/task{i + 1}_resnet.pth'))
        resnet.eval()
        resnet.classify = False
        for j in range(0, len(images), 64):
            x = images[j:j + 64].to('cuda')
            with torch.no_grad():
                y_pred, _ = resnet(x)
            y_pred = y_pred.cpu()
            image_embeddings[i].append(y_pred)

        image_embeddings[i] = torch.cat(image_embeddings[i], dim=0)
        del resnet
        torch.cuda.empty_cache()

    # Save embeddings
    for i in range(num_tasks):
        np.save(f'./data/NMC_evaluate/task_{i}_embeddings.npy', image_embeddings[i].numpy())

    num_images = len(images)
    issame_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Get all possible pairs except those in the diagonal
    same_pairs = np.argwhere(issame_matrix == True)
    diff_pairs = np.argwhere(issame_matrix == False)
    not_on_diag = same_pairs[0] != same_pairs[1]
    same_pairs = [same_pairs[0][not_on_diag], same_pairs[1][not_on_diag]]

    indices = np.arange(len(same_pairs[0]))
    np.random.seed(0)
    np.random.shuffle(indices)
    # Get 3000 random pairs
    indices = indices[:num_pairs]
    same_pairs = [same_pairs[0][indices], same_pairs[1][indices]]

    indices = np.arange(len(diff_pairs[0]))
    np.random.seed(0)
    np.random.shuffle(indices)
    # Get 3000 random pairs
    indices = indices[:num_pairs]
    diff_pairs = [diff_pairs[0][indices], diff_pairs[1][indices]]

    pairs_list = []
    issame_list = []
    for i in range(num_pairs):
        issame_list.append(True)
        pairs_list.extend([same_pairs[0][i].item(), same_pairs[1][i].item()])
        issame_list.append(False)
        pairs_list.extend([diff_pairs[0][i].item(), diff_pairs[1][i].item()])

    for task in range(num_tasks):
        embeddings = np.array([image_embeddings[task][pairs_list[i]].numpy() for i in range(len(pairs_list))])
        tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)
        print(f'Mean Accuracy: {np.mean(accuracy)}')