import pickle

import numpy as np
import torch
import torchvision

from customResNet50 import resNet50Costum


def create_models(num_classes):
    # create ResNet50
    res = torchvision.models.resnet50()
    res.fc = torch.nn.Linear(in_features=2048,
                             out_features=num_classes,
                             bias=True
                             )
    res.load_state_dict(torch.load('resnet50_augmentation.pt', map_location=torch.device('cpu')))

    # create ResNet50 for ii-loss, costum because has a particular strucutre
    res_ii = resNet50Costum(num_classes)
    res_ii.load_state_dict(torch.load('model_BEST9.pt', map_location=torch.device('cpu')))
    with open("pickle_thres_mean_iiloss9", "rb") as pickle_in:
        thres_mean = pickle.load(pickle_in)
    threshold = thres_mean[0]
    mean = thres_mean[1]
    threshold2 = thres_mean[2]

    return res, res_ii, threshold, threshold2, mean


def predict_res_ii(res_ii, threshold, threshold2, mean, input):
    out_z, out_y = res_ii(input)
    outlier_score_val = outlier_score(out_z, mean)
    if outlier_score_val <= threshold2:
        y_hat = np.argmax(out_y.detach().numpy())
    else:
        y_hat = -1

    return out_z, outlier_score_val, y_hat, out_y.detach()

def outlier_score(embeddings:torch.Tensor, train_class_means:torch.Tensor):
    '''
    Compute the outlier score for the given batch of embeddings and class means obtained from the training set.
    The outlier score for a single datapoint is defined as min_j(||z - m_j||^2), where j is a category and m_j is the mean embedding of this class.
    Parameters
    ----------
    embeddings: a torch.Tensor of shape (N, D) where N is the number of data points and D is the embedding dimension.
    train_class_means: a torch.Tensor of shape (K, D) where K is the number of classes.
    Returns
    -------
    a torch.Tensor of shape (N), representing the outlier score for each of the data points.
    '''
    assert len(embeddings.shape) == 2, f"Expected 2D tensor of shape N ⨉ D (N=datapoints, D=embedding dimension), got {embeddings.shape}"
    assert len(train_class_means.shape) == 2, f"Expected 2D tensor of shape K ⨉ D (K=num_classes, D=embedding dimension), got {train_class_means.shape}"
    # create an expanded version of the embeddings of dimension N ⨉ K ⨉ D, useful for subtracting means
    embeddings_repeated = embeddings.unsqueeze(1).repeat((1, train_class_means.shape[0], 1))
    # compute the difference between the embeddings and the class means
    difference_from_mean = embeddings_repeated - train_class_means
    # compute the squared norm of the difference (N ⨉ K matrix)
    norm_from_mean = difference_from_mean.norm(dim=2)**2
    # get the min for each datapoint
    return norm_from_mean.min(dim=1).values
