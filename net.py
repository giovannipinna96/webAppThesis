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

    # create ResNet50 for triplet loss
    res_triplet = torchvision.models.resnet50()
    res_triplet.fc = torch.nn.Linear(in_features=2048,
                                     out_features=64,
                                     bias=True
                                     )
    res_triplet.load_state_dict(torch.load('triplet_64out_100.pt', map_location=torch.device('cpu')))
    # load svc for triplet
    with open("svm_triplet3", "rb") as pickle_in:
        clf = pickle.load(pickle_in)
    with open("sc_triplet3", "rb") as pickle_in:
        sc = pickle.load(pickle_in)

    # create ResNet50 for ii-loss, costum because has a particular strucutre
    res_ii = resNet50Costum(num_classes)
    res_ii.load_state_dict(torch.load('model_BEST.pt', map_location=torch.device('cpu')))
    with open("pickle_thres_mean_iiloss2", "rb") as pickle_in:
        thres_mean = pickle.load(pickle_in)
    threshold = thres_mean[0]
    mean = thres_mean[1]

    return res, res_triplet, clf, sc, res_ii, threshold, mean


def predict_res_ii(res_ii, threshold, mean, input):
    out_z, out_y = res_ii(input)
    if (((mean - out_z).norm(dim=0) ** 2).min() >= threshold):
        y_hat = np.argmax(out_y.detach().numpy())
    else:
        y_hat = torch.tensor(-1).numpy()

    return y_hat, out_y.detach()
