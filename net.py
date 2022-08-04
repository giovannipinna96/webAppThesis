import torch
import torchvision

from customResNet50 import resNet50Costum


def create_models(num_classes):
    # create ResNet50
    res = torchvision.models.resnet50()
    res.load_state_dict(torch.load('triplet256e20m2_nearest.pt', map_location=torch.device('cpu')))

    # create ResNet50 for triplet loss
    res_triplet = torchvision.models.resnet50()
    res_triplet.fc = torch.nn.Linear(in_features=2048,
                                     out_features=256,
                                     bias=True
                                     )
    res_triplet.load_state_dict(torch.load('triplet256e20m2_nearest.pt', map_location=torch.device('cpu')))
    # load svc for triplet
    # TODO salvare con pikle

    # create ResNet50 for ii-loss, costum because has a particular strucutre
    res_ii = resNet50Costum(num_classes)
    res_ii.load_state_dict(torch.load('triplet256e20m2_nearest.pt', map_location=torch.device('cpu')))

    return res, res_triplet, res_ii


def predict_res(res, input_model):
    pass


def predict_res(res_triplet, input_model):
    pass


def predict_res_ii(res, input_model):
    pass

# def predictimg(model, input_model):
#     model.eval()
#     with torch.no_grad():
#         out = model(input_model)
#
#     #
#     # Load the file containing the 1,000 labels for the ImageNet dataset classes
#     #
#     with open('imagenet1000_clsidx_to_labels.txt') as f:
#         labels = [line.strip() for line in f.readlines()]
#     #
#     # Find the index (tensor) corresponding to the maximum score in the out tensor.
#     # Torch.max function can be used to find the information
#     #
#     _, index = torch.max(out, 1)
#     #
#     # Find the score in terms of percentage by using torch.nn.functional.softmax function
#     # which normalizes the output to range [0,1] and multiplying by 100
#     #
#     percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#     #
#     # Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
#     #
#     _, indices = torch.sort(out, descending=True)
#     top5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
#     return top5
