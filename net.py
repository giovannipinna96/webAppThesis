import torch
from torchvision.models import resnet50


def create_model():
    return resnet50(pretrained='Imagenet')


def predictimg(model, input_model):
    model.eval()
    with torch.no_grad():
        out = model(input_model)

    #
    # Load the file containing the 1,000 labels for the ImageNet dataset classes
    #
    with open('imagenet1000_clsidx_to_labels.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    #
    # Find the index (tensor) corresponding to the maximum score in the out tensor.
    # Torch.max function can be used to find the information
    #
    _, index = torch.max(out, 1)
    #
    # Find the score in terms of percentage by using torch.nn.functional.softmax function
    # which normalizes the output to range [0,1] and multiplying by 100
    #
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    #
    # Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
    #
    _, indices = torch.sort(out, descending=True)
    top5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
    return top5
