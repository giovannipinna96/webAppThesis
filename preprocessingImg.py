from PIL import Image
from torchvision import transforms


def preprocessimg(img):
    input_image = Image.open(img).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_model = input_tensor.unsqueeze(0)
    return input_model
