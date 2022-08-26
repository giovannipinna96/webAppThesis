from PIL import Image
from torchvision import transforms


def preprocessimg(img):
    input_image = Image.open(img).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    input_tensor = preprocess(input_image)
    input_model = input_tensor.unsqueeze(0)
    return input_model
