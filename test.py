import torch
from PIL import Image
from torchvision import transforms
from model import MaxViT

DEVICE = torch.device("cpu")
model = MaxViT(num_classes=11, win=7, drop_path_rate=0.15)
checkpoint = torch.load("maxvit_kaggle_best.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint.get('sd', checkpoint))
model.eval()

img = Image.new("RGB", (256, 256), color="red")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(img).unsqueeze(0)

try:
    with torch.no_grad():
        outputs = model(input_tensor)
        print("Success:", outputs.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
