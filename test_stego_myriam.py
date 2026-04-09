import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# modèle STEGO
from modules.stego import Stego
from utils.common import get_transform

device = "cpu"

# charger modèle pré-entraîné (cocostuff par exemple)
model_path = "../saved_models/cocostuff27_vit_base_5.ckpt"

model = Stego.load_from_checkpoint(model_path)
model = model.to(device)
model.eval()

# image IRM (la tienne)
img_path = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction/patient_0112_testA.png"

image = Image.open(img_path).convert("RGB")

transform = get_transform(res=320, is_label=False)
img_tensor = transform(image).unsqueeze(0).to(device)

# inference
with torch.no_grad():
    code = model(img_tensor)
    clusters = torch.argmax(code, dim=1)

# affichage
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(clusters[0].cpu(), cmap="jet")
plt.title("Segmentation STEGO")
plt.axis("off")

plt.show()
