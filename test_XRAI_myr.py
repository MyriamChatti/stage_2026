import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from saliency.core.xrai import XRAI
from saliency.core.base import INPUT_OUTPUT_GRADIENTS

# =========================
# MODELE (TEST)
# 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
model.eval()

# =========================
# LOAD IMAGE
# =========================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image introuvable : {path}")

    img = cv2.resize(img, (128,128))
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())

    return img

# =========================
# CALL MODEL FUNCTION


def call_model_function(images, call_model_args=None, expected_keys=None):

    images = torch.tensor(images).float()
    images = images.unsqueeze(1)
    images.requires_grad = True

    outputs = model(images)
    target = outputs[:, 1].sum()

    model.zero_grad()
    target.backward()

    gradients = images.grad.detach().numpy()
    gradients = np.squeeze(gradients, axis=1)

    return {INPUT_OUTPUT_GRADIENTS: gradients}

# =========================
# XRAI
# =========================
xrai = XRAI()

# =========================
# DOSSIERS
# =========================
image_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/predictions_xrai"

os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.endswith((".png",".jpg",".jpeg"))]

# =========================
# LOOP
# =========================
for img_name in image_files:

    print(f"Processing: {img_name}")

    image_path = os.path.join(image_folder, img_name)
    img = load_image(image_path)

    #  XRAI
    mask = xrai.GetMask(
        x_value=img,
        call_model_function=call_model_function
    )

    mask = np.abs(mask)
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    heatmap = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )

    save_path = os.path.join(output_folder, f"{img_name}_XRAI.png")
    cv2.imwrite(save_path, overlay)

    print(f"Saved: {save_path}")

print(" XRAI terminé")