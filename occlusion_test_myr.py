import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from saliency.core.occlusion import Occlusion
from saliency.core.base import OUTPUT_LAYER_VALUES
# =========================
# 1. MODELE (comme IG)
# =========================
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
# 2. LOAD IMAGE
# =========================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f" Image introuvable : {path}")

    img = cv2.resize(img, (128,128))
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())

    return img

# =========================
# 3. CALL MODEL FUNCTION ( différent IG)
# =========================
def call_model_function(images, call_model_args=None, expected_keys=None):

    images = torch.tensor(images).float()
    images = images.unsqueeze(1)  # (B,1,H,W)

    outputs = model(images)

    #  IMPORTANT : ici on retourne les scores
    # on prend la classe 1 (comme IG)
    scores = outputs[:, 1]   # shape (B,)
    return {OUTPUT_LAYER_VALUES: scores.detach().numpy()}
# =========================
# 4. OCCLUSION
# =========================
occ = Occlusion()

# =========================
# 5. DOSSIERS
# =========================
image_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/occlusion_results_test"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.endswith((".png",".jpg",".jpeg"))]

# =========================
# 6. LOOP
# =========================
for img_name in image_files:

    print(f"Processing: {img_name}")

    image_path = os.path.join(image_folder, img_name)
    img = load_image(image_path)

    mask = occ.GetMask(
        x_value=img,
        call_model_function=call_model_function,
        size=15,     # taille fenêtre
        value=0      # pixel noir
    )

    # =========================
    # NORMALISATION
    # =========================
    mask = np.abs(mask)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    heatmap = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )

    save_path = os.path.join(output_folder, f"{img_name}_occlusion.png")
    cv2.imwrite(save_path, overlay)

    print(f"Saved: {save_path}")

print("\n Occlusion terminé !")