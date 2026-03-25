import os
import numpy as np
import tensorflow as tf
import cv2

from layer_cam import LayerCAM
# =========================
# PARAMÈTRES
# =========================
input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/layercam_results"
os.makedirs(output_folder, exist_ok=True)

IMG_SIZE = (224, 224)

# =========================
# MODELE (ResNet50)
# =========================
print("[INFO] Chargement du modèle...")
model = tf.keras.applications.ResNet50(weights='imagenet')

# couches multi-échelle
LAYER_NAMES = [
    'conv2_block3_out',
    'conv3_block4_out',
    'conv4_block6_out',
    'conv5_block3_out',
]

# =========================
# INIT LAYERCAM
# =========================
layercam = LayerCAM(
    model=model,
    layer_names=LAYER_NAMES,
    class_idx=None,
    fusion='weighted'
)

# =========================
# LOOP SUR IMAGES
# =========================
image_files = [
    f for f in os.listdir(input_folder)
    if f.endswith((".png", ".jpg", ".jpeg"))
]

for img_name in image_files:

    print(f"\nProcessing: {img_name}")

    image_path = os.path.join(input_folder, img_name)

    # load image
    img = cv2.imread(image_path)
    if img is None:
        print(f" erreur image: {image_path}")
        continue

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # =========================
    # LAYERCAM
    # =========================
    fused_map, layer_maps = layercam.explain(
        img,
        smooth_sigma=1.5,
        verbose=False
    )

    # =========================
    # HEATMAP
    # =========================
    heatmap = cv2.applyColorMap(
        (fused_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        (img * 255).astype(np.uint8),
        0.6,
        heatmap,
        0.4,
        0
    )

    # =========================
    # SAVE
    # =========================
    save_path = os.path.join(output_folder, f"{img_name}_layercam.png")
    cv2.imwrite(save_path, overlay)

    print(f" Saved: {save_path}")

print("\n LayerCAM terminé !")