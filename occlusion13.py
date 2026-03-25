import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from saliency.core.occlusion import Occlusion
from saliency.core.base import OUTPUT_LAYER_VALUES
from tensorflow.keras.applications.resnet50 import preprocess_input


# -----------------------------
# chemins
# -----------------------------

input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"

output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/occlusion_results"
os.makedirs(output_folder, exist_ok=True)


# -----------------------------
# modèle
# -----------------------------

model = tf.keras.applications.ResNet50(weights="imagenet")


# -----------------------------
# fonction pour saliency
# -----------------------------

def call_model_function(images, call_model_args=None, expected_keys=None):

    preds = model(images, training=False).numpy()

    # prendre la classe prédite
    class_idx = np.argmax(preds, axis=1)

    # récupérer le score de cette classe
    scores = preds[np.arange(len(preds)), class_idx]

    return {OUTPUT_LAYER_VALUES: scores}


# -----------------------------
# méthode occlusion
# -----------------------------

occlusion = Occlusion()


# -----------------------------
# paramètres (important)
# -----------------------------

WINDOW_SIZE = 60
STRIDE = 20


# -----------------------------
# boucle sur les images
# -----------------------------

for filename in os.listdir(input_folder):

    if filename.endswith(".png"):

        print("Processing :", filename)

        image_path = os.path.join(input_folder, filename)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((224,224))

        image_np = np.array(image).astype(np.float32)
        image_np = preprocess_input(image_np)


        # -----------------------------
        # occlusion rapide
        # -----------------------------

        mask = np.zeros_like(image_np)

        for r in range(0, 224 - WINDOW_SIZE, STRIDE):
            for c in range(0, 224 - WINDOW_SIZE, STRIDE):

                occluded = image_np.copy()
                occluded[r:r+WINDOW_SIZE, c:c+WINDOW_SIZE, :] = 0

                y1 = call_model_function(np.expand_dims(image_np,0))[OUTPUT_LAYER_VALUES]
                y2 = call_model_function(np.expand_dims(occluded,0))[OUTPUT_LAYER_VALUES]

                diff = y1 - y2

                mask[r:r+WINDOW_SIZE, c:c+WINDOW_SIZE] += diff


        # -----------------------------
        # normalisation
        # -----------------------------

        mask = np.abs(mask)
        mask = (mask - mask.min())/(mask.max() - mask.min() + 1e-8)


        # -----------------------------
        # affichage
        # -----------------------------

        plt.figure(figsize=(6,6))

        plt.imshow(image)
        plt.imshow(mask.mean(axis=-1), cmap="jet", alpha=0.5)

        plt.axis("off")

        save_path = os.path.join(output_folder, "occlusion_" + filename)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


print("Occlusion terminé")