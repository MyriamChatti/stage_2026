import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import saliency.core as saliency

# dossier images
input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"

# dossier sortie
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/xrai_results"
os.makedirs(output_folder, exist_ok=True)

# modèle CNN (exemple)
model = tf.keras.applications.ResNet50(weights="imagenet")

# fonction nécessaire pour XRAI
def call_model_function(images, call_model_args=None, expected_keys=None):

    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)

        class_idx = tf.argmax(preds[0])
        output = preds[:, class_idx]

    grads = tape.gradient(output, images)

    return {saliency.INPUT_OUTPUT_GRADIENTS: grads.numpy()}


# initialiser XRAI
xrai = saliency.XRAI()


# parcourir toutes les images
for file in os.listdir(input_folder):

    if file.endswith(".png"):

        image_path = os.path.join(input_folder, file)

        print("Processing :", file)

        image = np.array(Image.open(image_path).convert("RGB"))

        image = tf.image.resize(image, (224,224)).numpy()
        image = image / 255.0

        # calcul XRAI
        xrai_mask = xrai.GetMask(
            image,
            call_model_function
        )

        # sauvegarde
        plt.figure(figsize=(6,5))
        plt.imshow(xrai_mask, cmap="jet")
        plt.axis("off")

        save_path = os.path.join(output_folder, file.replace(".png","_xrai.png"))
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

print("XRAI terminé pour toutes les images.")