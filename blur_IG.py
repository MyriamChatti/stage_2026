import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import saliency.core as saliency


# dossier IRM
input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"

# dossier résultats
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/blurig_results"
os.makedirs(output_folder, exist_ok=True)


# modèle
model = tf.keras.applications.ResNet50(weights="imagenet")


# fonction modèle
def call_model_function(images, call_model_args=None, expected_keys=None):

    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        tape.watch(images)

        preds = model(images)

        class_idx = tf.argmax(preds[0])
        output = preds[:, class_idx]

    grads = tape.gradient(output, images)

    return {
        saliency.INPUT_OUTPUT_GRADIENTS: grads.numpy()
    }


# méthode BlurIG
blurig = saliency.BlurIG()


# boucle dataset
for file in os.listdir(input_folder):

    if file.endswith(".png"):

        print("Processing :", file)

        image_path = os.path.join(input_folder, file)

        image = np.array(Image.open(image_path).convert("RGB"))

        image = tf.image.resize(image, (224,224)).numpy()
        image = image / 255.0


        mask = blurig.GetMask(
            image,
            call_model_function,
            max_sigma=50,
            steps=50
        )


        vis = np.sum(np.abs(mask), axis=2)

        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)


        plt.figure(figsize=(6,5))
        plt.imshow(vis, cmap="inferno")
        plt.axis("off")


        save_path = os.path.join(
            output_folder,
            file.replace(".png","_blurig.png")
        )

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()


print("BlurIG terminé.")