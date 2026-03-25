import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


# chemins

input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"

output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/scorecam_results"
os.makedirs(output_folder, exist_ok=True)


# modèle

model = tf.keras.applications.ResNet50(weights="imagenet")

last_conv_layer = model.get_layer("conv5_block3_out")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)


# normalisation

def normalize(x):
    x = x - np.min(x)
    return x / (np.max(x) + 1e-8)


# ScoreCAM

def scorecam(image):

    image_tensor = tf.convert_to_tensor(image[np.newaxis])

    conv_outputs, preds = grad_model(image_tensor)

    class_idx = tf.argmax(preds[0])

    conv_outputs = conv_outputs[0].numpy()

    heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i in range(conv_outputs.shape[-1]):

        fmap = conv_outputs[:, :, i]

        fmap = normalize(fmap)

        fmap_resized = tf.image.resize(
            fmap[..., np.newaxis],
            (224,224)
        ).numpy()

        masked_image = image * fmap_resized

        masked_image = preprocess_input(masked_image.astype(np.float32))

        score = model.predict(masked_image[np.newaxis], verbose=0)[0][class_idx]

        heatmap += score * fmap

    heatmap = np.maximum(heatmap, 0)

    heatmap = normalize(heatmap)

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (224,224)
    ).numpy()

    return heatmap[:, :, 0]


# boucle images

for file in os.listdir(input_folder):

    if file.endswith(".png"):

        print("Processing :", file)

        path = os.path.join(input_folder, file)

        image = np.array(Image.open(path).convert("RGB"))

        image = tf.image.resize(image, (224,224)).numpy()

        original = image.astype(np.uint8)

        image = image / 255.0

        heatmap = scorecam(image)

        plt.figure(figsize=(6,6))

        plt.imshow(original)

        plt.imshow(heatmap, cmap="inferno", alpha=0.5)

        plt.axis("off")

        save_path = os.path.join(
            output_folder,
            file.replace(".png","_scorecam.png")
        )

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

        plt.close()


print("ScoreCAM terminé.")