import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import saliency.core as saliency
from tensorflow.keras.applications.resnet50 import preprocess_input

# 
# chemins
# 

input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/xai_consensus_results"
os.makedirs(output_folder, exist_ok=True)

# 
# modèle

model = tf.keras.applications.ResNet50(weights="imagenet")

last_conv_layer = model.get_layer("conv5_block3_out")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

# utilitaire

def normalize(x):
    x = x - np.min(x)
    return x / (np.max(x) + 1e-8)

# GradCAM interface

def call_gradcam(images, call_model_args=None, expected_keys=None):

    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:

        conv_outputs, preds = grad_model(images)

        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    return {
        saliency.CONVOLUTION_LAYER_VALUES: conv_outputs.numpy(),
        saliency.CONVOLUTION_OUTPUT_GRADIENTS: grads.numpy()
    }

# Gradient interface

def call_gradient(images, call_model_args=None, expected_keys=None):

    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:

        tape.watch(images)
        preds = model(images)

        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, images)

    return {saliency.INPUT_OUTPUT_GRADIENTS: grads.numpy()}

# méthodes

gradcam = saliency.GradCam()
gradient = saliency.GradientSaliency()
ig = saliency.IntegratedGradients()

# boucle images

for file in os.listdir(input_folder):

    if file.endswith(".png"):

        print("Processing :", file)

        path = os.path.join(input_folder, file)

        image = np.array(Image.open(path).convert("RGB"))
        image = tf.image.resize(image, (224,224)).numpy()

        original = image.astype(np.uint8)

        image = preprocess_input(image.astype(np.float32))

        # GradCAM
        mask_gradcam = gradcam.GetMask(image, call_gradcam)
        mask_gradcam = np.sum(mask_gradcam, axis=2)
        mask_gradcam = normalize(mask_gradcam)

        # SmoothGrad
        mask_smooth = gradient.GetSmoothedMask(
            image,
            call_gradient,
            stdev_spread=0.15,
            nsamples=25
        )
        mask_smooth = np.sum(np.abs(mask_smooth), axis=2)
        mask_smooth = normalize(mask_smooth)

        # Integrated Gradients
        mask_ig = ig.GetMask(image, call_gradient, x_steps=25)
        mask_ig = np.sum(np.abs(mask_ig), axis=2)
        mask_ig = normalize(mask_ig)

        # fusion robuste
        combined = np.median(
            [mask_gradcam, mask_smooth, mask_ig],
            axis=0
        )

        combined = normalize(combined)

        # affichage
        plt.figure(figsize=(6,6))

        plt.imshow(original)
        plt.imshow(combined, cmap="inferno", alpha=0.5)

        plt.axis("off")

        save_path = os.path.join(
            output_folder,
            file.replace(".png","_xai_consensus.png")
        )

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

print("Pipeline XAI terminé.")