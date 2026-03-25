import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input


# -----------------------
# chemins
# -----------------------

input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"

output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/smoothgrad_results"
os.makedirs(output_folder, exist_ok=True)


# -----------------------
# modèle
# -----------------------

model = tf.keras.applications.ResNet50(weights="imagenet")


# -----------------------
# fonction gradient
# -----------------------

def compute_gradient(image):

    image = tf.convert_to_tensor(image)

    with tf.GradientTape() as tape:

        tape.watch(image)

        preds = model(image)

        class_idx = tf.argmax(preds[0])

        score = preds[:, class_idx]

    grads = tape.gradient(score, image)

    return grads.numpy()[0]


# -----------------------
# paramètres smoothgrad
# -----------------------

N_SAMPLES = 30
NOISE_LEVEL = 0.15


# -----------------------
# boucle images
# -----------------------

for filename in os.listdir(input_folder):

    if filename.endswith(".png"):

        print("Processing :", filename)

        path = os.path.join(input_folder, filename)

        image = Image.open(path).convert("RGB")
        image = image.resize((224,224))

        image_np = np.array(image).astype(np.float32)

        image_np = preprocess_input(image_np)

        image_np = np.expand_dims(image_np, axis=0)

        smooth_grad = np.zeros_like(image_np[0])

        for i in range(N_SAMPLES):

            noise = np.random.normal(0, NOISE_LEVEL, image_np.shape)

            noisy_image = image_np + noise

            grad = compute_gradient(noisy_image)

            smooth_grad += grad

        smooth_grad /= N_SAMPLES

        smooth_grad = np.abs(smooth_grad)

        smooth_grad = (smooth_grad - smooth_grad.min()) / (smooth_grad.max() - smooth_grad.min() + 1e-8)


        plt.figure(figsize=(6,6))

        plt.imshow(image)
        plt.imshow(smooth_grad.mean(axis=-1), cmap="magma", alpha=0.5)

        plt.axis("off")

        save_path = os.path.join(output_folder, "smoothgrad_" + filename)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


print("SmoothGrad terminé")