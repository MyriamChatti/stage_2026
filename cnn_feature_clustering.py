import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import preprocess_input


input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"

output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/cnn_clusters"

os.makedirs(output_folder, exist_ok=True)


# modèle
model = tf.keras.applications.ResNet50(weights="imagenet")

feature_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer("conv5_block3_out").output
)


for file in os.listdir(input_folder):

    if file.endswith(".png"):

        print("Processing :", file)

        path = os.path.join(input_folder, file)

        image = Image.open(path).convert("RGB")
        image = image.resize((224,224))

        img = np.array(image)

        original = img.copy()

        img = preprocess_input(img.astype(np.float32))

        features = feature_model.predict(img[np.newaxis])[0]

        h,w,c = features.shape

        features = features.reshape(-1,c)

        kmeans = KMeans(n_clusters=4)

        labels = kmeans.fit_predict(features)

        clusters = labels.reshape(h,w)

        clusters = tf.image.resize(
            clusters[...,np.newaxis],
            (224,224),
            method="nearest"
        ).numpy()

        plt.figure(figsize=(6,6))

        plt.imshow(original)

        plt.imshow(clusters[:,:,0], cmap="viridis", alpha=0.5)

        plt.axis("off")

        save_path = os.path.join(
            output_folder,
            file.replace(".png","_clusters.png")
        )

        plt.savefig(save_path, bbox_inches="tight")

        plt.close()

print("Clustering terminé.")