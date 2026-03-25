import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions



# chemins


input_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"

output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/predictions_gradcam"

os.makedirs(output_folder, exist_ok=True)


# modèle

model = tf.keras.applications.ResNet50(weights="imagenet")

last_conv_layer = model.get_layer("conv5_block3_out")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)


# fonction GradCAM

def compute_gradcam(image):

    img_tensor = tf.convert_to_tensor(image[np.newaxis])

    with tf.GradientTape() as tape:

        conv_outputs, preds = grad_model(img_tensor)

        class_idx = tf.argmax(preds[0])

        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    conv_outputs = conv_outputs[0]
    grads = grads[0]

    weights = tf.reduce_mean(grads, axis=(0,1))

    cam = np.zeros(conv_outputs.shape[:2])

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:,:,i]

    cam = np.maximum(cam, 0)

    cam = cam / (cam.max() + 1e-8)

    cam = tf.image.resize(cam[...,np.newaxis], (224,224)).numpy()

    return cam[:,:,0]


# boucle images

for file in os.listdir(input_folder):

    if file.endswith(".png"):

        print("\nProcessing :", file)

        path = os.path.join(input_folder, file)

        image = Image.open(path).convert("RGB")
        image = image.resize((224,224))

        img = np.array(image)

        original = img.copy()

        x = preprocess_input(img.astype(np.float32))

        x = np.expand_dims(x, axis=0)


        # prédiction

        preds = model.predict(x)

        decoded = decode_predictions(preds, top=5)[0]

        print("Top classes :")

        for rank, (imagenetID, label, prob) in enumerate(decoded):
            print(f"{rank+1}. {label} : {prob:.3f}")


        # GradCAM

        heatmap = compute_gradcam(preprocess_input(img.astype(np.float32)))


        # affichage
    

        plt.figure(figsize=(6,6))

        plt.imshow(original)

        plt.imshow(heatmap, cmap="inferno", alpha=0.5)

        title = f"{decoded[0][1]} ({decoded[0][2]:.2f})"

        plt.title(title)

        plt.axis("off")


        save_path = os.path.join(
            output_folder,
            file.replace(".png","_gradcam_pred.png")
        )

        plt.savefig(save_path, bbox_inches="tight")

        plt.close()


print("\nAnalyse terminée.")