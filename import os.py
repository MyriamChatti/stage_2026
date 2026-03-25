import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# PREPROCESSING

def preprocess(img):
    img = cv2.resize(img, (128,128))
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img



# KMEANS SEGMENTATION

def kmeans_segmentation(img, n_clusters=4):
    pixels = img.reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(img.shape)


# DATASET

class SpineDataset(Dataset):
    def __init__(self, image_folder):
        self.paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = preprocess(img)

        label = kmeans_segmentation(img, n_clusters=4)

        img = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(label).long()

        return img, label


# MINI U-NET

class UNetSmall(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x2 = self.pool(x1)

        x3 = F.relu(self.enc2(x2))

        x4 = F.interpolate(x3, scale_factor=2)
        x5 = F.relu(self.dec1(x4))

        out = self.out(x5)

        self.feature_maps = x3
        self.feature_maps.retain_grad()

        return out


# TRAINING

def train(model, dataloader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")


# GRAD-CAM PAR CLASSE

def grad_cam(model, input_tensor, class_idx):
    model.eval()

    output = model(input_tensor)

    class_map = output[:, class_idx, :, :]
    loss = class_map.mean()

    model.zero_grad()
    loss.backward()

    gradients = model.feature_maps.grad
    activations = model.feature_maps

    weights = torch.mean(gradients, dim=(2,3), keepdim=True)

    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = F.relu(cam)

    cam = cam.detach().numpy()
    cam = cv2.resize(cam, (128,128))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam


# 7. HEATMAP

def overlay_heatmap(img, cam):
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)

    img = (img*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay


# 8. MAIN PIPELINE

def main():
    dataset_path = "dataset_folder" 

    dataset = SpineDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNetSmall(n_classes=4)

    print(" Training...")
    train(model, loader, epochs=5)

    print(" Test sur une image...")

    img, _ = dataset[0]
    img_input = img.unsqueeze(0)

    structures = {
        "Sac thecal": 1,
        "Disque": 2,
        "Muscle": 3
    }

    plt.figure(figsize=(12,4))

    for i, (name, cls) in enumerate(structures.items()):
        cam = grad_cam(model, img_input, cls)
        overlay = overlay_heatmap(img.squeeze().numpy(), cam)

        plt.subplot(1,3,i+1)
        plt.title(name)
        plt.imshow(overlay)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# RUN

if __name__ == "__main__":
    main()