import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from trainer import Trainer
import sys
import torch


import time
import datetime
import numpy as np
import matplotlib.cm as cm

from torch.utils.data import DataLoader
from scipy.ndimage import median_filter
from skimage.restoration import denoise_nl_means, estimate_sigma


import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import argparse
#pour segmentation des tructures en couleurs
from skimage import measure
from data.nuclei_dataset import NucleiDataset
from trainer import Trainer
from scipy.ndimage import sobel
from metric import ImageMetrics
from sklearn.cluster import KMeans



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='adgan')
parser.add_argument('--resume', type=str, default='latest.pth')
parser.add_argument('--seed', type=int, default=10)

#Network
parser.add_argument('--dimensions', type=int, default=2, help='use 2D or 3D data, 2 for 2D, 3 for 3D')
parser.add_argument('--num_c_dim', type=int, default=64, help='the number of dimensions for the channel.')

#Datasets
parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                    help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')

parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--no_synB', action='store_true', help='if specified, do not synthesis datasetsB')
parser.add_argument('--no_inst', action='store_true', help='if specified, do not use instance segmentation')

parser.add_argument('--ellipse_min_radius', type=int, default=20)
parser.add_argument('--ellipse_max_radius', type=int, default=30)
parser.add_argument('--ellipse_min_num', type=int, default=5)
parser.add_argument('--ellipse_max_num', type=int, default=15)

parser.add_argument('--preprocess', type=str, default='crop')
parser.add_argument('--dataroot', default='datasets/YourDATA')

# GAN
parser.add_argument('--lambda_rec', type=float, default=20,help='weight for image-level reconstruction')
parser.add_argument('--lambda_cyc', type=float, default=20,help='weight for cycle consistency loss')
parser.add_argument('--lambda_ctr', type=float, default=1,help='weight for feature-level reconstruction')
parser.add_argument('--no_adt', action='store_true', help='if specified, do not Aligned Disentangling Training')
parser.add_argument('--gan_mode', type=str, default='lsgan',
                    help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')

# Optimization
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--iter_count', type=int, default=1, help='the starting iteration count')
parser.add_argument('--n_iters', type=int, default=5000, help='number of iterations with the initial learning rate')
parser.add_argument('--n_iters_decay', type=int, default=5000, help='number of iterations to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='linear',
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')


opts = parser.parse_args()




# Création du fichier de sortie test avec datetime

tm = time.time()
dt_str = datetime.datetime.fromtimestamp(tm).strftime("%Y-%m-%d_%H-%M-%S")

results_dir = "logs"
os.makedirs(results_dir, exist_ok=True)

out_csv = os.path.join(results_dir, f"output_{dt_str}.csv")

print("[TEST] Résultats sauvegardés dans :", out_csv)






#######################################################
#pour la segmentation des structures en couleurs

def colorize_segmentation(mask, color=(255, 0, 0)):
    h, w = mask.shape
    #création du masque coloré et convertion du masque gris en couleurs
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask > 0] = color
    return color_mask


def extract_contours(mask):
    contours = measure.find_contours(mask.astype(float), level=0.5)
    return contours

#superposition du masque

def overlay_multiclass(image_gray, labels):

    image_rgb = np.stack([image_gray]*3, axis=-1)
    overlay = image_rgb.copy()

    n_clusters = len(np.unique(labels))

    # palette daltonisme-compatible pour les 10 clusters
    cmap = cm.get_cmap("tab10", n_clusters)  #palette de couleurs adapté (daltonisme) = palette colorblind-safe
    colors = (cmap(range(n_clusters))[:, :3] * 255).astype(np.uint8)

    alpha = 0.5

    for k in range(n_clusters):
        mask = labels == k
        overlay[mask] = (
            (1-alpha)*image_rgb[mask] +
            alpha*colors[k]
        ).astype(np.uint8)

    return overlay




def save_overlay_with_contours(image, overlay, contours, save_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='yellow')
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()















def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.
    Args:
        seed: seed to check
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Using manual seed: {seed}".format(seed=seed))
    return

def dice_loss_chill(output, gt):
    num = (output*gt).sum(dim=[2, 3])
    denom = output.sum(dim=[2, 3]) + gt.sum(dim=[2, 3]) + 0.001
    return num, denom




def save_prediction(tensor, save_path):
    #Sauvegarde une prédiction du modèle sous forme d'image PNG
    
    print("[DEBUG] Type tensor :", type(tensor))
    print("[DEBUG] Shape tensor :", tensor.shape)

    # tensor --> numpy
    img = tensor[0, 0].detach().cpu().numpy()

    print("[DEBUG] Numpy min/max :", img.min(), img.max())

    # normalisation [-1,1] --> [0,255]
    img = (img + 1) * 127.5
    img = img.clip(0, 255).astype("uint8")

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print("[DEBUG] Image sauvegardée :", save_path)




#création d'une nouvelle classe pour le débruitage (essaie d'amélioration)
#Correction d’un gradient linéaire de fond(inhomogénéité d’illumination).
#Débruitage linéaire



def save_denoised_image(img, save_path):

    img_np = img[0, 0].detach().cpu().numpy()
    img_np = (img_np + 1) * 127.5
    img_np = img_np.clip(0, 255).astype("uint8")

    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# débruitage gradient linéaire

class LinearGradientDenoiser:
    
    def __init__(self, kernel_size=51, save_dir="YourDATA/denoised/linear"):
        self.kernel_size = kernel_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def denoise(self, img, idx):
        background = F.avg_pool2d(
            img,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )

        img_corrected = img - background

        save_denoised_image(
            img_corrected,
            os.path.join(self.save_dir, f"img_{idx:03d}.png")
        )

        return img_corrected




# filtre gaussien

def gaussian_blur_manual(img, kernel_size=5, sigma=1.0):
    #flou gaussien implémenté manuellement (compatible toutes versions PyTorch)

    coords = torch.arange(kernel_size, device=img.device) - kernel_size // 2
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")

    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    return F.conv2d(img, kernel, padding=kernel_size // 2)


class GaussianDenoiser:
    def __init__(self, kernel_size=5, sigma=1.0, save_dir="YourDATA/denoised/gaussian"):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def denoise(self, img, idx):
        img_denoised = gaussian_blur_manual(
            img,
            kernel_size=self.kernel_size,
            sigma=self.sigma
        )

        save_denoised_image(
            img_denoised,
            os.path.join(self.save_dir, f"img_{idx:03d}.png")
        )

        return img_denoised







# filtre median

class MedianDenoiser:
    def __init__(self, kernel_size=3, save_dir="YourDATA/denoised/median"):
        self.kernel_size = kernel_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def denoise(self, img, idx):
        # torch à  numpy
        img_np = img[0, 0].detach().cpu().numpy()

        # filtre médian
        img_denoised_np = median_filter(img_np, size=self.kernel_size)

        # numpy à torch
        img_denoised = torch.from_numpy(img_denoised_np).float()
        img_denoised = img_denoised.unsqueeze(0).unsqueeze(0).to(img.device)

        save_denoised_image(
            img_denoised,
            os.path.join(self.save_dir, f"img_{idx:03d}.png")
        )

        return img_denoised




# bruit ricien

class RicianDenoiser:
    def __init__(self, sigma=0.05, save_dir="YourDATA/denoised/rician"):
        self.sigma = sigma
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def denoise(self, img, idx):
        n1 = torch.randn_like(img) * self.sigma
        n2 = torch.randn_like(img) * self.sigma
        img_denoised = torch.sqrt((img + n1) ** 2 + n2 ** 2)

        save_denoised_image(
            img_denoised,
            os.path.join(self.save_dir, f"img_{idx:03d}.png")
        )

        return img_denoised








# NLM 


class NLMDenoiser:
    
    def __init__(self, patch_size=5, patch_distance=6, save_dir="YourDATA/denoised/nlm"):
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)



    def denoise(self, img, idx):
        
        img_np = img[0, 0].detach().cpu().numpy()

        # remise en [0,1] pour skimage
        img_np = (img_np + 1) / 2.0
        
        sigma_est = np.mean(estimate_sigma(img_np, channel_axis=None))

        
        img_denoised_np = denoise_nl_means(
            img_np,
            h=0.8 * sigma_est,
            sigma=sigma_est,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            fast_mode=True,
            channel_axis=None
        )

        img_denoised = torch.from_numpy(img_denoised_np).float()
        img_denoised = img_denoised * 2 - 1  # retour en [-1,1]
        img_denoised = img_denoised.unsqueeze(0).unsqueeze(0).to(img.device)

    
        save_denoised_image(
            img_denoised,
            os.path.join(self.save_dir, f"img_{idx:03d}.png")
        )

        return img_denoised








class DenoisedDataset(torch.utils.data.Dataset):
    
    
    def __init__(self, base_dataset, denoiser):
        self.base_dataset = base_dataset
        self.denoiser = denoiser

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        img = data["A"].unsqueeze(0)        # (1,1,H,W)
        img = self.denoiser.denoise(img, idx)
        data["A"] = img.squeeze(0)          # (1,H,W)
        return data





# test
if __name__ == "__main__":

    tm = time.time()
    dt_str = datetime.datetime.fromtimestamp(tm).strftime("%Y-%m-%d_%H-%M-%S")

    print(f"[INFO] Lancement test & segmentation à {dt_str}")

 
    check_manual_seed(opts.seed)

    trainer = Trainer(opts)
    trainer.load(opts.resume)


    #pour calcul des metrics
    metrics = ImageMetrics()


    test_loader = DataLoader(dataset=NucleiDataset(opts, "test"), batch_size=1, shuffle=False)

    denoisers = {
    "linear": LinearGradientDenoiser(kernel_size=51),
    "gaussian": GaussianDenoiser(),
    "rician_0.1": RicianDenoiser(sigma=0.1),
    "rician_0.5": RicianDenoiser(sigma=0.5),
    "rician_0.9": RicianDenoiser(sigma=0.9),
    "median": MedianDenoiser(),
    "nlm": NLMDenoiser()
    }

    # Boucle filtres + segmentation couleur
    
    for name, denoiser in denoisers.items():

        results_dir = f"logs/results_75_epochs_{name}"
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n=== Test avec filtre : {name.upper()} ===")

        for i, test_data in enumerate(test_loader):
            img = test_data["A"].to(trainer.device)

            img_denoised = denoiser.denoise(img, i)

            with torch.no_grad():
                prediction = trainer(img_denoised)
            # image originale normalisée
            img_np = img[0, 0].cpu().numpy()
            img_np = (img_np + 1) / 2.0

            # prédiction normalisée
            pred_np = prediction[0, 0].cpu().numpy()
            pred_np = (pred_np + 1) / 2.0

            ssim_value = metrics.compute_ssim(
                img_np.astype(np.float32),
                pred_np.astype(np.float32)
            )

            print(f"[SSIM] {name} | image {i} : {ssim_value:.4f}")



            mse = np.mean((img_np - pred_np) ** 2)

            # pour éviter division par zéro
            if mse == 0:
                psnr = float("inf")
            else:
                psnr = 10 * np.log10(1.0 / mse)

            print(f"[PSNR] {name} | image {i} : {psnr:.2f}")




            # Sauvegarde prédiction brute
            save_prediction(
                prediction,
                os.path.join(results_dir, f"prediction_{i:03d}.png")
            )


           # ================================
            # SEGMENTATION MULTICLASSE
            # ================================

            # Image IRM originale en niveaux de gris
            img_gray = img[0, 0].cpu().numpy()
            img_gray = ((img_gray + 1) * 127.5).astype(np.uint8)

            pred = (prediction[0, 0].cpu().numpy() + 1) / 2.0
            H, W = pred.shape

            # Gradient (contours)
            grad_x = sobel(pred, axis=1)
            grad_y = sobel(pred, axis=0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Coordonnées spatiales
            y_coords, x_coords = np.indices((H, W))

            # Features
            X = np.stack([
                pred,
                grad_mag,
                y_coords / H,
                x_coords / W
            ], axis=-1)

            flat = X.reshape(-1, 4)

            # ================================
            # TEST DIFFÉRENTS N_CLUSTERS
            # ================================

            for n_clusters in [2, 3, 10]:

                print(f"\n[INFO] Clustering avec {n_clusters} clusters")

                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit(flat)

                labels = kmeans.labels_.reshape(H, W)
                unique_labels = np.unique(labels) #dire combien d'objets il arrive a reconnaître

                print(f"[RESULT] {name} | image {i}")
                print(f"Clusters demandés : {n_clusters}")
                print(f"Clusters détectés : {len(unique_labels)}")

                overlay = overlay_multiclass(img_gray, labels)

                save_path = os.path.join(
                    results_dir,
                    f"overlay_{i:03d}_{n_clusters}clusters.png"
                )

                plt.figure(figsize=(5, 5))
                plt.imshow(overlay)
                plt.title(f"{n_clusters} clusters")
                plt.axis("off")
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                plt.close()
