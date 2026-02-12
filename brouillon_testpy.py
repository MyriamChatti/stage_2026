import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from trainer import Trainer
import sys
import torch
import time
import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

import argparse
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

from data.nuclei_dataset import NucleiDataset



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
        img_denoised = F.median_pool2d(
            img,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )

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
    def __init__(self, kernel_size=7, save_dir="YourDATA/denoised/nlm"):
        self.kernel_size = kernel_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def denoise(self, img, idx):
        img_denoised = F.avg_pool2d(
            img,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )

        save_denoised_image(
            img_denoised,
            os.path.join(self.save_dir, f"img_{idx:03d}.png")
        )

        return img_denoised


# test

if __name__ == "__main__":

    tm = time.time()
    dt_str = datetime.datetime.fromtimestamp(tm).strftime("%Y-%m-%d_%H-%M-%S")

    check_manual_seed(opts.seed)

    test_loader = DataLoader(
        dataset=NucleiDataset(opts, "test"),
        batch_size=1,
        shuffle=False
    )

    trainer = Trainer(opts)
    trainer.load(opts.resume)
    trainer.evaluate(test_loader)


    os.makedirs("logs/results_75_epochs_lineargradient", exist_ok=True)

    denoiser = LinearGradientDenoiser(kernel_size=51)

    denoisers = {
    "linear": LinearGradientDenoiser(kernel_size=51),
    "gaussian": GaussianDenoiser(),
    "median": MedianDenoiser(),
    "rician": RicianDenoiser(),
    "nlm": NLMDenoiser()
}


    for name, denoiser in denoisers.items():
        results_dir = f"logs/results_75_epochs_{name}"
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n=== Test avec tout les filtres : {name.upper()} ===")

        for i, test_data in enumerate(test_loader):
            img = test_data["A"].to(trainer.device)

            img_denoised = denoiser.denoise(img, i)

            with torch.no_grad():
                prediction = trainer(img_denoised)

            save_prediction(
                prediction,
                os.path.join(results_dir, f"prediction_{i:03d}.png")
            )
