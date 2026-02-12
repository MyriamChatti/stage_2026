import math
import time
import datetime
import numpy as np
import scipy
from scipy import stats
from scipy.ndimage import uniform_filter

class ImageMetrics:

    def __init__(self, c1=6.5025, c2=58.5225):
        # initialise les constantes de stabilité (valeurs standards pour des images 8 bits)
        self.C1 = c1
        self.C2 = c2
        self.C3 = c2 / 2


    def compute_ssim(self, img1, img2):

        # calcul de l'indice SSIM entre deux images de même type.
        # les images doivent être des tableaux numpy de type float.
        if img1.shape != img2.shape:
            raise ValueError("Les images doivent avoir les mêmes dimensions.")

        
        # paramètres de la fenêtre locale (exemple : fenêtre carrée 11x11)
        win_size = 11
        
        # moyennes (mu)
        mu1 = uniform_filter(img1, win_size)
        mu2 = uniform_filter(img2, win_size)

        #c'est la moyenne de l'image 1 au carré. Elle est utilisée au dénominateur pour comparer la luminosité.
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # variances et covariance (sigma)
        sigma1_sq = uniform_filter(img1**2, win_size) - mu1_sq
        sigma2_sq = uniform_filter(img2**2, win_size) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, win_size) - mu1_mu2



        # calcul des composants
        # luminance * contraste (numérateur et dénominateur combinés pour l'efficacité du calcul(performance donc vitesse du calcul)
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)

        ssim_map = numerator / denominator
        #analyse la ressemblance partout dans l'image
        # luminance * contraste (numérateur et dénominateur combinés pour l'efficacité)
        # Le SSIM global est la moyenne de la carte locale
        return np.mean(ssim_map)
        # np.mean condense toute cette analyse en une seule note finale facile à lire






if __name__ == "__main__":
    dossier = r"C:\Users\myria\Desktop\Stage_M2\Stage_2026\code"
    nom_fichier = "L1_0001_D4.png"
    chemin_complet = os.path.join(dossier, nom_fichier)
    # il faudrait un chemin complet 2
    # instancier imageArray et appeler load image 2 fois pour chacune d'entre elle 
    # à définir np1 ET NP2 ET les initialiser
    instance_im = ImageMetrics()
    map_final = instance_im.compute_ssim(np1,np2)


 
