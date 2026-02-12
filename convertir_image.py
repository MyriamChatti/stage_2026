import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Configuration du chemin
dossier = r"C:\Users\myria\Desktop\Stage_M2\Stage_2026\code"
nom_fichier = "L1_0001_D4.png"
chemin_complet = os.path.join(dossier, nom_fichier)


class imageArray:
    "classe pour charger une image et la convertir en tableau NumPy"

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.array = None
    

    def load_image(self):
        print(chemin_complet)
        print(os.path.exists(chemin_complet))

        self.image = Image.open(self.image_path)
        return self.image
        "chargement de l'image"
        
    def convert_rgb_gray(self):
        self.convert = self.image.convert("L")
        "L pour niveau de gris"
        
        
        
    def convert_to_array(self):
        "convertit l'image en tableau NumPy"
        self.array = np.array(self.convert)
        return self.array

    def get_array_info(self):
        return {
            "shape": self.array.shape,
            "dtype": self.array.dtype,
            "min_value": self.array.min(),
            "max_value": self.array.max()
        }
    def show_image(self):
        plt.imshow(self.array, cmap= 'gray')
        plt.show()
        
"phase d’exécution (pipeline) : on crée un objet, on appelle ses méthodes, puis on récupère les résultats."

converter = imageArray(chemin_complet)
"création de l'objet"
converter.load_image()
converter.convert_rgb_gray()
"lit le fichier image, créer objet image (PIL) puis le stocke dans le self.image"
converter.convert_to_array()
"convertie l'image en tableau numpy"

converter.show_image()
