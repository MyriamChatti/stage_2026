"""
============================================================
  drise_saliency.py
  Implémentation de DRISE (Detection Randomized Input
  Sampling for Explanation) adaptée à l'imagerie médicale
  (IRM, segmentation).

  DRISE génère des masques aléatoires, les applique sur
  l'image, mesure l'impact sur la sortie du modèle, puis
  pondère les masques par leur impact pour obtenir une
  carte de saillance.

  Avantages pour l'IRM :
    - 100% model-agnostic (pas besoin de gradients)
    - Pas de dépendance à une couche conv spécifique
    - Résolution spatiale bien meilleure que GradCAM
    - Robuste au bruit

  Usage :
      python drise_saliency.py
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image

# chemin
input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/attention_MAPS"
os.makedirs(output_folder, exist_ok=True)



def generate_random_masks(image_shape, n_masks=1000,
                           grid_size=(8, 8), proba=0.5):
    """Génère N masques binaires aléatoires de faible résolution,
    puis les upscale à la taille de l'image (effet de lissage naturel).

    Args:
        image_shape : tuple (H, W) — taille de l'image.
        n_masks     : nombre de masques à générer.
        grid_size   : taille de la grille basse résolution (lignes, cols).
                      Plus petit = régions plus grandes.
                      Plus grand = régions plus fines.
        proba       : probabilité qu'une cellule soit ACTIVE (visible).

    Returns:
        masks : ndarray (n_masks, H, W) de float32 dans [0, 1].
    """
    H, W    = image_shape
    gh, gw  = grid_size
    masks   = np.zeros((n_masks, H, W), dtype=np.float32)

    for i in range(n_masks):
        # Masque basse résolution
        low_res = (np.random.rand(gh, gw) < proba).astype(np.float32)

        # Upscale bilinéaire vers la taille de l'image
        zh = H / gh
        zw = W / gw
        upscaled = zoom(low_res, (zh, zw), order=1)

        # Ajustement exact de la taille (zoom peut déborder d'1 pixel)
        upscaled = upscaled[:H, :W]
        if upscaled.shape != (H, W):
            pad_h = H - upscaled.shape[0]
            pad_w = W - upscaled.shape[1]
            upscaled = np.pad(upscaled,
                              ((0, pad_h), (0, pad_w)),
                              mode='edge')

        masks[i] = upscaled

    return masks



class DRISE:
    """Implémentation de DRISE adaptée à l'imagerie médicale.

    Référence originale :
    Petsiuk et al., "RISE: Randomized Input Sampling for
    Explanation of Black-box Models", BMVC 2018.
    DRISE = extension aux modèles de détection/segmentation.

    Paramètres
    ----------
    model_fn   : callable — fonction qui prend une image
                 (H, W, C) float32 [0,1] et retourne un
                 score scalaire (confiance, probabilité, etc.)
    n_masks    : nombre de masques aléatoires (défaut 1000,
                 augmenter pour plus de précision)
    grid_size  : tuple (gh, gw) résolution de la grille
    proba      : probabilité d'activation par cellule
    batch_size : nombre de masques traités en parallèle
    baseline   : valeur de remplacement pour les zones masquées
                 (0 = noir, 0.5 = gris, ou moyenne de l'image)
    """

    def __init__(self,
                 model_fn,
                 n_masks=1000,
                 grid_size=(8, 8),
                 proba=0.5,
                 batch_size=50,
                 baseline='mean'):

        self.model_fn   = model_fn
        self.n_masks    = n_masks
        self.grid_size  = grid_size
        self.proba      = proba
        self.batch_size = batch_size
        self.baseline   = baseline

    def _get_baseline(self, image):
        """Calcule la valeur de baseline selon le paramètre choisi."""
        if self.baseline == 'mean':
            return np.mean(image, axis=(0, 1), keepdims=True)
        elif self.baseline == 'blur':
            if image.ndim == 3:
                return np.stack([
                    gaussian_filter(image[:, :, c], sigma=10)
                    for c in range(image.shape[2])
                ], axis=-1)
            else:
                return gaussian_filter(image, sigma=10)
        elif isinstance(self.baseline, (int, float)):
            return np.full_like(image, self.baseline)
        else:
            return np.zeros_like(image)

    def explain(self, image, verbose=True):
        """Calcule la carte de saillance DRISE pour une image.

        Args:
            image   : ndarray (H, W) ou (H, W, C), float32 [0,1].
            verbose : affiche la progression.

        Returns:
            saliency_map : ndarray (H, W) normalisée dans [0, 1].
            scores_all   : ndarray (n_masks,) — scores bruts du modèle.
        """
        H, W          = image.shape[:2]
        baseline_val  = self._get_baseline(image)

        # Génération de tous les masques
        if verbose:
            print(f"[DRISE] Génération de {self.n_masks} masques "
                  f"(grille {self.grid_size}, p={self.proba})...")
        masks = generate_random_masks(
            (H, W), self.n_masks, self.grid_size, self.proba)

        # Calcul des scores pour chaque masque
        scores    = np.zeros(self.n_masks, dtype=np.float32)
        n_batches = int(np.ceil(self.n_masks / self.batch_size))

        for b in range(n_batches):
            start = b * self.batch_size
            end   = min(start + self.batch_size, self.n_masks)

            if verbose and (b % 5 == 0 or b == n_batches - 1):
                pct = int(100 * end / self.n_masks)
                print(f"  Batch {b+1}/{n_batches} "
                      f"({pct}%) ...", end='\r', flush=True)

            for i in range(start, end):
                m = masks[i]                         # (H, W)

                # Application du masque sur l'image
                if image.ndim == 3:
                    masked = image * m[..., np.newaxis] + \
                             baseline_val * (1 - m[..., np.newaxis])
                else:
                    masked = image * m + baseline_val * (1 - m)

                scores[i] = float(self.model_fn(masked))

        if verbose:
            print(f"\n[DRISE] Scores calculés. "
                  f"min={scores.min():.4f}, "
                  f"max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}")

        # --------------------------------------------------------
        # Calcul de la carte de saillance :
        # saliency[x,y] = somme(score_i * mask_i[x,y]) /
        #                 somme(mask_i[x,y])
        # = espérance du score sachant que le pixel (x,y) est visible
        # --------------------------------------------------------
        scores_pos   = np.maximum(scores, 0)          # ne garder que positif
        weights      = scores_pos - scores_pos.mean() # centrage

        saliency = np.zeros((H, W), dtype=np.float64)
        normaliz = np.zeros((H, W), dtype=np.float64)

        for i in range(self.n_masks):
            saliency += weights[i] * masks[i]
            normaliz += masks[i]

        # Division sécurisée
        normaliz  = np.maximum(normaliz, 1e-8)
        saliency /= normaliz

        # Normalisation dans [0, 1]
        saliency -= saliency.min()
        if saliency.max() > 0:
            saliency /= saliency.max()

        return saliency.astype(np.float32), scores






def visualize_drise(image, saliency_map,
                    scores=None,
                    save_path=None,
                    title='DRISE — Carte de saillance'):
    """Génère une figure de synthèse DRISE.

    Colonnes : image originale | heatmap | overlay | masque binaire | contour
    """
    # Préparation de l'image affichable
    img01 = np.clip(image, 0, 1)
    if img01.ndim == 2:
        img01 = np.stack([img01] * 3, axis=-1)

    # Heatmap colorée
    cmap    = plt.get_cmap('inferno')
    heatmap = cmap(saliency_map)[..., :3]

    # Overlay
    overlay = np.clip(0.45 * img01 + 0.55 * heatmap, 0, 1)

    # Masque binaire (seuil Otsu simplifié = percentile 70)
    threshold    = np.percentile(saliency_map, 70)
    binary_mask  = (saliency_map >= threshold).astype(np.uint8) * 255

    # Contour sur image originale
    from scipy.ndimage import binary_erosion
    mask_bool   = saliency_map >= threshold
    eroded      = binary_erosion(mask_bool, iterations=2)
    contour_map = mask_bool & ~eroded

    img_contour = img01.copy()
    img_contour[contour_map] = [0, 1, 0]   # contour vert

    # Figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor('black')

    panels = [
        (img01,       'Image originale',  None),
        (heatmap,     'Heatmap DRISE',    None),
        (overlay,     'Overlay',          None),
        (binary_mask, 'Masque (p70)',     'gray'),
        (img_contour, 'Contour',          None),
    ]

    for ax, (img, label, cmap_name) in zip(axes, panels):
        ax.imshow(img, cmap=cmap_name)
        ax.set_title(label, color='white', fontsize=10, fontweight='bold')
        ax.axis('off')
        ax.set_facecolor('black')

    # Distribution des scores (mini-histogramme dans le titre)
    if scores is not None:
        score_info = (f"scores : μ={scores.mean():.3f}  "
                      f"σ={scores.std():.3f}  "
                      f"max={scores.max():.3f}")
        fig.suptitle(f"{title}\n{score_info}",
                     color='white', fontsize=11, fontweight='bold')
    else:
        fig.suptitle(title, color='white', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[DRISE] Figure sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_score_distribution(scores, save_path=None):
    """Histogramme de la distribution des scores DRISE."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    ax.hist(scores, bins=50, color='#e07b39', edgecolor='none', alpha=0.85)
    ax.axvline(scores.mean(), color='white', linestyle='--',
               linewidth=1.5, label=f'μ = {scores.mean():.3f}')
    ax.set_xlabel('Score modèle', color='white')
    ax.set_ylabel('Nombre de masques', color='white')
    ax.set_title('Distribution des scores DRISE', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#333', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a1a')
    else:
        plt.show()
    plt.close(fig)



if __name__ == '__main__':

    import tensorflow as tf

    input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
    output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/drise_results"
    os.makedirs(output_folder, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    IMG_SIZE  = (224, 224)
    CLASS_IDX = 0           # classe à expliquer

    print("[INFO] Chargement du modèle...")
    base_model = tf.keras.applications.ResNet50(weights='imagenet')

    # ============================================================
    #  FONCTION MODÈLE pour DRISE
    #  Doit prendre une image (H, W, C) float32 [0,1]
    #  et retourner un score scalaire.

    def model_fn(image):
        """Retourne le score softmax pour la classe CLASS_IDX."""
        # Préparation batch (1, H, W, C)
        x = np.expand_dims(image, axis=0).astype(np.float32)
        x = x * 255.0   # ResNet50 attend [0, 255]

        # Prétraitement ResNet50
        x = tf.keras.applications.resnet50.preprocess_input(x)

        # Prédiction
        preds = base_model(x, training=False).numpy()

        # Score = probabilité softmax de la classe cible
        score = float(preds[0, CLASS_IDX])
        return score

    # ============================================================
    #  INITIALISATION DRISE
    #
    #  Paramètres recommandés pour IRM :
    #    n_masks=2000, grid_size=(14,14) → précision fine
    #    n_masks=500,  grid_size=(8,8)   → version rapide
    # ============================================================
    drise = DRISE(
        model_fn   = model_fn,
        n_masks    = 1000,       # augmenter pour plus de précision
        grid_size  = (14, 14),   # grille fine → bonne résolution spatiale
        proba      = 0.5,
        batch_size = 50,
        baseline   = 'blur',     # blur = meilleur pour IRM (pas de noir brutal)
    )





    image_paths = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])

    if not image_paths:
        print(f"[ERREUR] Aucune image trouvée dans : {input_folder}")
    else:
        print(f"\n[INFO] {len(image_paths)} image(s) trouvée(s) dans :")
        print(f"       {input_folder}")
        print(f"[INFO] Résultats sauvegardés dans :")
        print(f"       {output_folder}\n")

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'='*60}")
        print(f"[IMAGE] {img_name}")
        print(f"{'='*60}")

        # Chargement 
        try:
            pil_img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            image   = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"  [ERREUR chargement] {e} — image ignorée.")
            continue

        #  Calcul DRISE 
        try:
            saliency_map, scores = drise.explain(image, verbose=True)
        except Exception as e:
            print(f"  [ERREUR DRISE] {e} — image ignorée.")
            continue

        # Sauvegarde figure principale 
        fig_path = os.path.join(
            output_folder, f"{img_name}_drise.png")
        visualize_drise(
            image, saliency_map, scores=scores,
            save_path=fig_path,
            title=f"DRISE — {img_name}"
        )

        #  Sauvegarde distribution des scores 
        dist_path = os.path.join(
            output_folder, f"{img_name}_scores_dist.png")
        plot_score_distribution(scores, save_path=dist_path)

        
        np.save(
            os.path.join(output_folder, f"{img_name}_drise.npy"),
            saliency_map
        )

        print(f"  [OK] → {fig_path}")

    print(f"\n[TERMINÉ] Toutes les images ont été traitées.")
    print(f"Résultats dans : {output_folder}")