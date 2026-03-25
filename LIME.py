"""
============================================================
  lime_saliency.py
  Implémentation de LIME (Local Interpretable Model-agnostic
  Explanations) adaptée à l'imagerie médicale (IRM).

  LIME segmente l'image en superpixels, génère des versions
  perturbées en cachant certains superpixels, mesure l'impact
  sur la sortie du modèle, puis entraîne un modèle linéaire
  local pour attribuer une importance à chaque superpixel.

  Avantages pour l'IRM :
    - 100% model-agnostic (pas besoin de gradients)
    - Superpixels = régions anatomiquement cohérentes
    - Explication locale fidèle au modèle
    - Résultat très interprétable visuellement

  Installation :
      pip install lime scikit-image scikit-learn
                  matplotlib numpy pillow tensorflow

  Usage :
      python lime_saliency.py
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.ndimage import gaussian_filter

# chemin
input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/LIME_results"
os.makedirs(output_folder, exist_ok=True)



def load_image(path, img_size=(224, 224)):
    """Charge une image, la redimensionne et la normalise dans [0, 1]."""
    pil_img = Image.open(path).convert('RGB').resize(img_size)
    return np.array(pil_img, dtype=np.float32) / 255.0


def normalize_map(smap, percentile=99):
    """Normalise une carte dans [0, 1] en clippant au percentile."""
    smap = np.abs(smap)
    vmax = np.percentile(smap, percentile)
    if vmax == 0:
        return smap
    return np.clip(smap / vmax, 0, 1)


def overlay_heatmap(image, heatmap, alpha=0.55, colormap='inferno'):
    """Superpose une heatmap sur l'image originale."""
    cmap  = plt.get_cmap(colormap)
    hmap3 = cmap(heatmap)[..., :3]
    img01 = np.clip(image, 0, 1)
    return np.clip((1 - alpha) * img01 + alpha * hmap3, 0, 1)


# SEGMENTATION EN SUPERPIXELS

def segment_image(image, method='slic',
                  n_segments=80, compactness=10,
                  sigma=1.0):
    """Segmente l'image en superpixels.

    Args:
        image       : ndarray (H, W, 3) float32 [0, 1].
        method      : 'slic' | 'felzenszwalb' | 'quickshift'.
                      'slic' recommandé pour IRM.
        n_segments  : nombre approximatif de superpixels (SLIC).
        compactness : compacité des superpixels (SLIC).
                      Faible = suit les contours,
                      Élevé  = formes carrées régulières.
        sigma       : lissage avant segmentation.

    Returns:
        segments : ndarray (H, W) d'entiers — label de chaque pixel.
        n_labels : nombre de superpixels distincts.
    """
    from skimage.segmentation import slic, felzenszwalb, quickshift

    if method == 'slic':
        segments = slic(image,
                        n_segments=n_segments,
                        compactness=compactness,
                        sigma=sigma,
                        start_label=0,
                        channel_axis=-1)
    elif method == 'felzenszwalb':
        segments = felzenszwalb(image,
                                scale=100,
                                sigma=sigma,
                                min_size=50)
    elif method == 'quickshift':
        segments = quickshift(image,
                              kernel_size=3,
                              max_dist=6,
                              ratio=0.5)
    else:
        raise ValueError(f"Méthode inconnue : {method}. "
                         f"Choisir parmi : slic, felzenszwalb, quickshift")

    n_labels = segments.max() + 1
    return segments, n_labels


#  CLASSE PRINCIPALE LIME IMAGE

class LIMEImage:
    """LIME adapté à l'imagerie médicale.

    Paramètres
    ----------
    model_fn        : callable — prend une image (H, W, 3) float32
                      et retourne un score scalaire.
    n_samples       : nombre de perturbations générées (défaut 1000).
    seg_method      : algorithme de segmentation ('slic' recommandé).
    n_segments      : nombre de superpixels.
    compactness     : compacité superpixels SLIC.
    baseline        : valeur pour les superpixels cachés.
                      'mean' | 'blur' | 'black' | 'noise'
    positive_only   : si True, ne montre que les superpixels
                      qui augmentent la prédiction (supports).
    top_k           : nombre de superpixels les plus importants
                      à afficher (None = tous).
    """

    def __init__(self,
                 model_fn,
                 n_samples=1000,
                 seg_method='slic',
                 n_segments=80,
                 compactness=10,
                 sigma=1.0,
                 baseline='blur',
                 positive_only=True,
                 top_k=10):

        self.model_fn      = model_fn
        self.n_samples     = n_samples
        self.seg_method    = seg_method
        self.n_segments    = n_segments
        self.compactness   = compactness
        self.sigma         = sigma
        self.baseline      = baseline
        self.positive_only = positive_only
        self.top_k         = top_k

    #  Calcul de la baseline

    def _get_baseline(self, image):
        if self.baseline == 'mean':
            return np.ones_like(image) * image.mean(axis=(0, 1))
        elif self.baseline == 'blur':
            if image.ndim == 3:
                return np.stack([
                    gaussian_filter(image[:, :, c], sigma=15)
                    for c in range(image.shape[2])
                ], axis=-1)
            return gaussian_filter(image, sigma=15)
        elif self.baseline == 'noise':
            return np.random.uniform(0, 0.1, image.shape).astype(np.float32)
        else:  # 'black'
            return np.zeros_like(image)

    #  Application d'un masque de superpixels


    def _apply_mask(self, image, baseline, segments, active_labels):
        """Crée une image où seuls les superpixels actifs sont visibles."""
        masked = baseline.copy()
        for label in active_labels:
            masked[segments == label] = image[segments == label]
        return masked

    #  Explication LIME

    def explain(self, image, verbose=True):
        """Calcule l'explication LIME pour une image.

        Returns
        -------
        result : dict avec :
            'saliency_map'    : ndarray (H, W) normalisée [0, 1]
            'segments'        : ndarray (H, W) labels superpixels
            'coefs'           : ndarray (n_labels,) coefficients ridge
            'scores'          : ndarray (n_samples,) scores modèle
            'perturbations'   : ndarray (n_samples, n_labels) binaire
            'top_segments'    : liste des labels les plus importants
            'top_mask'        : masque binaire des top superpixels
            'n_labels'        : nombre de superpixels
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import normalize as sk_normalize

        H, W       = image.shape[:2]
        baseline   = self._get_baseline(image)

        # Segmentation
        if verbose:
            print(f"[LIME] Segmentation ({self.seg_method}, "
                  f"~{self.n_segments} superpixels)...")
        segments, n_labels = segment_image(
            image,
            method      = self.seg_method,
            n_segments  = self.n_segments,
            compactness = self.compactness,
            sigma       = self.sigma
        )
        if verbose:
            print(f"[LIME] {n_labels} superpixels créés.")

        #  Score de référence 
        score_original = float(self.model_fn(image))
        if verbose:
            print(f"[LIME] Score original : {score_original:.4f}")

        #  Génération des perturbations
        if verbose:
            print(f"[LIME] Génération de {self.n_samples} perturbations...")

        # Matrice binaire : 1 = superpixel visible, 0 = caché
        perturbations = np.random.randint(
            0, 2, size=(self.n_samples, n_labels)).astype(np.float32)

        # Toujours inclure l'image complète en première ligne
        perturbations[0] = np.ones(n_labels)

        scores = np.zeros(self.n_samples, dtype=np.float32)

        for i in range(self.n_samples):
            if verbose and (i % 100 == 0 or i == self.n_samples - 1):
                pct = int(100 * (i + 1) / self.n_samples)
                print(f"  Perturbation {i+1}/{self.n_samples} "
                      f"({pct}%) ...", end='\r', flush=True)

            active = np.where(perturbations[i] == 1)[0]
            masked = self._apply_mask(image, baseline, segments, active)
            scores[i] = float(self.model_fn(masked))

        if verbose:
            print(f"\n[LIME] Scores calculés. "
                  f"min={scores.min():.4f}, max={scores.max():.4f}")

        # Pondération par proximité (noyau cosinus) 
        # Les perturbations proches de l'image originale
        # ont plus de poids dans le modèle linéaire local
        ref        = np.ones(n_labels)
        distances  = np.sqrt(
            np.sum((perturbations - ref) ** 2, axis=1))
        kernel_width = 0.25 * np.sqrt(n_labels)
        weights    = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

        # Modèle linéaire local (Ridge)
        if verbose:
            print("[LIME] Entraînement du modèle linéaire Ridge...")

        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(perturbations,
                  scores,
                  sample_weight=weights)

        coefs = ridge.coef_   # (n_labels,)  importance de chaque superpixel

        if verbose:
            print(f"[LIME] Coefficients Ridge : "
                  f"min={coefs.min():.4f}, max={coefs.max():.4f}")

        # Construction de la carte de saillance 
        saliency_map = np.zeros((H, W), dtype=np.float32)
        for label in range(n_labels):
            saliency_map[segments == label] = coefs[label]

        # Filtrage positif ou absolu
        if self.positive_only:
            saliency_map = np.maximum(saliency_map, 0)

        # Normalisation
        saliency_map = normalize_map(saliency_map)

        #  Top K superpixels 
        if self.positive_only:
            sorted_labels = np.argsort(-coefs)
        else:
            sorted_labels = np.argsort(-np.abs(coefs))

        k         = self.top_k if self.top_k else n_labels
        top_segs  = sorted_labels[:k].tolist()

        top_mask  = np.zeros((H, W), dtype=bool)
        for label in top_segs:
            top_mask[segments == label] = True

        return {
            'saliency_map' : saliency_map,
            'segments'     : segments,
            'coefs'        : coefs,
            'scores'       : scores,
            'perturbations': perturbations,
            'top_segments' : top_segs,
            'top_mask'     : top_mask,
            'n_labels'     : n_labels,
            'score_original': score_original,
        }


#visualisation
def visualize_lime(image, result,
                   save_path=None,
                   title='LIME — Explication locale'):
    """Figure de synthèse LIME à 6 panneaux."""

    img01        = np.clip(image, 0, 1)
    saliency_map = result['saliency_map']
    segments     = result['segments']
    top_mask     = result['top_mask']
    coefs        = result['coefs']
    scores       = result['scores']

    # Heatmap colorée
    cmap_h   = plt.get_cmap('RdYlGn')
    # Centrage des coefs pour que 0 = jaune, + = vert, - = rouge
    coefs_n  = result['coefs'].copy()
    if coefs_n.max() != coefs_n.min():
        coefs_n = (coefs_n - coefs_n.min()) / \
                  (coefs_n.max() - coefs_n.min())
    coef_map = np.zeros(image.shape[:2], dtype=np.float32)
    for label in range(result['n_labels']):
        coef_map[segments == label] = coefs_n[label]

    heatmap  = cmap_h(coef_map)[..., :3]
    overlay  = np.clip(0.4 * img01 + 0.6 * heatmap, 0, 1)

    # Contours de superpixels
    from skimage.segmentation import mark_boundaries
    img_boundaries = mark_boundaries(img01, segments,
                                     color=(0.2, 0.8, 0.2),
                                     mode='thick')

    # Top superpixels sur image originale
    img_top = img01.copy() * 0.35   # assombrir le fond
    img_top[top_mask] = img01[top_mask]

    # Contour des top superpixels
    from scipy.ndimage import binary_erosion
    eroded   = binary_erosion(top_mask, iterations=2)
    contour  = top_mask & ~eroded
    img_contour = img01.copy()
    img_contour[contour] = [0, 1, 0]

    # Histogramme des scores
    fig = plt.figure(figsize=(26, 5), facecolor='black')
    gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.06)

    panels = [
        (img01,          'Image originale',        None),
        (img_boundaries, 'Superpixels',             None),
        (heatmap,        'Importance (RdYlGn)',     None),
        (overlay,        'Overlay',                 None),
        (img_top,        f'Top {result["top_segments"].__len__()} '
                         f'superpixels',            None),
        (img_contour,    'Contour top superpixels', None),
    ]

    for i, (ax_img, label, cmap_name) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(ax_img, cmap=cmap_name)
        ax.set_title(label, color='white', fontsize=9, fontweight='bold')
        ax.axis('off')
        ax.set_facecolor('black')

    score_info = (f"Score original : {result['score_original']:.3f}  |  "
                  f"Scores perturbés : μ={scores.mean():.3f}  "
                  f"σ={scores.std():.3f}")
    fig.suptitle(f"{title}\n{score_info}",
                 color='white', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[LIME] Figure sauvegardée → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_lime_coefs(result, top_n=20, save_path=None):
    """Barplot des top N coefficients Ridge (importance par superpixel)."""
    coefs  = result['coefs']
    idx    = np.argsort(-np.abs(coefs))[:top_n]
    vals   = coefs[idx]
    labels = [f"SP {i}" for i in idx]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor='none')
    ax.axvline(0, color='white', linewidth=0.8)
    ax.set_xlabel('Coefficient Ridge (importance)', color='white')
    ax.set_title(f'Top {top_n} superpixels LIME',
                 color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a1a')
        print(f"[LIME] Barplot sauvegardé → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_score_distribution(scores, save_path=None):
    """Histogramme des scores de perturbation."""
    fig, ax = plt.subplots(figsize=(7, 3), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.hist(scores, bins=50, color='#3498db', edgecolor='none', alpha=0.85)
    ax.axvline(scores.mean(), color='white', linestyle='--',
               linewidth=1.5, label=f'μ = {scores.mean():.3f}')
    ax.set_xlabel('Score modèle', color='white')
    ax.set_ylabel('Nombre de perturbations', color='white')
    ax.set_title('Distribution des scores (perturbations LIME)',
                 color='white', fontweight='bold')
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


# MAIN — BOUCLE SUR VOS IMAGES IRM

if __name__ == '__main__':

    import tensorflow as tf

    input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
    output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/lime_results"
    os.makedirs(output_folder, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

   
    IMG_SIZE  = (224, 224)
    CLASS_IDX = 0 

    print("[INFO] Chargement du modèle...")
    base_model = tf.keras.applications.ResNet50(weights='imagenet')

    def model_fn(image):
        """Retourne le score softmax pour CLASS_IDX.
        
        → Remplacez cette fonction par votre propre modèle.
          Elle doit prendre (H, W, 3) float32 [0,1]
          et retourner un scalaire float.
        """
        x = np.expand_dims(image, axis=0).astype(np.float32)
        x = x * 255.0
        x = tf.keras.applications.resnet50.preprocess_input(x)
        preds = base_model(x, training=False).numpy()
        return float(preds[0, CLASS_IDX])




    # ============================================================
    #  INITIALISATION LIME
    #
    #  Paramètres recommandés pour IRM :
    #    n_samples=2000, n_segments=100 → précision maximale
    #    n_samples=500,  n_segments=50  → version rapide



    lime_explainer = LIMEImage(
        model_fn      = model_fn,
        n_samples     = 1000,    # nombre de perturbations
        seg_method    = 'slic',  # slic = meilleur pour IRM
        n_segments    = 80,      # ~80 superpixels
        compactness   = 5,       # faible = suit les contours anatomiques
        sigma         = 1.0,     # lissage avant segmentation
        baseline      = 'blur',  # blur = plus naturel que le noir pour IRM
        positive_only = True,    # ne montrer que les zones qui aident
        top_k         = 10,      # top 10 superpixels les plus importants
    )




    #  BOUCLE SUR TOUTES LES IMAGES
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

        #  Chargement 
        try:
            image = load_image(img_path, IMG_SIZE)
        except Exception as e:
            print(f"  [ERREUR chargement] {e} — image ignorée.")
            continue

        # Calcul LIME 
        try:
            result = lime_explainer.explain(image, verbose=True)
        except Exception as e:
            print(f"  [ERREUR LIME] {e} — image ignorée.")
            continue

        # Sauvegarde figure principale 
        fig_path = os.path.join(output_folder,
                                f"{img_name}_lime.png")
        visualize_lime(image, result,
                       save_path=fig_path,
                       title=f"LIME — {img_name}")

        #  Barplot des coefficients
        bar_path = os.path.join(output_folder,
                                f"{img_name}_lime_coefs.png")
        plot_lime_coefs(result, top_n=20, save_path=bar_path)

        #  Distribution des scores 
        dist_path = os.path.join(output_folder,
                                 f"{img_name}_lime_scores.png")
        plot_score_distribution(result['scores'], save_path=dist_path)

        # Sauvegarde numpy 
        npy_dir = os.path.join(output_folder, img_name)
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, 'saliency_map.npy'),
                result['saliency_map'])
        np.save(os.path.join(npy_dir, 'segments.npy'),
                result['segments'])
        np.save(os.path.join(npy_dir, 'coefs.npy'),
                result['coefs'])
        np.save(os.path.join(npy_dir, 'scores.npy'),
                result['scores'])
        np.save(os.path.join(npy_dir, 'top_mask.npy'),
                result['top_mask'])

        print(f"  [OK] → {fig_path}")
        print(f"  Score original : {result['score_original']:.4f}")
        print(f"  Superpixels importants : {result['top_segments']}")

    print(f"\n[TERMINÉ] Toutes les images ont été traitées.")
    print(f"Résultats dans : {output_folder}")