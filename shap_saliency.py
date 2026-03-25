"""
============================================================
  shap_saliency.py
  Implémentation complète de SHAP (SHapley Additive
  exPlanations) adaptée à l'imagerie médicale (IRM).

  Trois variantes implémentées :
    1. DeepSHAP     — rapide, basé sur les gradients
    2. GradientSHAP — DeepSHAP + bruit stochastique
    3. PartitionSHAP — model-agnostic, par superpixels
                       (le plus précis pour l'IRM)

  Référence :
  Lundberg & Lee, "A Unified Approach to Interpreting
  Model Predictions", NeurIPS 2017.

  Avantages pour l'IRM :
    - Fondement théorique solide (théorie des jeux)
    - PartitionSHAP respecte la structure spatiale
    - Valeurs SHAP signées : positif = contribue à la classe
                             négatif = inhibe la classe
  Usage :
      python shap_saliency.py
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image
import tensorflow as tf






# chemin
input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/shap"
os.makedirs(output_folder, exist_ok=True)


# ============================================================
#  VÉRIFICATION INSTALLATION SHAP
try:
    import shap
    print(f"[INFO] shap version : {shap.__version__}")
except ImportError:
    raise ImportError(
        "\n[ERREUR] Le package 'shap' n'est pas installé.\n"
        "Installez-le avec : pip install shap\n"
    )



def normalize_map(smap, percentile=99):
    """Normalise une carte dans [0, 1] avec clip au percentile."""
    smap_abs = np.abs(smap)
    vmax = np.percentile(smap_abs, percentile)
    if vmax == 0:
        return smap_abs
    return np.clip(smap_abs / vmax, 0, 1)


def normalize_signed(smap, percentile=99):
    """Normalise en conservant le signe (valeurs dans [-1, 1]).
    Utile pour visualiser contributions positives ET négatives.
    """
    vmax = np.percentile(np.abs(smap), percentile)
    if vmax == 0:
        return smap
    return np.clip(smap / vmax, -1, 1)


def smooth(smap, sigma=1.5):
    """Lissage gaussien."""
    return gaussian_filter(smap, sigma=sigma)


def resize_to(smap, target_shape):
    """Redimensionne une carte 2D vers target_shape (H, W)."""
    zh = target_shape[0] / smap.shape[0]
    zw = target_shape[1] / smap.shape[1]
    return zoom(smap, (zh, zw), order=1)


def preprocess_resnet(image):
    """Prétraitement ResNet50 : [0,1] → centré ImageNet.
    Retourne toujours un numpy array.
    """
    x = image[np.newaxis] * 255.0
    result = tf.keras.applications.resnet50.preprocess_input(x)
    # Compatibilité TF eager / numpy
    if hasattr(result, 'numpy'):
        return result.numpy()
    return np.array(result)


# ============================================================
#   DEEP SHAP

class DeepSHAPExplainer:
    """DeepSHAP — rapide, utilise les gradients du modèle.

    Nécessite un ensemble de background images (baseline).
    Recommandé : 20-100 images représentatives du dataset.
    Si non disponibles, on utilise des images noires ou bruitées.

    Paramètres
    ----------
    model       : modèle Keras.
    background  : ndarray (N, H, W, C) — images de référence.
    class_idx   : indice de la classe à expliquer.
    """

    def __init__(self, model, background, class_idx=0):
        self.model     = model
        self.class_idx = class_idx

        # Prétraitement du background
        bg_pp = np.concatenate([
            preprocess_resnet(bg)
            for bg in background
        ], axis=0)

        print(f"[DeepSHAP] Initialisation avec "
              f"{len(background)} images background...")

        # Sous-modèle qui retourne uniquement la classe cible
        inp = model.input
        out = model.output[:, class_idx:class_idx+1]
        self.target_model = tf.keras.Model(inputs=inp, outputs=out)

        # Explainer SHAP
        self.explainer = shap.DeepExplainer(
            self.target_model, bg_pp)

        print("[DeepSHAP] Prêt.")

    def explain(self, image, smooth_sigma=1.5, verbose=True):
        """Calcule les valeurs SHAP pour une image.

        Returns:
            shap_map   : ndarray (H, W) normalisée [0, 1].
            shap_signed: ndarray (H, W) signée [-1, 1].
            shap_raw   : ndarray (H, W, C) valeurs brutes.
        """
        if verbose:
            print("[DeepSHAP] Calcul des valeurs SHAP...")

        x_pp  = preprocess_resnet(image)
        shap_values = self.explainer.shap_values(x_pp)

        # shap_values : liste ou array (1, H, W, C)
        if isinstance(shap_values, list):
            sv = shap_values[0][0]   # (H, W, C)
        else:
            sv = shap_values[0]      # (H, W, C)

        # Agrégation sur les canaux couleur
        shap_raw    = sv                           # (H, W, C)
        shap_2d     = sv.sum(axis=-1)             # (H, W)

        shap_signed = normalize_signed(shap_2d)
        shap_map    = normalize_map(shap_2d)

        if smooth_sigma > 0:
            shap_map    = smooth(shap_map,    sigma=smooth_sigma)
            shap_signed = smooth(shap_signed, sigma=smooth_sigma)

        if verbose:
            print(f"[DeepSHAP] Done. "
                  f"min={shap_2d.min():.4f}, "
                  f"max={shap_2d.max():.4f}")

        return shap_map.astype(np.float32), \
               shap_signed.astype(np.float32), \
               shap_raw


# ============================================================
#  GRADIENT SHAP


class GradientSHAPExplainer:
    """GradientSHAP = DeepSHAP + perturbations stochastiques.

    Plus robuste que DeepSHAP pur car moyenne sur plusieurs
    baselines bruitées → réduit la variance des estimations.

    Paramètres
    ----------
    model       : modèle Keras.
    background  : ndarray (N, H, W, C) — images de référence.
    class_idx   : indice de la classe à expliquer.
    n_samples   : nombre de perturbations stochastiques.
    """

    def __init__(self, model, background,
                 class_idx=0, n_samples=50):
        self.model     = model
        self.class_idx = class_idx
        self.n_samples = n_samples

        bg_pp = np.concatenate([
            preprocess_resnet(bg)
            for bg in background
        ], axis=0)

        print(f"[GradientSHAP] Initialisation avec "
              f"{len(background)} background, "
              f"{n_samples} samples...")

        inp = model.input
        out = model.output[:, class_idx:class_idx+1]
        self.target_model = tf.keras.Model(inputs=inp, outputs=out)

        self.explainer = shap.GradientExplainer(
            self.target_model, bg_pp)

        print("[GradientSHAP] Prêt.")

    def explain(self, image, smooth_sigma=1.5, verbose=True):
        """Calcule les valeurs GradientSHAP pour une image."""
        if verbose:
            print(f"[GradientSHAP] Calcul "
                  f"({self.n_samples} samples)...")

        x_pp = preprocess_resnet(image)
        shap_values = self.explainer.shap_values(
            x_pp, nsamples=self.n_samples)

        if isinstance(shap_values, list):
            sv = shap_values[0][0]
        else:
            sv = shap_values[0]

        shap_raw    = sv
        shap_2d     = sv.sum(axis=-1)
        shap_signed = normalize_signed(shap_2d)
        shap_map    = normalize_map(shap_2d)

        if smooth_sigma > 0:
            shap_map    = smooth(shap_map,    sigma=smooth_sigma)
            shap_signed = smooth(shap_signed, sigma=smooth_sigma)

        if verbose:
            print(f"[GradientSHAP] Done. "
                  f"min={shap_2d.min():.4f}, "
                  f"max={shap_2d.max():.4f}")

        return shap_map.astype(np.float32), \
               shap_signed.astype(np.float32), \
               shap_raw


# ============================================================
# PARTITION SHAP (model-agnostic, superpixels)

class PartitionSHAPExplainer:
    """PartitionSHAP — le plus adapté à l'IRM.

    Model-agnostic : ne nécessite pas de gradients.
    Fonctionne par superpixels : regroupe les pixels en
    régions cohérentes (respecte l'anatomie), puis calcule
    la contribution de chaque région à la prédiction.

    C'est la méthode la plus lente mais la plus précise
    pour l'imagerie médicale.

    Paramètres
    ----------
    model_fn     : callable — prend (H, W, C) float32 [0,1],
                   retourne un score scalaire.
    class_idx    : indice de la classe à expliquer.
    max_evals    : budget de calcul (500 = rapide, 2000 = précis).
    masker_type  : 'blur' | 'inpaint' | 'black'
                   blur    = zones masquées floutées (recommandé IRM)
                   inpaint = reconstruction contextuelle
                   black   = zones masquées à 0
    """

    def __init__(self, model_fn, class_idx=0,
                 max_evals=1000, masker_type='blur'):
        self.model_fn    = model_fn
        self.class_idx   = class_idx
        self.max_evals   = max_evals
        self.masker_type = masker_type
        print(f"[PartitionSHAP] max_evals={max_evals}, "
              f"masker='{masker_type}'")

    def _build_masker(self, image):
        """Construit le masker SHAP selon le type choisi."""
        if self.masker_type == 'blur':
            return shap.maskers.Image("blur(128,128)", image.shape)
        elif self.masker_type == 'inpaint':
            return shap.maskers.Image("inpaint_telea", image.shape)
        else:
            return shap.maskers.Image(
                np.zeros_like(image), image.shape)

    def explain(self, image, smooth_sigma=1.5, verbose=True):
        """Calcule les valeurs PartitionSHAP pour une image.

        Returns:
            shap_map   : ndarray (H, W) normalisée [0, 1].
            shap_signed: ndarray (H, W) signée [-1, 1].
            shap_raw   : ndarray (H, W, C) valeurs brutes.
        """
        H, W = image.shape[:2]

        if verbose:
            print(f"[PartitionSHAP] Calcul "
                  f"(max_evals={self.max_evals})...")

        # Masker
        masker = self._build_masker(image)

        # Fonction de prédiction compatible SHAP
        def predict_fn(images):
            """Prend (N, H, W, C) retourne (N, 1)."""
            scores = []
            for img in images:
                img_clipped = np.clip(img, 0, 1)
                s = self.model_fn(img_clipped)
                scores.append([float(s)])
            return np.array(scores)

        # Explainer PartitionSHAP
        explainer = shap.Explainer(
            predict_fn,
            masker,
            algorithm="partition"
        )

        # Calcul (image doit être (1, H, W, C))
        img_batch = image[np.newaxis]
        shap_values = explainer(
            img_batch,
            max_evals   = self.max_evals,
            batch_size  = 50,
            outputs     = shap.Explanation.argsort.flip[:1]
        )

        # Extraction
        # shap_values.values : (1, H, W, C, 1)
        sv = shap_values.values[0, ..., 0]    # (H, W, C)

        shap_raw    = sv
        shap_2d     = sv.sum(axis=-1)         # (H, W)
        shap_signed = normalize_signed(shap_2d)
        shap_map    = normalize_map(shap_2d)

        if smooth_sigma > 0:
            shap_map    = smooth(shap_map,    sigma=smooth_sigma)
            shap_signed = smooth(shap_signed, sigma=smooth_sigma)

        if verbose:
            print(f"[PartitionSHAP] Done. "
                  f"min={shap_2d.min():.4f}, "
                  f"max={shap_2d.max():.4f}")

        return shap_map.astype(np.float32), \
               shap_signed.astype(np.float32), \
               shap_raw




def visualize_shap(image, shap_map, shap_signed,
                   method_name='SHAP',
                   save_path=None,
                   title=None):
    """Figure complète SHAP :
    image | heatmap abs | heatmap signée | overlay | contour
    + colorbar rouge/bleu pour les contributions signées.
    """
    img01 = np.clip(image, 0, 1)
    if img01.ndim == 2:
        img01 = np.stack([img01] * 3, axis=-1)

    title = title or f"{method_name} — Carte de saillance"

    # Heatmap absolue
    cmap_abs    = plt.get_cmap('inferno')
    heatmap_abs = cmap_abs(shap_map)[..., :3]

    # Heatmap signée (rouge = positif, bleu = négatif)
    cmap_signed    = plt.get_cmap('RdBu_r')
    # Normalisation de [-1,1] vers [0,1] pour colormap
    shap_norm      = (shap_signed + 1) / 2
    heatmap_signed = cmap_signed(shap_norm)[..., :3]

    # Overlay
    overlay = np.clip(0.45 * img01 + 0.55 * heatmap_abs, 0, 1)

    # Masque binaire
    threshold   = np.percentile(shap_map, 70)
    mask_bool   = shap_map >= threshold

    # Contour
    from scipy.ndimage import binary_erosion
    eroded      = binary_erosion(mask_bool, iterations=2)
    contour_map = mask_bool & ~eroded
    img_contour = img01.copy()
    img_contour[contour_map] = [0, 1, 0]

    # Figure
    fig = plt.figure(figsize=(26, 5))
    fig.patch.set_facecolor('black')
    gs  = gridspec.GridSpec(1, 6, figure=fig,
                            wspace=0.05,
                            width_ratios=[1, 1, 1, 1, 1, 0.05])

    panels = [
        (img01,          'Image originale',   None,       0),
        (heatmap_abs,    f'{method_name}\n(abs)',
                                              None,       1),
        (heatmap_signed, f'{method_name}\n(+ rouge / - bleu)',
                                              None,       2),
        (overlay,        'Overlay',           None,       3),
        (img_contour,    'Contour (p70)',     None,       4),
    ]

    ax_list = []
    for img_p, label, _, col in panels:
        ax = fig.add_subplot(gs[col])
        ax.imshow(img_p)
        ax.set_title(label, color='white',
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        ax_list.append(ax)

    # Colorbar pour la carte signée
    ax_cb = fig.add_subplot(gs[5])
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap('RdBu_r'),
        norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cb)
    cb.set_label('Contribution SHAP', color='white', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    fig.suptitle(title, color='white', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[SHAP] Figure sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_shap_comparison(image, results_dict,
                               save_path=None):
    """Compare plusieurs méthodes SHAP sur une même figure.

    Args:
        results_dict : {nom_méthode: (shap_map, shap_signed)}
    """
    img01 = np.clip(image, 0, 1)
    if img01.ndim == 2:
        img01 = np.stack([img01] * 3, axis=-1)

    n_methods = len(results_dict)
    n_cols    = n_methods * 2 + 1   # orig + (abs + signed) par méthode
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.patch.set_facecolor('black')

    if n_cols == 1:
        axes = [axes]

    col = 0
    axes[col].imshow(img01)
    axes[col].set_title('Image originale', color='white',
                         fontsize=9, fontweight='bold')
    axes[col].axis('off')
    col += 1

    cmap_abs    = plt.get_cmap('inferno')
    cmap_signed = plt.get_cmap('RdBu_r')

    for name, (smap, ssigned) in results_dict.items():
        # Carte absolue
        axes[col].imshow(cmap_abs(smap)[..., :3])
        axes[col].set_title(f'{name}\n(abs)',
                             color='#f5a623', fontsize=8)
        axes[col].axis('off')
        col += 1

        # Carte signée
        shap_norm = (ssigned + 1) / 2
        axes[col].imshow(cmap_signed(shap_norm)[..., :3])
        axes[col].set_title(f'{name}\n(+/−)',
                             color='#7ed321', fontsize=8)
        axes[col].axis('off')
        col += 1

    fig.suptitle('Comparaison méthodes SHAP',
                 color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[SHAP] Comparaison sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================
#  GÉNÉRATION DU BACKGROUND

def build_background(input_folder, n=20,
                      img_size=(224, 224),
                      img_extensions=None):
    """Construit un ensemble de background à partir de
    quelques images du dossier d'entrée.

    Si moins de n images sont disponibles, complète avec
    des images noires et grises.
    """
    if img_extensions is None:
        img_extensions = {'.png', '.jpg', '.jpeg',
                          '.bmp', '.tiff', '.tif'}

    paths = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in img_extensions
    ])[:n]

    background = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB').resize(img_size)
            background.append(
                np.array(img, dtype=np.float32) / 255.0)
        except Exception:
            pass

    # Complétion si nécessaire
    while len(background) < 5:
        background.append(
            np.zeros((img_size[0], img_size[1], 3),
                     dtype=np.float32))
        background.append(
            np.full((img_size[0], img_size[1], 3),
                    0.5, dtype=np.float32))

    background = background[:n]
    print(f"[Background] {len(background)} images utilisées.")
    return background




if __name__ == '__main__':

    input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
    output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/shap_results"
    os.makedirs(output_folder, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                      '.bmp', '.tiff', '.tif'}
    IMG_SIZE  = (224, 224)
    CLASS_IDX = 0

    print("[INFO] Chargement du modèle ResNet50...")
    model = tf.keras.applications.ResNet50(weights='imagenet')


    def model_fn(image):
        x     = preprocess_resnet(image)
        preds = model(x, training=False).numpy()
        return float(preds[0, CLASS_IDX])

   
    print("\n[INFO] Construction du background...")
    background = build_background(
        input_folder, n=20,
        img_size=IMG_SIZE,
        img_extensions=IMG_EXTENSIONS
    )



    #  DeepSHAP 
    deep_explainer = DeepSHAPExplainer(
        model      = model,
        background = background,
        class_idx  = CLASS_IDX
    )

    # GradientSHAP (plus robuste que DeepSHAP)
    grad_explainer = GradientSHAPExplainer(
        model      = model,
        background = background,
        class_idx  = CLASS_IDX,
        n_samples  = 50
    )

    # PartitionSHAP (le plus précis, le plus lent)
    partition_explainer = PartitionSHAPExplainer(
        model_fn     = model_fn,
        class_idx    = CLASS_IDX,
        max_evals    = 1000,    # augmenter pour plus de précision
        masker_type  = 'blur'   # 'blur' recommandé pour IRM
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
            print(f"  [ERREUR chargement] {e} — ignorée.")
            continue

        npy_dir = os.path.join(output_folder, img_name)
        os.makedirs(npy_dir, exist_ok=True)

        results_dict = {}

        # DeepSHAP
        try:
            smap, ssigned, sraw = deep_explainer.explain(
                image, smooth_sigma=1.5)
            results_dict['DeepSHAP'] = (smap, ssigned)

            visualize_shap(
                image, smap, ssigned,
                method_name = 'DeepSHAP',
                save_path   = os.path.join(
                    output_folder,
                    f"{img_name}_deepshap.png"),
                title       = f"DeepSHAP — {img_name}"
            )
            np.save(os.path.join(npy_dir, "deepshap.npy"),    smap)
            np.save(os.path.join(npy_dir, "deepshap_raw.npy"),sraw)
            print(f"  [OK] DeepSHAP")
        except Exception as e:
            print(f"  [ERREUR DeepSHAP] {e}")

        #  GradientSHAP
        try:
            smap, ssigned, sraw = grad_explainer.explain(
                image, smooth_sigma=1.5)
            results_dict['GradSHAP'] = (smap, ssigned)

            visualize_shap(
                image, smap, ssigned,
                method_name = 'GradientSHAP',
                save_path   = os.path.join(
                    output_folder,
                    f"{img_name}_gradshap.png"),
                title       = f"GradientSHAP — {img_name}"
            )
            np.save(os.path.join(npy_dir, "gradshap.npy"),    smap)
            np.save(os.path.join(npy_dir, "gradshap_raw.npy"),sraw)
            print(f"  [OK] GradientSHAP")
        except Exception as e:
            print(f"  [ERREUR GradientSHAP] {e}")

        #  PartitionSHAP 
        try:
            smap, ssigned, sraw = partition_explainer.explain(
                image, smooth_sigma=1.5)
            results_dict['PartitionSHAP'] = (smap, ssigned)

            visualize_shap(
                image, smap, ssigned,
                method_name = 'PartitionSHAP',
                save_path   = os.path.join(
                    output_folder,
                    f"{img_name}_partitionshap.png"),
                title       = f"PartitionSHAP — {img_name}"
            )
            np.save(os.path.join(npy_dir, "partitionshap.npy"),    smap)
            np.save(os.path.join(npy_dir, "partitionshap_raw.npy"),sraw)
            print(f"  [OK] PartitionSHAP")
        except Exception as e:
            print(f"  [ERREUR PartitionSHAP] {e}")

        # Figure de comparaison des 3 méthodes
        if len(results_dict) > 1:
            try:
                visualize_shap_comparison(
                    image, results_dict,
                    save_path=os.path.join(
                        output_folder,
                        f"{img_name}_shap_comparison.png")
                )
                print(f"  [OK] Figure comparaison")
            except Exception as e:
                print(f"  [ERREUR comparaison] {e}")

        print(f"  Résultats → {npy_dir}")

    print(f"\n[TERMINÉ] Toutes les images ont été traitées.")
    print(f"Résultats dans : {output_folder}")
