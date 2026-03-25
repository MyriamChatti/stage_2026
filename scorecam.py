"""
============================================================
  scorecam_saliency.py
  Implémentation complète de ScoreCAM adaptée à l'imagerie
  médicale (IRM, segmentation).

  ScoreCAM ne dépend PAS des gradients — il mesure l'impact
  de chaque canal d'activation directement sur le score
  du modèle en masquant l'image par les cartes d'activation.

  Référence :
  Wang et al., "Score-CAM: Score-Weighted Visual Explanations
  for Convolutional Neural Networks", CVPR Workshop 2020.
  https://arxiv.org/abs/1910.01279

  Avantages pour l'IRM :
    - Zéro dépendance aux gradients → pas de bruit de gradient
    - Résolution spatiale bien meilleure que GradCAM
    - Plus stable sur les images médicales à faible contraste
    - Fonctionne avec n'importe quelle couche conv
    - Résultats reproductibles (pas de stochasticité)

  Usage :
      python scorecam_saliency.py
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
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/scorecam"
os.makedirs(output_folder, exist_ok=True)




def normalize_map(smap, percentile=99):
    """Normalise une carte dans [0, 1] avec clip au percentile."""
    smap = np.maximum(smap, 0)
    vmax = np.percentile(smap, percentile)
    if vmax == 0:
        return smap
    return np.clip(smap / vmax, 0, 1)


def smooth(smap, sigma=1.5):
    """Lissage gaussien léger."""
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
    if hasattr(result, 'numpy'):
        return result.numpy()
    return np.array(result)


# ============================================================
#  CLASSE SCORECAM

class ScoreCAM:
    """Implémentation de ScoreCAM pour TensorFlow/Keras.

    Algorithme :
    ────────────
    Pour chaque canal k de la couche cible :
      1. Upscale l'activation A_k à la taille de l'image
      2. Normalise A_k dans [0, 1] → masque M_k
      3. Applique M_k sur l'image : I_k = image * M_k + baseline * (1 - M_k)
      4. Passe I_k dans le modèle → score S_k pour la classe cible
    Carte finale = ReLU( somme_k(S_k * A_k_upscaled) )

    C'est la pondération par le SCORE (pas le gradient) qui
    rend ScoreCAM robuste et sans bruit.

    Paramètres
    ----------
    model      : modèle Keras.
    layer_name : nom de la couche conv cible.
                 Si None, détecte automatiquement la dernière conv.
    class_idx  : indice de la classe à expliquer.
                 Si None, utilise la classe prédite.
    batch_size : nombre de canaux traités en parallèle.
                 Réduire si mémoire insuffisante.
    baseline   : 'zeros' | 'mean' | 'blur'
                 Valeur de remplacement pour les zones non masquées.
    """

    def __init__(self, model, layer_name=None,
                 class_idx=None, batch_size=32,
                 baseline='zeros'):
        self.model      = model
        self.class_idx  = class_idx
        self.batch_size = batch_size
        self.baseline   = baseline

        # Sélection de la couche
        if layer_name is None:
            self.layer_name = self._find_last_conv()
        else:
            self.layer_name = layer_name

        print(f"[ScoreCAM] Couche cible : {self.layer_name}")
        print(f"[ScoreCAM] batch_size={batch_size}, "
              f"baseline='{baseline}'")

        # Modèle qui expose activations + logits
        self.activation_model = tf.keras.Model(
            inputs  = model.input,
            outputs = [
                model.get_layer(self.layer_name).output,
                model.output
            ]
        )

    def _find_last_conv(self):
        """Trouve automatiquement la dernière couche Conv2D."""
        conv_layers = [
            l.name for l in self.model.layers
            if isinstance(l, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D))
               and len(l.output_shape) == 4
        ]
        if not conv_layers:
            raise ValueError("Aucune couche Conv2D trouvée.")
        return conv_layers[-1]

    def _get_baseline(self, image, img_size):
        """Calcule la baseline selon le paramètre choisi."""
        if self.baseline == 'zeros':
            return np.zeros((1,) + image.shape, dtype=np.float32)
        elif self.baseline == 'mean':
            mean_val = image.mean(axis=(0, 1), keepdims=True)
            return np.tile(mean_val, (1, img_size[0], img_size[1], 1))
        elif self.baseline == 'blur':
            blurred = np.stack([
                gaussian_filter(image[:, :, c], sigma=15)
                for c in range(image.shape[2])
            ], axis=-1)
            return blurred[np.newaxis]
        else:
            return np.zeros((1,) + image.shape, dtype=np.float32)

    def explain(self, image, smooth_sigma=1.5, verbose=True):
        """Calcule la carte ScoreCAM pour une image.

        Args:
            image        : ndarray (H, W, C) float32 [0, 1].
            smooth_sigma : sigma du lissage gaussien.
            verbose      : affiche la progression.

        Returns:
            scorecam_map : ndarray (H, W) normalisée [0, 1].
            channel_scores: ndarray (n_channels,) scores par canal.
        """
        H, W = image.shape[:2]

        # Prétraitement
        x_pp = preprocess_resnet(image)   # (1, H, W, C)

        # Récupération des activations et prédiction originale
        activations, logits = self.activation_model(
            x_pp, training=False)
        activations = activations.numpy()[0]   # (h, w, n_channels)
        logits      = logits.numpy()[0]        # (n_classes,)

        # Classe cible
        if self.class_idx is not None:
            target_class = self.class_idx
        else:
            target_class = int(np.argmax(logits))
            if verbose:
                print(f"[ScoreCAM] Classe prédite : {target_class} "
                      f"(score={logits[target_class]:.4f})")

        n_channels = activations.shape[-1]
        if verbose:
            print(f"[ScoreCAM] {n_channels} canaux à traiter "
                  f"(batch_size={self.batch_size})...")

        # Baseline préprocessée
        baseline_raw = self._get_baseline(image, (H, W))
        baseline_pp  = preprocess_resnet(
            np.clip(baseline_raw[0], 0, 1))   # (1, H, W, C)

        channel_scores = np.zeros(n_channels, dtype=np.float32)

        # Traitement par batch de canaux
        n_batches = int(np.ceil(n_channels / self.batch_size))

        for b in range(n_batches):
            start = b * self.batch_size
            end   = min(start + self.batch_size, n_channels)

            if verbose:
                pct = int(100 * end / n_channels)
                print(f"  Canaux {start}–{end-1} "
                      f"({pct}%) ...", end='\r', flush=True)

            batch_images = []

            for k in range(start, end):
                # Upscale du canal k vers (H, W)
                act_k = activations[:, :, k]
                if act_k.shape != (H, W):
                    act_k = resize_to(act_k, (H, W))

                # Normalisation dans [0, 1]
                a_min, a_max = act_k.min(), act_k.max()
                if a_max - a_min > 1e-8:
                    mask_k = (act_k - a_min) / (a_max - a_min)
                else:
                    mask_k = np.zeros_like(act_k)

                # Image masquée : I_k = image * mask + baseline * (1-mask)
                mask_k3   = mask_k[..., np.newaxis]      # (H, W, 1)
                masked_img = (image * mask_k3
                              + baseline_raw[0] * (1 - mask_k3))
                masked_img = np.clip(masked_img, 0, 1)

                # Prétraitement
                masked_pp = preprocess_resnet(masked_img)
                batch_images.append(masked_pp[0])

            # Inférence sur le batch
            batch_np = np.stack(batch_images, axis=0)  # (B, H, W, C)
            preds    = self.model(batch_np, training=False).numpy()
            # preds : (B, n_classes)

            for i, k in enumerate(range(start, end)):
                channel_scores[k] = preds[i, target_class]

        if verbose:
            print(f"\n[ScoreCAM] Scores calculés. "
                  f"min={channel_scores.min():.4f}, "
                  f"max={channel_scores.max():.4f}")
            


        # --------------------------------------------------------
        # Calcul de la carte finale ScoreCAM
        # score_cam = ReLU( somme_k( score_k * upscale(A_k) ) )
        # --------------------------------------------------------
        # Score original de la classe (référence)
        original_score = logits[target_class]

        # On pondère par la différence de score (contribution)
        weights = channel_scores - original_score   # (n_channels,)
        weights = np.maximum(weights, 0)             # ReLU

        scorecam = np.zeros((H, W), dtype=np.float64)

        for k in range(n_channels):
            if weights[k] == 0:
                continue
            act_k = activations[:, :, k]
            if act_k.shape != (H, W):
                act_k = resize_to(act_k, (H, W))
            scorecam += weights[k] * act_k

        # ReLU finale + normalisation
        scorecam = np.maximum(scorecam, 0)
        scorecam = normalize_map(scorecam)

        if smooth_sigma > 0:
            scorecam = smooth(scorecam, sigma=smooth_sigma)
            scorecam = normalize_map(scorecam)

        return scorecam.astype(np.float32), channel_scores


# ============================================================
#  SCORECAM MULTI-COUCHES (LayerCAM-style)

class MultiLayerScoreCAM:
    """ScoreCAM appliqué sur plusieurs couches et fusionné.

    Combine la précision de ScoreCAM avec l'approche
    multi-échelle de LayerCAM pour une meilleure couverture
    spatiale des structures anatomiques.

    Paramètres
    ----------
    model       : modèle Keras.
    layer_names : liste de couches conv à utiliser.
                  Si None, sélection automatique de 4 couches.
    class_idx   : classe à expliquer.
    batch_size  : taille des batchs de canaux.
    fusion      : 'mean' | 'max' | 'weighted'
    baseline    : 'zeros' | 'mean' | 'blur'
    """

    def __init__(self, model, layer_names=None,
                 class_idx=None, batch_size=32,
                 fusion='weighted', baseline='blur'):
        self.fusion = fusion

        if layer_names is None:
            layer_names = self._auto_select_layers(model)

        print(f"[MultiLayerScoreCAM] {len(layer_names)} couches :")
        self.explainers = {}
        for name in layer_names:
            print(f"  - {name}")
            self.explainers[name] = ScoreCAM(
                model      = model,
                layer_name = name,
                class_idx  = class_idx,
                batch_size = batch_size,
                baseline   = baseline
            )

    @staticmethod
    def _auto_select_layers(model, n=4):
        conv_layers = [
            l.name for l in model.layers
            if isinstance(l, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D))
               and len(l.output_shape) == 4
        ]
        n_total = len(conv_layers)
        step    = max(1, n_total // n)
        return conv_layers[::step][-n:]

    def explain(self, image, smooth_sigma=1.5, verbose=True):
        """Calcule ScoreCAM pour chaque couche et fusionne.

        Returns:
            fused_map  : ndarray (H, W) normalisée [0, 1].
            layer_maps : dict {layer_name: carte (H, W)}.
        """
        H, W       = image.shape[:2]
        layer_maps = {}

        for name, explainer in self.explainers.items():
            if verbose:
                print(f"\n[MultiLayerScoreCAM] Couche : {name}")
            try:
                smap, _ = explainer.explain(
                    image,
                    smooth_sigma=smooth_sigma,
                    verbose=verbose
                )
                layer_maps[name] = smap
            except Exception as e:
                print(f"  [ERREUR] {name} : {e}")

        if not layer_maps:
            raise RuntimeError("Aucune couche calculée.")

        maps_list = list(layer_maps.values())

        if self.fusion == 'mean':
            fused = np.mean(maps_list, axis=0)
        elif self.fusion == 'max':
            fused = np.max(maps_list, axis=0)
        else:
            # Weighted : poids croissants (couches profondes > poids)
            n       = len(maps_list)
            weights = np.linspace(0.5, 1.5, n)
            weights /= weights.sum()
            fused   = sum(w * m for w, m in zip(weights, maps_list))

        fused = normalize_map(np.array(fused))
        if smooth_sigma > 0:
            fused = smooth(fused, sigma=smooth_sigma * 0.5)
            fused = normalize_map(fused)

        return fused.astype(np.float32), layer_maps






def visualize_scorecam(image, scorecam_map,
                        layer_maps=None,
                        save_path=None,
                        title='ScoreCAM'):
    """Figure complète ScoreCAM.

    Colonnes : image | heatmap | overlay | masque | contour
    Si layer_maps fourni : ajoute les cartes par couche.
    """
    img01 = np.clip(image, 0, 1)
    if img01.ndim == 2:
        img01 = np.stack([img01] * 3, axis=-1)

    cmap_hot = plt.get_cmap('inferno')

    # Heatmap
    heatmap = cmap_hot(scorecam_map)[..., :3]

    # Overlay
    overlay = np.clip(0.45 * img01 + 0.55 * heatmap, 0, 1)

    # Masque binaire (seuil p70)
    threshold  = np.percentile(scorecam_map, 70)
    mask_bool  = scorecam_map >= threshold

    # Contour
    from scipy.ndimage import binary_erosion
    eroded      = binary_erosion(mask_bool, iterations=2)
    contour_map = mask_bool & ~eroded
    img_contour = img01.copy()
    img_contour[contour_map] = [0, 1, 0]

    # Colonnes de base
    base_panels = [
        (img01,        'Image originale'),
        (heatmap,      'ScoreCAM heatmap'),
        (overlay,      'Overlay'),
        (img_contour,  'Contour (p70)'),
    ]

    # Ajout des couches individuelles si disponibles
    layer_panels = []
    if layer_maps:
        for name, lmap in layer_maps.items():
            ov = np.clip(0.4 * img01 + 0.6 * cmap_hot(lmap)[..., :3],
                         0, 1)
            short = name.split('/')[-1][-16:]
            layer_panels.append((ov, short))

    all_panels = base_panels + layer_panels
    n_cols     = len(all_panels)
    figsize    = (4.5 * n_cols, 5)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('black')
    gs  = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.04)

    for col, (img_p, label) in enumerate(all_panels):
        ax = fig.add_subplot(gs[col])
        ax.imshow(img_p)
        color = '#f5a623' if col >= len(base_panels) else 'white'
        weight = 'bold' if col < len(base_panels) else 'normal'
        ax.set_title(label, color=color,
                     fontsize=8, fontweight=weight)
        ax.axis('off')

    fig.suptitle(title, color='white', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[ScoreCAM] Figure sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_channel_scores(channel_scores, top_k=20,
                         save_path=None):
    """Histogramme des top-K canaux les plus importants."""
    top_idx    = np.argsort(channel_scores)[-top_k:][::-1]
    top_scores = channel_scores[top_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    bars = ax.bar(range(top_k), top_scores,
                  color='#e07b39', edgecolor='none', alpha=0.85)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f'ch{i}' for i in top_idx],
                        rotation=45, ha='right',
                        color='white', fontsize=7)
    ax.set_ylabel('Score canal', color='white')
    ax.set_title(f'Top {top_k} canaux ScoreCAM',
                 color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a1a')
        print(f"[ScoreCAM] Scores canaux → {save_path}")
    else:
        plt.show()
    plt.close(fig)



if __name__ == '__main__':

    input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
    output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/scorecam_results"
    os.makedirs(output_folder, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                      '.bmp', '.tiff', '.tif'}
    IMG_SIZE  = (224, 224)
    CLASS_IDX = 0


    print("[INFO] Chargement du modèle ResNet50...")
    model = tf.keras.applications.ResNet50(weights='imagenet')

    # ============================================================
    #  COUCHES CIBLES POUR MULTI-LAYER SCORECAM
    #
    #  Pour ResNet50, ces 4 couches couvrent toutes les échelles.
    
    LAYER_NAMES = [
        'conv2_block3_out',   # détails fins
        'conv3_block4_out',   # textures moyennes
        'conv4_block6_out',   # structures anatomiques
        'conv5_block3_out',   # sémantique globale
    ]

    #  MODE : 'single' ou 'multi'
    #    single = ScoreCAM sur une seule couche (plus rapide)
    #    multi  = MultiLayerScoreCAM (plus précis, plus lent)


    MODE = 'multi'   

    if MODE == 'single':
        print("[INFO] Mode : ScoreCAM simple couche")
        explainer = ScoreCAM(
            model      = model,
            layer_name = 'conv5_block3_out',
            class_idx  = CLASS_IDX,
            batch_size = 32,
            baseline   = 'blur'
        )
    else:
        print("[INFO] Mode : MultiLayerScoreCAM")
        explainer = MultiLayerScoreCAM(
            model       = model,
            layer_names = LAYER_NAMES,
            class_idx   = CLASS_IDX,
            batch_size  = 32,
            fusion      = 'weighted',
            baseline    = 'blur'
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

        #Chargement
        try:
            pil_img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            image   = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"  [ERREUR chargement] {e} — ignorée.")
            continue

        # Calcul ScoreCAM
        try:
            if MODE == 'single':
                scorecam_map, channel_scores = explainer.explain(
                    image, smooth_sigma=1.5, verbose=True)
                layer_maps = None
            else:
                scorecam_map, layer_maps = explainer.explain(
                    image, smooth_sigma=1.5, verbose=True)
                channel_scores = None

        except Exception as e:
            print(f"  [ERREUR ScoreCAM] {e} — ignorée.")
            continue

        # figure principale
        fig_path = os.path.join(
            output_folder, f"{img_name}_scorecam.png")
        visualize_scorecam(
            image, scorecam_map,
            layer_maps = layer_maps,
            save_path  = fig_path,
            title      = f"ScoreCAM — {img_name}"
        )

        # Histogramme des scores canaux (mode single)
        if channel_scores is not None:
            score_path = os.path.join(
                output_folder, f"{img_name}_channel_scores.png")
            plot_channel_scores(
                channel_scores, top_k=20,
                save_path=score_path)

        # sauvegarde
        npy_dir = os.path.join(output_folder, img_name)
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, "scorecam.npy"), scorecam_map)

        if layer_maps:
            for name, lmap in layer_maps.items():
                safe = name.replace('/', '_')
                np.save(os.path.join(npy_dir, f"{safe}.npy"), lmap)

        print(f"  [OK] → {fig_path}")

    print(f"\n[TERMINÉ] Toutes les images ont été traitées.")
    print(f"Résultats dans : {output_folder}")