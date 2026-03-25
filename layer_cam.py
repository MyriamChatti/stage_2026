"""
============================================================
  layercam_saliency.py
  Implémentation complète de LayerCAM adaptée à l'imagerie
  médicale (IRM, segmentation).

  LayerCAM améliore GradCAM en combinant plusieurs couches
  intermédiaires du réseau, donnant une carte multi-échelle
  plus précise et plus fidèle à l'anatomie.

  Référence :
  Jiang et al., "LayerCAM: Exploring Hierarchical Class
  Activation Maps for Localization", IEEE TIP 2021.

  Avantages pour l'IRM :
    - Multi-échelle : couches profondes = sémantique,
      couches peu profondes = détails fins anatomiques
    - Meilleure résolution spatiale que GradCAM standard
    - Pas de blob unique comme GradCAM
    - Facile à combiner avec un masque de segmentation

  Usage :
      python layercam_saliency.py
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
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/layer_cam"
os.makedirs(output_folder, exist_ok=True)

# 
#   GÉNÉRATEUR DE MASQUES ALÉATOIRES


def normalize_map(smap, percentile=99):
    """Normalise une carte dans [0, 1] avec clip au percentile."""
    smap = np.maximum(smap, 0)
    vmax = np.percentile(smap, percentile)
    if vmax == 0:
        return smap
    return np.clip(smap / vmax, 0, 1)


def resize_to(smap, target_shape):
    """Redimensionne une carte 2D vers target_shape (H, W)."""
    zh = target_shape[0] / smap.shape[0]
    zw = target_shape[1] / smap.shape[1]
    return zoom(smap, (zh, zw), order=1)


def smooth(smap, sigma=1.5):
    """Lissage gaussien léger."""
    return gaussian_filter(smap, sigma=sigma)


#  CLASSE LAYERCAM

class LayerCAM:
    """Implémentation de LayerCAM pour TensorFlow/Keras.

    LayerCAM calcule une carte de saillance pour chaque couche
    spécifiée, puis les fusionne par moyenne pondérée.

    Pour chaque couche :
        CAM_l = ReLU( somme_c( ReLU(grad_c) * activation_c ) )

    La pondération par ReLU(grad) est la clé de LayerCAM :
    elle ne retient que les gradients POSITIFS (qui contribuent
    à augmenter le score de la classe cible), ce qui donne une
    carte plus propre que GradCAM classique.

    Paramètres
    ----------
    model      : modèle Keras compilé.
    layer_names: liste de noms de couches à utiliser.
                 Si None, détection automatique des couches conv.
    class_idx  : indice de la classe à expliquer.
                 Si None, utilise la classe prédite.
    fusion     : 'mean' | 'max' | 'weighted' — stratégie de
                 fusion des cartes multi-couches.
    """

    def __init__(self, model, layer_names=None,
                 class_idx=None, fusion='weighted'):
        self.model     = model
        self.class_idx = class_idx
        self.fusion    = fusion

        # Détection automatique des couches conv si non spécifiées
        if layer_names is None:
            self.layer_names = self._auto_select_layers()
        else:
            self.layer_names = layer_names

        print(f"[LayerCAM] Couches utilisées ({len(self.layer_names)}) :")
        for name in self.layer_names:
            print(f"  - {name}")

        # Construction du modèle multi-sorties
        self._build_grad_model()

    def _auto_select_layers(self, n_layers=4):
        """Sélectionne automatiquement N couches conv réparties
        dans le réseau (début, milieu, fin) pour couvrir
        toutes les échelles."""
        conv_layers = [
            l.name for l in self.model.layers
            if isinstance(l, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D))
               and len(l.output_shape) == 4
        ]

        if not conv_layers:
            raise ValueError(
                "Aucune couche Conv2D trouvée dans le modèle.")

        # Répartition uniforme sur l'ensemble du réseau
        n    = len(conv_layers)
        step = max(1, n // n_layers)
        selected = conv_layers[::step][-n_layers:]
        return selected

    def _build_grad_model(self):
        """Construit un sous-modèle qui expose les activations
        de chaque couche + la sortie finale."""
        outputs = [
            self.model.get_layer(name).output
            for name in self.layer_names
        ]
        outputs.append(self.model.output)

        self.grad_model = tf.keras.Model(
            inputs  = self.model.input,
            outputs = outputs
        )

    def _compute_one_layer(self, image_tensor, layer_idx):
        """Calcule la carte LayerCAM pour UNE couche donnée.

        Returns:
            cam : ndarray 2D (h_layer, w_layer), non normalisée.
        """
        with tf.GradientTape() as tape:
            outputs    = self.grad_model(image_tensor, training=False)
            activations= outputs[layer_idx]        # (1, h, w, C)
            logits     = outputs[-1]               # (1, n_classes)

            # Classe cible
            if self.class_idx is not None:
                target = logits[:, self.class_idx]
            else:
                target = logits[:, tf.argmax(logits[0])]

            tape.watch(activations)

        # Gradient de la sortie par rapport aux activations
        grads = tape.gradient(target, activations)  # (1, h, w, C)

        # LayerCAM : ReLU(grad) * activation, puis somme sur C
        grads_relu = tf.nn.relu(grads)              # ne garder que >0
        cam        = tf.reduce_sum(
            grads_relu * activations, axis=-1)      # (1, h, w)
        cam        = tf.nn.relu(cam)               # ReLU finale
        cam        = cam[0].numpy()                # (h, w)

        return cam

    def explain(self, image, smooth_sigma=1.5, verbose=True):
        """Calcule la carte LayerCAM fusionnée pour toutes les couches.

        Args:
            image        : ndarray (H, W, C) ou (H, W), float32 [0,1].
            smooth_sigma : sigma du lissage gaussien post-fusion.
            verbose      : affiche la progression.

        Returns:
            fused_map    : ndarray (H, W) normalisée [0, 1].
            layer_maps   : dict {layer_name: carte normalisée (H, W)}.
        """
        H, W = image.shape[:2]

        # Préparation du tenseur batch (1, H, W, C)
        if image.ndim == 2:
            image_3c = np.stack([image] * 3, axis=-1)
        else:
            image_3c = image

        # Prétraitement ResNet50 (valeurs centrées)
        image_pp = tf.keras.applications.resnet50.preprocess_input(
            image_3c[np.newaxis] * 255.0)

        layer_maps = {}

        for idx, name in enumerate(self.layer_names):
            if verbose:
                print(f"  → couche [{idx+1}/{len(self.layer_names)}] "
                      f"{name} ...", end=' ', flush=True)
            try:
                cam = self._compute_one_layer(image_pp, idx)

                # Redimensionnement à la taille de l'image
                if cam.shape != (H, W):
                    cam = resize_to(cam, (H, W))

                cam = normalize_map(cam)

                if smooth_sigma > 0:
                    cam = smooth(cam, sigma=smooth_sigma)

                layer_maps[name] = cam.astype(np.float32)
                if verbose:
                    print("OK")

            except Exception as e:
                if verbose:
                    print(f"ERREUR ({e})")

        if not layer_maps:
            raise RuntimeError(
                "Aucune couche n'a pu être calculée.")
        



        # --------------------------------------------------------
        # Fusion multi-couches
        maps_list = list(layer_maps.values())

        if self.fusion == 'mean':
            fused = np.mean(maps_list, axis=0)

        elif self.fusion == 'max':
            fused = np.max(maps_list, axis=0)

        else:
            # 'weighted' : les couches profondes ont plus de poids
            # (plus sémantiques) mais les couches peu profondes
            # affinent les contours
            n = len(maps_list)
            # Poids croissants : couche la plus profonde = poids max
            weights = np.linspace(0.5, 1.5, n)
            weights /= weights.sum()
            fused   = np.zeros_like(maps_list[0], dtype=np.float64)
            for w, m in zip(weights, maps_list):
                fused += w * m

        # Normalisation finale
        fused = normalize_map(fused)
        if smooth_sigma > 0:
            fused = smooth(fused, sigma=smooth_sigma * 0.5)
            fused = normalize_map(fused)

        return fused.astype(np.float32), layer_maps


# ============================================================
# VISUALISATION

def visualize_layercam(image, fused_map, layer_maps,
                        save_path=None,
                        title='LayerCAM'):
    """Figure complète : image | cartes par couche | fusion | overlay | contour."""

    img01 = np.clip(image, 0, 1)
    if img01.ndim == 2:
        img01 = np.stack([img01] * 3, axis=-1)

    cmap_hot    = plt.get_cmap('inferno')
    n_layers    = len(layer_maps)

    # Colonnes : orig | couches individuelles | fusion | overlay | contour
    n_cols  = n_layers + 4
    figsize = (4 * n_cols, 4.5)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('black')
    gs = gridspec.GridSpec(1, n_cols, figure=fig,
                           wspace=0.05, hspace=0.05)
    col = 0

    # image 
    ax = fig.add_subplot(gs[col]); col += 1
    ax.imshow(img01)
    ax.set_title('Image\noriginale', color='white',
                 fontsize=9, fontweight='bold')
    ax.axis('off')

    # Cartes par couche 
    for name, cam in layer_maps.items():
        ax = fig.add_subplot(gs[col]); col += 1
        ax.imshow(cmap_hot(cam)[..., :3])
        short_name = name.split('/')[-1][-18:]  
        ax.set_title(f'{short_name}', color='#f5a623',
                     fontsize=7)
        ax.axis('off')

    # Fusion 
    ax = fig.add_subplot(gs[col]); col += 1
    im = ax.imshow(cmap_hot(fused_map)[..., :3])
    ax.set_title('Fusion\n(LayerCAM)', color='#7ed321',
                 fontsize=9, fontweight='bold')
    ax.axis('off')

    # Overlay
    overlay = np.clip(
        0.45 * img01 + 0.55 * cmap_hot(fused_map)[..., :3], 0, 1)
    ax = fig.add_subplot(gs[col]); col += 1
    ax.imshow(overlay)
    ax.set_title('Overlay', color='white', fontsize=9, fontweight='bold')
    ax.axis('off')

    # Contour
    threshold   = np.percentile(fused_map, 70)
    mask_bool   = fused_map >= threshold
    from scipy.ndimage import binary_erosion
    eroded      = binary_erosion(mask_bool, iterations=2)
    contour_map = mask_bool & ~eroded
    img_contour = img01.copy()
    img_contour[contour_map] = [0, 1, 0]

    ax = fig.add_subplot(gs[col])
    ax.imshow(img_contour)
    ax.set_title('Contour (p70)', color='white',
                 fontsize=9, fontweight='bold')
    ax.axis('off')

    fig.suptitle(title, color='white', fontsize=12,
                 fontweight='bold', y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[LayerCAM] Figure sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_layer_grid(image, layer_maps, save_path=None):
    """Grille dédiée aux cartes individuelles par couche (figure séparée)."""
    img01    = np.clip(image, 0, 1)
    if img01.ndim == 2:
        img01 = np.stack([img01] * 3, axis=-1)

    cmap_hot = plt.get_cmap('inferno')
    n        = len(layer_maps)
    ncols    = min(4, n)
    nrows    = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4.5 * nrows))
    fig.patch.set_facecolor('black')

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    flat_axes = [ax for row in axes for ax in row]

    for ax, (name, cam) in zip(flat_axes, layer_maps.items()):
        overlay = np.clip(
            0.4 * img01 + 0.6 * cmap_hot(cam)[..., :3], 0, 1)
        ax.imshow(overlay)
        ax.set_title(name.split('/')[-1],
                     color='#f5a623', fontsize=8)
        ax.axis('off')

    # Axes vides
    for ax in flat_axes[len(layer_maps):]:
        ax.axis('off')

    fig.suptitle('LayerCAM — Cartes par couche (overlay)',
                 color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[LayerCAM] Grille couches sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================
# MAIN - BOUCLE SUR MESIMAGES IRM

if __name__ == '__main__':

   
    input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
    output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/layercam_results"
    os.makedirs(output_folder, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

   
    IMG_SIZE  = (224, 224)
    CLASS_IDX = 0

    print("[INFO] Chargement du modèle ResNet50...")
    model = tf.keras.applications.ResNet50(weights='imagenet')

   
    # couches :
    #  Pour ResNet50, ces 4 couches couvrent toutes les échelles :
    #    - conv2_block3_out  → détails fins (bas niveau)
    #    - conv3_block4_out  → textures moyennes
    #    - conv4_block6_out  → structures anatomiques
    #    - conv5_block3_out  → sémantique globale (haut niveau)
    #

    LAYER_NAMES = [
        'conv2_block3_out',   # échelle fine
        'conv3_block4_out',   # échelle moyenne-fine
        'conv4_block6_out',   # échelle moyenne-grossière
        'conv5_block3_out',   # échelle grossière (sémantique)
    ]

    # ============================================================
    #  INITIALISATION LAYERCAM
    layercam = LayerCAM(
        model       = model,
        layer_names = LAYER_NAMES,
        class_idx   = CLASS_IDX,
        fusion      = 'weighted',   # 'mean' | 'max' | 'weighted'
    )

    # ============================================================
    #  BOUCLE 
    

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

        # chargement
        try:
            pil_img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            image   = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"  [ERREUR chargement] {e} — image ignorée.")
            continue

        # Calcul LayerCAM 
        try:
            fused_map, layer_maps = layercam.explain(
                image,
                smooth_sigma = 1.5,
                verbose      = True
            )
        except Exception as e:
            print(f"  [ERREUR LayerCAM] {e} — image ignorée.")
            continue

        #  Figure (toutes couches + fusion)
        fig_path = os.path.join(
            output_folder, f"{img_name}_layercam.png")
        visualize_layercam(
            image, fused_map, layer_maps,
            save_path = fig_path,
            title     = f"LayerCAM — {img_name}"
        )

        # Grille dédiée aux couches individuelles
        grid_path = os.path.join(
            output_folder, f"{img_name}_layercam_grid.png")
        visualize_layer_grid(image, layer_maps, save_path=grid_path)

        # Sauvegarde numpy 
        npy_dir = os.path.join(output_folder, img_name)
        os.makedirs(npy_dir, exist_ok=True)

        np.save(os.path.join(npy_dir, "fused.npy"), fused_map)
        for name, cam in layer_maps.items():
            safe_name = name.replace('/', '_')
            np.save(os.path.join(npy_dir, f"{safe_name}.npy"), cam)

        print(f"  [OK] → {fig_path}")

    print(f"\n[TERMINÉ] Toutes les images ont été traitées.")
    print(f"Résultats dans : {output_folder}")