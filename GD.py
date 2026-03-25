"""
============================================================
  gradcampp_saliency.py
  GradCAM++ adapté à l'imagerie médicale (IRM).

  Référence :
    Chattopadhyay et al., "Grad-CAM++", WACV 2018.
    https://arxiv.org/abs/1710.11063

  Installation :
      pip install tensorflow numpy matplotlib pillow scipy

  Usage :
      python gradcampp_saliency.py
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom



# chemin
input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/GD_results"
os.makedirs(output_folder, exist_ok=True)



#utilitaire
def load_image(path, img_size=(224, 224)):
    pil_img = Image.open(path).convert('RGB').resize(img_size)
    return np.array(pil_img, dtype=np.float32) / 255.0


def normalize_map(smap, percentile=99):
    smap = np.abs(smap)
    vmax = np.percentile(smap, percentile)
    if vmax == 0:
        return smap
    return np.clip(smap / vmax, 0, 1)


def resize_map(smap, target_shape):
    zh = target_shape[0] / smap.shape[0]
    zw = target_shape[1] / smap.shape[1]
    return zoom(smap, (zh, zw), order=1)


def smooth_map(smap, sigma=1.5):
    return gaussian_filter(smap, sigma=sigma)


def overlay_heatmap(image, heatmap, alpha=0.55, colormap='inferno'):
    cmap  = plt.get_cmap(colormap)
    hmap3 = cmap(heatmap)[..., :3]
    img01 = np.clip(image, 0, 1)
    return np.clip((1 - alpha) * img01 + alpha * hmap3, 0, 1)




#  CONSTRUCTION DU MODÈLE À DEUX SORTIES

def build_grad_model(base_model, conv_layer_name):
    """Construit un modèle exposant [conv_features, logits].

    Args:
        base_model      : tf.keras.Model backbone complet.
        conv_layer_name : str — nom de la couche conv cible.

    Returns:
        grad_model : tf.keras.Model à deux sorties.
    """
    import tensorflow as tf

    # Vérification que la couche existe
    layer_names = [l.name for l in base_model.layers]
    if conv_layer_name not in layer_names:
        print(f"\n[ERREUR] Couche '{conv_layer_name}' introuvable.")
        print("[INFO] Couches disponibles (15 dernières) :")
        for layer in base_model.layers[-15:]:
            try:
                shape = str(layer.output_shape)
            except AttributeError:
                shape = "N/A"
            print(f"  {layer.name:55s} {shape}")
        raise ValueError(f"Couche introuvable : {conv_layer_name}")

    conv_out = base_model.get_layer(conv_layer_name).output

    grad_model = tf.keras.Model(

        inputs  = base_model.input,
        outputs = [conv_out, base_model.output],
        name    = 'grad_model'
    )
    print(f"[build_grad_model] Couche conv : {conv_out.shape}")
    print(f"[build_grad_model] Logits      : {base_model.output.shape}")
    return grad_model



#  GRADCAM++

class GradCAMPlusPlus:
    """GradCAM++ — gradients d'ordre 1, 2 et 3.

    Paramètres
    ----------
    grad_model   : tf.keras.Model à DEUX sorties
                   [conv_features, logits]
                   Construire avec build_grad_model().
    class_idx    : int — classe à expliquer (None = auto).
    smooth_sigma : float — lissage post-calcul.
    """

    def __init__(self, grad_model, class_idx=None, smooth_sigma=1.5):
        import tensorflow as tf
        self.tf           = tf
        self.grad_model   = grad_model
        self.class_idx    = class_idx
        self.smooth_sigma = smooth_sigma
        print("[GradCAM++] Initialisé.")

    def explain(self, image, verbose=True):
        """Carte GradCAM++ pour une image prétraitée.

        Args:
            image   : ndarray (H, W, 3) float32 prétraité.
            verbose : affiche progression.

        Returns:
            dict : saliency_map, gradcam_raw, class_idx,
                   class_score, conv_shape
        """
        tf   = self.tf
        H, W = image.shape[:2]

        img_t = tf.cast(tf.expand_dims(image, 0), tf.float32)

        if verbose:
            print("[GradCAM++] Calcul gradients ordre 1, 2, 3...")

        # Trois GradientTape imbriqués pour ordres 1, 2, 3
        with tf.GradientTape(persistent=True) as t3:
            with tf.GradientTape(persistent=True) as t2:
                with tf.GradientTape(persistent=True) as t1:
                    conv_out, preds = \
                        self.grad_model(img_t, training=False)
                    t1.watch(conv_out)
                    t2.watch(conv_out)
                    t3.watch(conv_out)

                    cidx = self.class_idx if self.class_idx is not None \
                           else int(tf.argmax(preds[0]).numpy())
                    score = preds[:, cidx]

                g1 = t1.gradient(score, conv_out)
            g2 = t2.gradient(g1, conv_out)
        g3 = t3.gradient(g2, conv_out)
        del t1, t2, t3

        class_score = float(preds[0, cidx].numpy())

        if verbose:
            print(f"[GradCAM++] Classe={cidx}  Score={class_score:.4f}")

        # numpy
        c  = conv_out[0].numpy()   # (h, w, C)
        g1 = g1[0].numpy()
        g2 = g2[0].numpy()
        g3 = g3[0].numpy()

        # Coefficients alpha
        eps     = 1e-7
        gsum    = np.sum(c * g3, axis=(0, 1), keepdims=True)
        alpha   = g2 / (2.0 * g2 + gsum + eps)
        weights = np.sum(alpha * np.maximum(g1, 0), axis=(0, 1))

        # Carte brute
        raw = np.zeros(c.shape[:2], dtype=np.float32)
        for i, w in enumerate(weights):
            raw += w * c[:, :, i]
        raw = np.maximum(raw, 0)

        # Upscale + normalisation + lissage
        smap = resize_map(raw, (H, W)) if raw.shape != (H, W) else raw.copy()
        smap = normalize_map(smap)
        if self.smooth_sigma > 0:
            smap = normalize_map(smooth_map(smap, self.smooth_sigma))

        if verbose:
            print(f"[GradCAM++] Conv shape : {c.shape}")

        return {
            'saliency_map': smap,
            'gradcam_raw' : raw,
            'class_idx'   : cidx,
            'class_score' : class_score,
            'conv_shape'  : c.shape,
        }


#  GRADCAM CLASSIQUE (comparaison)

class GradCAMClassic:
    """GradCAM classique — même interface que GradCAMPlusPlus."""

    def __init__(self, grad_model, class_idx=None, smooth_sigma=1.5):
        import tensorflow as tf
        self.tf           = tf
        self.grad_model   = grad_model
        self.class_idx    = class_idx
        self.smooth_sigma = smooth_sigma
        print("[GradCAM]   Initialisé.")

    def explain(self, image, verbose=False):
        tf   = self.tf
        H, W = image.shape[:2]
        img_t = tf.cast(tf.expand_dims(image, 0), tf.float32)

        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(img_t, training=False)
            tape.watch(conv_out)
            cidx  = self.class_idx if self.class_idx is not None \
                    else int(tf.argmax(preds[0]).numpy())
            score = preds[:, cidx]

        grads = tape.gradient(score, conv_out)

        c = conv_out[0].numpy()
        g = grads[0].numpy()

        weights = np.mean(g, axis=(0, 1))
        raw     = np.zeros(c.shape[:2], dtype=np.float32)
        for i, w in enumerate(weights):
            raw += w * c[:, :, i]
        raw  = np.maximum(raw, 0)

        smap = resize_map(raw, (H, W)) if raw.shape != (H, W) else raw.copy()
        smap = normalize_map(smap)
        if self.smooth_sigma > 0:
            smap = normalize_map(smooth_map(smap, self.smooth_sigma))

        return {
            'saliency_map': smap,
            'gradcam_raw' : raw,
            'class_idx'   : cidx,
            'class_score' : float(preds[0, cidx].numpy()),
            'conv_shape'  : c.shape,
        }


#visualisation


def visualize_gradcampp(image, result_pp, result_classic=None,
                         save_path=None, title='GradCAM++'):

    img01   = np.clip(image, 0, 1)
    smap_pp = result_pp['saliency_map']
    cmap_h  = plt.get_cmap('inferno')

    heatmap_pp = cmap_h(smap_pp)[..., :3]
    overlay_pp = overlay_heatmap(img01, smap_pp)

    # Masque binaire p75
    thresh      = np.percentile(smap_pp, 75)
    binary_mask = (smap_pp >= thresh).astype(np.uint8) * 255

    # Contour vert
    from scipy.ndimage import binary_erosion
    mask_bool   = smap_pp >= thresh
    eroded      = binary_erosion(mask_bool, iterations=2)
    contour     = mask_bool & ~eroded
    img_contour = img01.copy()
    img_contour[contour] = [0, 1, 0]

    if result_classic is not None:
        smap_c    = result_classic['saliency_map']
        heatmap_c = cmap_h(smap_c)[..., :3]
        overlay_c = overlay_heatmap(img01, smap_c)

        fig, axes = plt.subplots(2, 4, figsize=(22, 10),
                                 facecolor='black')
        row1  = [img01, heatmap_c,  overlay_c,  img01]
        row2  = [img01, heatmap_pp, overlay_pp, img_contour]
        tits1 = ['Original', 'GradCAM Heatmap',
                 'GradCAM Overlay', '(référence)']
        tits2 = ['Original', 'GradCAM++ Heatmap',
                 'GradCAM++ Overlay', 'GradCAM++ Contour']

        for ax, img, t in zip(axes[0], row1, tits1):
            ax.imshow(img); ax.set_title(t, color='#aaa', fontsize=9)
            ax.axis('off'); ax.set_facecolor('black')
        for ax, img, t in zip(axes[1], row2, tits2):
            ax.imshow(img)
            ax.set_title(t, color='white', fontsize=9,
                         fontweight='bold')
            ax.axis('off'); ax.set_facecolor('black')

        info = (f"GradCAM score={result_classic['class_score']:.3f}  |  "
                f"GradCAM++ score={result_pp['class_score']:.3f}  "
                f"classe={result_pp['class_idx']}")
    else:
        fig, axes = plt.subplots(1, 5, figsize=(24, 5),
                                 facecolor='black')
        panels = [
            (img01,       'Image originale'),
            (heatmap_pp,  'Heatmap'),
            (overlay_pp,  'Overlay'),
            (binary_mask, 'Masque (p75)'),
            (img_contour, 'Contour'),
        ]
        for ax, (img, t) in zip(axes, panels):
            kw = {'cmap': 'gray'} if img.ndim == 2 else {}
            ax.imshow(img, **kw)
            ax.set_title(t, color='white', fontsize=9,
                         fontweight='bold')
            ax.axis('off'); ax.set_facecolor('black')

        info = (f"Classe={result_pp['class_idx']}  "
                f"Score={result_pp['class_score']:.4f}  "
                f"Conv={result_pp['conv_shape']}")

    fig.suptitle(f"{title}\n{info}",
                 color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        print(f"[Visu] → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_activation_hist(result, save_path=None):
    flat = result['gradcam_raw'].flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.hist(flat, bins=50, color='#e07b39', edgecolor='none',
            alpha=0.85)
    ax.axvline(flat.mean(), color='white', linestyle='--',
               linewidth=1.5, label=f'μ={flat.mean():.3f}')
    ax.set_xlabel('Activation brute', color='white')
    ax.set_ylabel('Pixels', color='white')
    ax.set_title('Distribution activations GradCAM++',
                 color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#333', labelcolor='white')
    for s in ax.spines.values():
        s.set_edgecolor('#555')
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
    output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/gradcampp_results"
    os.makedirs(output_folder, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


    IMG_SIZE   = (224, 224)
    CONV_LAYER = 'conv5_block3_out'
    CLASS_IDX  = 0

    #  MODÈLE
    print("[INFO] Chargement du modèle...")
    base_model = tf.keras.applications.ResNet50(weights='imagenet')

    print("\n[INFO] 10 dernières couches :")
    for layer in base_model.layers[-10:]:
        try:
            shape = str(layer.output_shape)
        except AttributeError:
            shape = "N/A"
        print(f"  {layer.name:55s} {shape}")



    #  CONSTRUCTION DU MODÈLE À DEUX SORTIES
    #  grad_model : image → [conv_features, logits]
    grad_model = build_grad_model(base_model, CONV_LAYER)
    



    def preprocess_image(image):
        x = np.expand_dims(image * 255.0, axis=0).astype(np.float32)
        result = tf.keras.applications.resnet50.preprocess_input(x)
    # Compatibilité numpy/tensorflow
        if hasattr(result, 'numpy'):
            return result[0].numpy()
        return np.array(result)[0]
    
    
    #  INSTANCIATION
    gradcampp = GradCAMPlusPlus(
        grad_model   = grad_model,
        class_idx    = CLASS_IDX,
        smooth_sigma = 1.5,
    )

    gradcam = GradCAMClassic(
        grad_model   = grad_model,
        class_idx    = CLASS_IDX,
        smooth_sigma = 1.5,
    )

    #  BOUCLE SUR LES IMAGES
    image_paths = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])

    if not image_paths:
        print(f"\n[ERREUR] Aucune image dans : {input_folder}")
    else:
        print(f"\n[INFO] {len(image_paths)} image(s) dans : {input_folder}")
        print(f"[INFO] Résultats dans       : {output_folder}\n")

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'='*60}\n[IMAGE] {img_name}\n{'='*60}")

        try:
            image      = load_image(img_path, IMG_SIZE)
            image_prep = preprocess_image(image)
        except Exception as e:
            print(f"  [ERREUR chargement] {e}"); continue

        try:
            result_pp = gradcampp.explain(image_prep, verbose=True)
        except Exception as e:
            print(f"  [ERREUR GradCAM++] {e}"); continue

        try:
            result_c = gradcam.explain(image_prep, verbose=False)
            print(f"[GradCAM] Score={result_c['class_score']:.4f}")
        except Exception as e:
            print(f"  [WARN GradCAM] {e}"); result_c = None

        # Figures
        visualize_gradcampp(
            image, result_pp, result_classic=result_c,
            save_path=os.path.join(
                output_folder,
                f"{img_name}_gradcampp_vs_gradcam.png"),
            title=f"GradCAM++ vs GradCAM — {img_name}"
        )
        visualize_gradcampp(
            image, result_pp, result_classic=None,
            save_path=os.path.join(
                output_folder,
                f"{img_name}_gradcampp_only.png"),
            title=f"GradCAM++ — {img_name}"
        )
        plot_activation_hist(
            result_pp,
            save_path=os.path.join(
                output_folder,
                f"{img_name}_gradcampp_hist.png")
        )

        # Numpy
        npy_dir = os.path.join(output_folder, img_name)
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, 'saliency_gradcampp.npy'),
                result_pp['saliency_map'])
        np.save(os.path.join(npy_dir, 'gradcam_raw.npy'),
                result_pp['gradcam_raw'])
        if result_c is not None:
            np.save(os.path.join(npy_dir, 'saliency_gradcam.npy'),
                    result_c['saliency_map'])

        print(f"  [OK] score={result_pp['class_score']:.4f}  "
              f"conv={result_pp['conv_shape']}")

    print(f"\n[TERMINÉ] Résultats dans : {output_folder}")