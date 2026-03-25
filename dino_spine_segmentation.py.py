"""
============================================================
  dino_spine_segmentation.py

  Pipeline 100% non supervisé basé sur DINO (ViT)
  pour segmenter les 10 régions spinales et paraspinales
  sur IRM T2 axiale lombaire + analyse texture Mannil.

  Principe :
    1. Extraction des features DINO (ViT-S/8 pré-entraîné)
       → chaque patch 8×8 = vecteur de 384 dimensions
    2. Clustering K-Means sur les features DINO
       → les features DINO sont naturellement
         discriminantes pour les structures anatomiques
    3. Attribution anatomique par position + intensité T2
    4. Analyse texture Mannil 2018 par région

  Pourquoi DINO pour l'IRM ?
    - DINO apprend des features sémantiques sans labels
    - Les cartes d'attention de DINO segmentent
      naturellement les objets
    - Testé sur imagerie médicale avec de très bons résultats
    - Aucune annotation nécessaire

  Installation :
      pip install torch torchvision
      pip install scikit-learn scikit-image
      pip install numpy matplotlib pandas scipy pillow

  Usage :
      python dino_spine_segmentation.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from scipy.ndimage import (gaussian_filter,
                            binary_closing,
                            binary_fill_holes,
                            distance_transform_edt)
from scipy.stats import skew, kurtosis
from skimage import exposure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label as sk_label, regionprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/dino_segmentation_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


DINO_MODEL   = 'dino_vits8'   # ViT-Small patch 8 (meilleur
                               # résolution spatiale)
PATCH_SIZE   = 8               # taille des patches
IMG_SIZE     = 224             # taille entrée DINO
N_CLUSTERS   = 10              # nb de régions à segmenter
N_PCA_COMP   = 64              # réduction PCA avant K-Means


REGIONS = {
    0: ('Fond',                  [30,  30,  30]),
    1: ('Disque intervert.',     [255, 200,   0]),
    2: ('Sac thécal',            [  0, 180, 255]),
    3: ('Éminence postérieure',  [180,  90,   0]),
    4: ('Psoas gauche',          [220,  60, 180]),
    5: ('Psoas droit',           [ 80, 200,  60]),
    6: ('Multifidus gauche',     [ 50, 160,  80]),
    7: ('Multifidus droit',      [160,  60, 200]),
    8: ('Érecteur gauche',       [ 40,  80, 200]),
    9: ('Érecteur droit',        [210, 190,  40]),
}
N_REGIONS = len(REGIONS)



def load_dino_model(model_name=DINO_MODEL):
    """
    Charge le modèle DINO pré-entraîné depuis torch.hub.

    Modèles disponibles :
      - dino_vits8  : ViT-Small patch 8 (recommandé IRM)
      - dino_vits16 : ViT-Small patch 16 (plus rapide)
      - dino_vitb8  : ViT-Base  patch 8 (plus précis)
      - dino_vitb16 : ViT-Base  patch 16

    Patch 8 = meilleure résolution spatiale
    → préférable pour les petites structures anatomiques
    """
    import torch

    print(f"[DINO] Chargement du modèle {model_name}...")
    model = torch.hub.load(
        'facebookresearch/dino:main',
        model_name,
        pretrained=True
    )
    model.eval()

    device = 'cuda' if torch.cuda.is_available() \
             else 'cpu'
    model  = model.to(device)
    print(f"[DINO] Modèle chargé sur {device}")

    return model, device



def preprocess_for_dino(image_path, img_size=IMG_SIZE):
    """
    Prétraite l'image pour l'entrée DINO.

    DINO attend :
    - Image RGB normalisée ImageNet
    - Taille multiple de patch_size

    Pour IRM (niveaux de gris) → on duplique sur 3 canaux.
    On applique aussi CLAHE pour améliorer le contraste.
    """
    import torch
    import torchvision.transforms as T

    # Chargement + redimensionnement
    img = Image.open(str(image_path)).convert('L')
    img_arr = np.array(img, dtype=np.float32)
    img_arr = (img_arr - img_arr.min()) / \
              (img_arr.max() - img_arr.min() + 1e-8)

    # CLAHE pour révéler les structures musculaires
    img_clahe = exposure.equalize_adapthist(
        img_arr, clip_limit=0.02)

    # Taille multiple de patch_size
    size = (img_size // PATCH_SIZE) * PATCH_SIZE
    img_pil = Image.fromarray(
        (img_clahe * 255).astype(np.uint8))
    img_pil = img_pil.resize((size, size),
                              Image.BILINEAR)

    # RGB (3 canaux identiques)
    img_rgb = Image.merge('RGB', [img_pil]*3)

    # Normalisation ImageNet
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_rgb).unsqueeze(0)

    # Sauvegarder aussi l'image originale
    img_orig = np.array(
        Image.open(str(image_path))
             .convert('L')
             .resize((size, size), Image.BILINEAR),
        dtype=np.float32) / 255.0

    return tensor, img_orig, size



def extract_dino_features(model, tensor, device,
                           patch_size=PATCH_SIZE):
    """
    Extrait les features DINO (tokens patch) de l'image.

    DINO-ViT découpe l'image en patches et produit un
    vecteur de features par patch.

    Pour ViT-S/8 sur image 224×224 :
      - Nb patches = (224/8)² = 784 patches
      - Dim features = 384

    Retourne :
      features : ndarray (H_p, W_p, D)
                 H_p = W_p = img_size / patch_size
                 D = dimension des features (384)
    """
    import torch

    with torch.no_grad():
        tensor = tensor.to(device)

        # Extraction des features intermédiaires
        # (avant la tête de classification)
        features_dict = model.get_intermediate_layers(
            tensor, n=1)[0]

        # Shape : (1, N_patches + 1, D)
        # +1 pour le token [CLS]
        features = features_dict.squeeze(0)

        # Enlever le token [CLS] (premier token)
        patch_features = features[1:, :]  # (N_patches, D)

        # Reconstruire la grille spatiale
        h = w = int(patch_features.shape[0] ** 0.5)
        feat_map = patch_features.reshape(
            h, w, -1).cpu().numpy()

    print(f"  Features DINO : {feat_map.shape} "
          f"({h}×{w} patches × "
          f"{feat_map.shape[2]} dims)")

    return feat_map


def extract_dino_attention(model, tensor, device,
                            patch_size=PATCH_SIZE):
    """
    Extrait les cartes d'attention de DINO.

    Les cartes d'attention de DINO segmentent naturellement
    les objets — c'est une propriété émergente du modèle.

    Retourne :
      attention : ndarray (n_heads, H_p, W_p)
    """
    import torch

    with torch.no_grad():
        tensor   = tensor.to(device)
        h_feat   = tensor.shape[-2] // patch_size
        w_feat   = tensor.shape[-1] // patch_size

        # Accès aux attentions du dernier bloc
        attentions = model.get_last_selfattention(tensor)
        # Shape : (1, n_heads, N+1, N+1)

        n_heads = attentions.shape[1]
        # Attention du [CLS] vers les patches
        attn = attentions[0, :, 0, 1:].reshape(
            n_heads, h_feat, w_feat)
        attn = attn.cpu().numpy()

    return attn


# SEGMENTATION PAR CLUSTERING DES FEATURES DINO

def segment_with_dino(feat_map, img_orig,
                      n_clusters=N_CLUSTERS,
                      n_pca=N_PCA_COMP):
    """
    Segmente l'image en n_clusters régions en
    appliquant K-Means sur les features DINO.

    Étapes :
    1. Réduction PCA (384 → 64 dims)
    2. Ajout des features spatiales (x, y)
    3. K-Means
    4. Upscaling vers la taille originale

    Retourne :
      seg_map : ndarray (H_orig, W_orig) int
    """
    from scipy.ndimage import zoom

    h_p, w_p, d = feat_map.shape
    H, W         = img_orig.shape

    # Aplatir les features patches
    X = feat_map.reshape(-1, d)

    # --- PCA pour réduire la dimension ---
    n_comp = min(n_pca, X.shape[0]-1, X.shape[1])
    print(f"  PCA ({d} → {n_comp} dims)...",
          end=' ', flush=True)
    pca    = PCA(n_components=n_comp,
                 random_state=42)
    X_pca  = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_.sum()
    print(f"OK (variance expliquée : {var*100:.1f}%)")

    # --- Ajout des features spatiales ---
    yy, xx = np.mgrid[0:h_p, 0:w_p]
    yy_n   = (yy.flatten() / h_p).reshape(-1, 1)
    xx_n   = (xx.flatten() / w_p).reshape(-1, 1)

    # Pondération spatiale (0.5 = balance
    # features sémantiques vs position)
    spatial_weight = 0.5
    X_combined = np.hstack([
        X_pca,
        yy_n * spatial_weight * n_comp,
        xx_n * spatial_weight * n_comp,
    ])

    # Normalisation
    scaler     = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    # K-Means
    print(f"  K-Means ({n_clusters} clusters)...",
          end=' ', flush=True)
    kmeans  = KMeans(
        n_clusters  = n_clusters,
        random_state= 42,
        n_init      = 15,
        max_iter    = 500,
    )
    labels  = kmeans.fit_predict(X_combined)
    seg_map = labels.reshape(h_p, w_p)
    print("OK")

    # Upscaling vers la taille originale
    zh = H / h_p
    zw = W / w_p
    seg_full = zoom(seg_map.astype(float),
                    (zh, zw), order=0).astype(int)
    seg_full = np.clip(seg_full, 0, n_clusters - 1)

    return seg_full, kmeans


def assign_anatomy_dino(seg_map, img_orig, attention):
    """
    Attribue les labels anatomiques aux clusters DINO.

    Utilise :
    - La carte d'attention DINO (moyenne sur les têtes)
      → zones les plus saillantes = sac thécal + disques
    - L'intensité IRM T2
    - La position spatiale

    Règles T2 :
      Sac thécal      → haute attention + très brillant
                        + centre
      Disque          → haute intensité + centre-ant.
      Éminence post.  → basse intensité + centre-post.
      Psoas           → attention moyenne + latéral haut
      Multifidus      → postéro-central
      Érecteur        → postéro-latéral
    """
    from scipy.ndimage import zoom

    H, W = img_orig.shape
    cy, cx = H/2, W/2
    n_clusters = seg_map.max() + 1

    # Attention moyenne sur les têtes → (H, W)
    attn_mean = attention.mean(axis=0)
    # Upscale l'attention à la taille de l'image
    zh = H / attn_mean.shape[0]
    zw = W / attn_mean.shape[1]
    attn_full = zoom(attn_mean, (zh, zw), order=1)
    attn_full = (attn_full - attn_full.min()) / \
                (attn_full.max() - attn_full.min() + 1e-8)

    # Propriétés de chaque cluster
    props = []
    for k in range(n_clusters):
        m = seg_map == k
        n = m.sum()
        if n == 0:
            props.append({
                'k': k, 'n': 0,
                'mean_i': 0, 'mean_attn': 0,
                'cy': cy, 'cx': cx, 'dist_c': 0,
            })
            continue
        ys, xs   = np.where(m)
        mean_y   = float(ys.mean())
        mean_x   = float(xs.mean())
        dist_c   = np.sqrt(
            ((mean_y-cy)/H)**2 +
            ((mean_x-cx)/W)**2)
        props.append({
            'k'        : k,
            'n'        : n,
            'mean_i'   : float(img_orig[m].mean()),
            'std_i'    : float(img_orig[m].std()),
            'mean_attn': float(attn_full[m].mean()),
            'cy'       : mean_y,
            'cx'       : mean_x,
            'dist_c'   : dist_c,
        })

    remaining = list(props)

    # Fond : plus basse intensité 
    fond_k    = min(remaining,
                    key=lambda p: p['mean_i'])['k']
    remaining = [p for p in remaining
                 if p['k'] != fond_k]

    # Sac thécal : haute attention + brillant + centre 
    sac_score  = lambda p: (
        -p['mean_attn'] * 3
        - p['mean_i'] * 2
        + p['dist_c'] * 5)
    sac_k      = min(remaining,
                     key=sac_score)['k']
    remaining  = [p for p in remaining
                  if p['k'] != sac_k]

    # Disque : brillant + centre-antérieur 
    disc_score = lambda p: (
        -p['mean_i'] * 2
        + abs(p['cx']-cx)/W * 4
        + max(0, p['cy']-cy)/H * 3)
    disc_k     = min(remaining,
                     key=disc_score)['k']
    remaining  = [p for p in remaining
                  if p['k'] != disc_k]

    #  Éminence postérieure : sombre + centre post. 
    emin_score = lambda p: (
        p['mean_i'] * 2
        + abs(p['cx']-cx)/W * 4
        - max(0, p['cy']-cy)/H * 3)
    emin_k     = min(remaining,
                     key=emin_score)['k']
    remaining  = [p for p in remaining
                  if p['k'] != emin_k]

    # 6 régions musculaires
    # Séparer gauche / droite
    gauche = sorted(
        [p for p in remaining if p['cx'] < cx],
        key=lambda p: p['cy'])
    droite = sorted(
        [p for p in remaining if p['cx'] >= cx],
        key=lambda p: p['cy'])

    def assign_muscles(side_props, side):
        """Assigne psoas / multifidus / érecteur."""
        result = {}
        n = len(side_props)
        if n == 0:
            return result

        idx_psoas = 4 if side == 'G' else 5
        idx_multi = 6 if side == 'G' else 7
        idx_erect = 8 if side == 'G' else 9

        if n == 1:
            result[side_props[0]['k']] = idx_psoas
        elif n == 2:
            result[side_props[0]['k']] = idx_psoas
            result[side_props[1]['k']] = idx_erect
        else:
            # Le plus haut = psoas
            result[side_props[0]['k']] = idx_psoas
            # Parmi le reste : plus proche centre = multifidus
            rest = sorted(
                side_props[1:],
                key=lambda p: abs(p['cx']-cx))
            result[rest[0]['k']] = idx_multi
            for p in rest[1:]:
                result[p['k']] = idx_erect
        return result

    assignment = {
        fond_k: 0,
        disc_k: 1,
        sac_k : 2,
        emin_k: 3,
    }
    assignment.update(assign_muscles(gauche, 'G'))
    assignment.update(assign_muscles(droite, 'D'))

    # Carte anatomique finale
    anat_map = np.zeros((H, W), dtype=np.int8)
    for k in range(n_clusters):
        anat_map[seg_map == k] = assignment.get(k, 0)

    # Nettoyage morphologique
    for idx in range(1, N_REGIONS):
        m = anat_map == idx
        m = morphology.remove_small_objects(m,
                                            min_size=20)
        m = binary_closing(m, morphology.disk(2))
        anat_map[anat_map == idx] = 0
        anat_map[m]               = idx

    return anat_map, attn_full


# analyse texture de mannil
def extract_texture_mannil(image, mask):
    """Features texture Mannil 2018 (histogramme + GLCM)."""
    pixels = image[mask].astype(np.float32)
    if len(pixels) < 20:
        return None

    counts, _ = np.histogram(pixels, bins=256,
                              range=(0, 1))
    probs     = counts / (counts.sum() + 1e-8)
    pnz       = probs[probs > 0]

    feat = {
        'n_pixels'       : int(len(pixels)),
        'hist_mean'      : float(np.mean(pixels)),
        'hist_variance'  : float(np.var(pixels)),
        'hist_std'       : float(np.std(pixels)),
        'hist_skewness'  : float(skew(pixels)),
        'hist_kurtosis'  : float(kurtosis(pixels)),
        'hist_entropy'   : float(
            -np.sum(pnz * np.log2(pnz))),
        'hist_p10'       : float(np.percentile(pixels,10)),
        'hist_p25'       : float(np.percentile(pixels,25)),
        'hist_p50'       : float(np.percentile(pixels,50)),
        'hist_p75'       : float(np.percentile(pixels,75)),
        'hist_p90'       : float(np.percentile(pixels,90)),
    }

    try:
        img_u8 = (image * 255).astype(np.uint8)
        glcm   = graycomatrix(
            img_u8,
            distances=[1, 2, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256, symmetric=True, normed=True)
        feat['glcm_contrast']    = float(
            graycoprops(glcm, 'contrast').mean())
        feat['glcm_energy']      = float(
            graycoprops(glcm, 'energy').mean())
        feat['glcm_homogeneity'] = float(
            graycoprops(glcm, 'homogeneity').mean())
        feat['glcm_correlation'] = float(
            graycoprops(glcm, 'correlation').mean())
        p_nz = glcm[:,:,0,0]; p_nz = p_nz[p_nz > 0]
        feat['glcm_entropy']     = float(
            -np.sum(p_nz * np.log2(p_nz + 1e-10)))
    except Exception:
        for k in ['glcm_contrast','glcm_energy',
                  'glcm_homogeneity','glcm_correlation',
                  'glcm_entropy']:
            feat[k] = 0.0

    return feat





def make_color_map(anat_map):
    H, W      = anat_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (_, color) in REGIONS.items():
        color_img[anat_map == idx] = color
    return color_img


def visualize(img_orig, anat_map, attention,
              all_feat, img_name, save_path=None):
    """Figure complète : IRM + attention DINO +
    segmentation + overlay + histogrammes + features."""

    fig = plt.figure(figsize=(28, 16),
                     facecolor='black')
    fig.suptitle(
        f"DINO + Segmentation non supervisée + "
        f"Texture Mannil 2018 — {img_name}",
        color='white', fontsize=13,
        fontweight='bold')

    img01     = np.clip(img_orig, 0, 1)
    color_seg = make_color_map(anat_map)

    gs1 = fig.add_gridspec(
        1, 4, left=0.01, right=0.99,
        top=0.92, bottom=0.62, wspace=0.05)
    gs2 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.58, bottom=0.32, wspace=0.12)
    gs3 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.28, bottom=0.01, wspace=0.12)

    # IRM originale
    ax = fig.add_subplot(gs1[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('IRM T2 originale',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Carte d'attention DINO
    ax = fig.add_subplot(gs1[1])
    im = ax.imshow(attention, cmap='inferno')
    ax.set_title('Attention DINO\n(zones saillantes)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Segmentation colorée
    ax = fig.add_subplot(gs1[2])
    ax.imshow(color_seg)
    patches = [
        mpatches.Patch(
            color=np.array(REGIONS[i][1])/255,
            label=REGIONS[i][0])
        for i in range(N_REGIONS)]
    ax.legend(handles=patches,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.24),
              ncol=5, fontsize=6,
              facecolor='#222',
              labelcolor='white',
              framealpha=0.85)
    ax.set_title('Segmentation DINO\n(non supervisée)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Overlay
    ax = fig.add_subplot(gs1[3])
    overlay = np.stack([img01]*3, axis=-1)
    c_f     = color_seg.astype(np.float32) / 255
    ov      = np.clip(0.45*overlay + 0.55*c_f, 0, 1)
    ax.imshow(ov)
    ax.set_title('Overlay IRM + segmentation',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Histogrammes + tableaux par région
    for i in range(N_REGIONS):
        name  = REGIONS[i][0]
        color = np.array(REGIONS[i][1]) / 255
        feat  = all_feat.get(i)

        ax_h = fig.add_subplot(gs2[i])
        ax_h.set_facecolor('#111')
        mask_r = anat_map == i
        if feat and feat['n_pixels'] > 20:
            px = img_orig[mask_r]
            ax_h.hist(px, bins=30, color=color,
                      edgecolor='none', alpha=0.9)
            ax_h.axvline(feat['hist_mean'],
                         color='white',
                         linestyle='--',
                         linewidth=1.0)
        else:
            ax_h.text(0.5, 0.5, 'vide',
                      ha='center', va='center',
                      color='gray', fontsize=7,
                      transform=ax_h.transAxes)
        ax_h.set_title(name, color=color,
                       fontsize=6, fontweight='bold')
        ax_h.tick_params(colors='white', labelsize=5)
        for sp in ax_h.spines.values():
            sp.set_edgecolor('#333')

        ax_t = fig.add_subplot(gs3[i])
        ax_t.axis('off')
        if feat:
            keys = [
                ('hist_mean',        'Mean'),
                ('hist_variance',    'Variance'),
                ('hist_entropy',     'Entropy'),
                ('glcm_entropy',     'GLCM Entr.'),
                ('glcm_contrast',    'Contrast'),
                ('glcm_energy',      'Energy'),
                ('glcm_homogeneity', 'Homog.'),
                ('glcm_correlation', 'Corr.'),
            ]
            rows = [[lbl, f"{feat.get(k, 0):.3f}"]
                    for k, lbl in keys if k in feat]
            if rows:
                tbl = ax_t.table(
                    cellText  = rows,
                    colLabels = ['Feature', 'Val.'],
                    cellLoc   = 'center',
                    loc       = 'center',
                    bbox      = [0, 0, 1, 1])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(5.5)
                for (r, c), cell in \
                        tbl.get_celld().items():
                    cell.set_facecolor(
                        '#2a2a2a' if r%2==0 else '#111')
                    cell.set_text_props(color='white')
                    cell.set_edgecolor('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight',
                    facecolor='black')
        print(f"  [OK] → {save_path}")
    else:
        plt.show()
    plt.close(fig)



if __name__ == '__main__':

    import torch

    # Chargement DINO (une seule fois)
    model, device = load_dino_model(DINO_MODEL)

    image_paths = sorted([
        Path(INPUT_FOLDER) / f
        for f in os.listdir(INPUT_FOLDER)
        if Path(f).suffix.lower() in IMG_EXTENSIONS
    ])

    if not image_paths:
        print(f"[ERREUR] Aucune image dans : "
              f"{INPUT_FOLDER}")
        exit(1)

    print(f"\n[INFO] {len(image_paths)} image(s)")
    print(f"[INFO] Résultats → {OUTPUT_FOLDER}\n")

    all_rows = []

    for img_path in image_paths:
        name = img_path.stem
        print(f"\n{'='*60}")
        print(f"[IMAGE] {name}")
        print(f"{'='*60}")

        # Prétraitement
        try:
            tensor, img_orig, size = \
                preprocess_for_dino(img_path)
        except Exception as e:
            print(f"  [ERREUR prétraitement] {e}")
            continue

        # Extraction features DINO
        try:
            feat_map   = extract_dino_features(
                model, tensor, device)
            attention  = extract_dino_attention(
                model, tensor, device)
        except Exception as e:
            print(f"  [ERREUR DINO] {e}")
            continue

        # Segmentation K-Means sur features DINO
        try:
            seg_map, _ = segment_with_dino(
                feat_map, img_orig,
                n_clusters=N_CLUSTERS)
        except Exception as e:
            print(f"  [ERREUR segmentation] {e}")
            continue

        # Attribution anatomique
        try:
            anat_map, attn_full = assign_anatomy_dino(
                seg_map, img_orig, attention)
        except Exception as e:
            print(f"  [ERREUR anatomie] {e}")
            continue

        # Analyse texture Mannil
        all_feat = {}
        print("  Texture Mannil par région :")
        for idx in range(N_REGIONS):
            mask_r = anat_map == idx
            feat   = extract_texture_mannil(
                img_orig, mask_r)
            all_feat[idx] = feat
            rname  = REGIONS[idx][0]
            if feat:
                print(f"    {rname:22s} "
                      f"{feat['n_pixels']:5d}px "
                      f"mean={feat['hist_mean']:.3f} "
                      f"entr={feat['hist_entropy']:.2f}")
            else:
                print(f"    {rname:22s} → vide")

        # Visualisation
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_dino_mannil.png")
        visualize(img_orig, anat_map, attn_full,
                  all_feat, name, save_path=fig_path)

        # Sauvegardes
        np.save(
            os.path.join(OUTPUT_FOLDER,
                         f"{name}_anat_map.npy"),
            anat_map)
        Image.fromarray(
            make_color_map(anat_map)).save(
            os.path.join(OUTPUT_FOLDER,
                         f"{name}_color_seg.png"))

        # CSV
        row = {'patient': name}
        for idx in range(N_REGIONS):
            rname = REGIONS[idx][0].replace(' ', '_')
            feat  = all_feat.get(idx)
            if feat:
                for k, v in feat.items():
                    row[f"{rname}_{k}"] = v
        all_rows.append(row)

    # CSV global
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(
            OUTPUT_FOLDER,
            'all_patients_dino_mannil.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[CSV] → {csv_path}")
        print(f"  {df.shape[0]} patients × "
              f"{df.shape[1]} features")

        print("\n[RÉSUMÉ] hist_mean par région :")
        for idx in range(N_REGIONS):
            rname = REGIONS[idx][0]
            col   = (f"{rname.replace(' ','_')}"
                     f"_hist_mean")
            if col in df.columns:
                print(f"  {rname:25s} "
                      f"{df[col].mean():.4f} "
                      f"± {df[col].std():.4f}")

    print(f"\n[TERMINÉ] → {OUTPUT_FOLDER}")