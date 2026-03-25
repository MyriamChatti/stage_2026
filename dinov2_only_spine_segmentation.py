"""
============================================================
  dinov2_only_spine_segmentation.py

  Pipeline 100% non supervisé basé sur DINOv2 seul
  (sans STEGO) pour segmenter les 10 régions spinales
  et paraspinales sur IRM T2 axiale lombaire
  + analyse texture Mannil.

  Principe — DINOv2 seul :
    1. DINOv2 extrait les features sémantiques par patch
    2. PCA pour réduire les dimensions (384 → 32)
    3. K-Means directement sur les features DINOv2
       (pas de tête STEGO, pas d'entraînement)
    4. Attribution anatomique par position + intensité T2
    5. Analyse texture Mannil 2018

  Différence avec STEGO + DINOv2 :
    - Pas de tête de segmentation entraînable
    - Pas de perte contrastive
    - Plus rapide (pas d'entraînement)
    - Moins adapté à vos IRM spécifiques
    → Utile pour comparer avec STEGO et voir
      l'apport de la tête STEGO

  Installation :
      pip install torch torchvision
      pip install scikit-learn scikit-image scipy
      pip install numpy matplotlib pandas pillow

  Usage :
      python dinov2_only_spine_segmentation.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from scipy.ndimage import (binary_closing,
                            binary_fill_holes,
                            distance_transform_edt,
                            gaussian_filter)
from scipy.stats import skew, kurtosis
from skimage import exposure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler,
                                    normalize)
import warnings
warnings.filterwarnings('ignore')


# ============================================================
#  CHEMINS
# ============================================================

INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/dinov2_only_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
#  PARAMÈTRES
# ============================================================

IMG_SIZE       = 224
PATCH_SIZE     = 14      # DINOv2 vits14 patch 14
                          # (contexte anatomique plus large)
N_REGIONS      = 10
N_LAYERS       = 1       # couche DINO utilisée
STEGO_DIM      = 70      # dimension tête STEGO
STEGO_LR       = 5e-3    # learning rate micro-entraînement
STEGO_STEPS    = 30      # pas de gradient sur chaque image
STEGO_K        = 7       # nb voisins pour correspondances
STEGO_NEG_K    = 3       # nb voisins négatifs
TEMPERATURE    = 0.1     # température contrastive

# ============================================================
#  RÉGIONS ANATOMIQUES
# ============================================================

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


# ============================================================
#  1. CHARGEMENT DINOV2
# ============================================================

def load_dino(device, model_name='dinov2_vits14'):
    """
    Charge DINOv2 ViT-S/14 (Meta, 2023).

    DINOv2 vs DINO v1 :
      - Entraîné sur 142M images (vs 1.2M)
      - Patch 14×14 = contexte plus large
      - Features plus discriminantes pour IRM
    """
    import torch
    print(f"[DINOv2] Chargement {model_name}...",
          end=' ', flush=True)
    model = torch.hub.load(
        'facebookresearch/dinov2',
        model_name, pretrained=True)
    model.eval().to(device)
    print(f"OK ({device})")
    return model


# ============================================================
#  2. PRÉTRAITEMENT
# ============================================================

def preprocess(path, size=IMG_SIZE):
    """Prétraite l'IRM T2 pour DINO."""
    import torch
    import torchvision.transforms as T

    img = Image.open(str(path)).convert('L')
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.min()) / \
          (arr.max() - arr.min() + 1e-8)

    sz    = (size // PATCH_SIZE) * PATCH_SIZE
    img_r = Image.fromarray(
        (arr * 255).astype(np.uint8)
    ).resize((sz, sz), Image.BILINEAR)
    arr_r = np.array(img_r,
                     dtype=np.float32) / 255.0

    arr_c = exposure.equalize_adapthist(
        arr_r, clip_limit=0.02)

    img_rgb = Image.merge('RGB', [
        Image.fromarray(
            (arr_c * 255).astype(np.uint8))
    ] * 3)

    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    tensor = t(img_rgb).unsqueeze(0)

    return arr_r, arr_c, tensor, sz


# ============================================================
#  3. EXTRACTION FEATURES DINOV2 (CORRIGÉE)
# ============================================================

def extract_dino_features(model, tensor, device):
    """
    Extrait les features DINOv2 patch par patch.

    CORRECTION API DINOv2 :
    - reshape=False → retourne (1, N_patches, D)
    - On reshape manuellement en (H_p, W_p, D)

    Pour DINOv2 vits14 sur image 224×224 :
    → (224/14)² = 256 patches × 384 dims
    """
    import torch

    with torch.no_grad():
        tensor = tensor.to(device)

        # DINOv2 : reshape=False obligatoire
        out = model.get_intermediate_layers(
            tensor, n=1, reshape=False)[0]
        # Shape : (1, N_patches, D)
        f         = out.squeeze(0).cpu().numpy()
        n_patches = f.shape[0]
        d         = f.shape[1]
        h = w     = int(n_patches ** 0.5)
        feat_map  = f.reshape(h, w, d)

    print(f"  Features DINOv2 : {feat_map.shape} "
          f"({h}×{w} patches × {d} dims)")
    return feat_map


# ============================================================
#  4. SEGMENTATION PURE DINOV2 (sans STEGO)
# ============================================================

def dinov2_segment(feat_map, img_orig,
                   n_segments=N_REGIONS):
    """
    Segmentation directe sur features DINOv2
    sans tête STEGO — pipeline simple :

    1. Aplatir features (N_patches, 384)
    2. Normalisation L2
    3. PCA (384 → 32 dims)
    4. Ajout position spatiale
    5. K-Means

    Plus rapide que STEGO mais moins adapté
    aux données spécifiques.
    """
    from scipy.ndimage import zoom

    h_p, w_p, d = feat_map.shape
    H, W         = img_orig.shape
    N            = h_p * w_p

    # Normalisation L2
    feat_flat = feat_map.reshape(N, d)
    feat_flat = normalize(feat_flat, norm='l2')

    # PCA
    n_comp = min(32, N-1, d)
    print(f"  PCA DINOv2 ({d}→{n_comp} dims)...",
          end=' ', flush=True)
    pca      = PCA(n_components=n_comp,
                   random_state=42)
    feat_pca = pca.fit_transform(feat_flat)
    var      = pca.explained_variance_ratio_.sum()
    print(f"OK ({var*100:.1f}% variance)")

    # Ajout position spatiale
    yy, xx = np.mgrid[0:h_p, 0:w_p]
    pos    = np.stack([
        yy.flatten() / h_p * 0.4,
        xx.flatten() / w_p * 0.4,
    ], axis=1)

    X = np.hstack([feat_pca, pos])
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # K-Means
    print(f"  K-Means ({n_segments} clusters)...",
          end=' ', flush=True)
    km     = KMeans(n_clusters=n_segments,
                    random_state=42,
                    n_init=15, max_iter=500)
    labels = km.fit_predict(X_s)
    print("OK")

    seg_patch = labels.reshape(h_p, w_p)
    zh = H / h_p
    zw = W / w_p
    seg_full = zoom(seg_patch.astype(float),
                    (zh, zw), order=0).astype(int)
    seg_full = np.clip(seg_full, 0, n_segments-1)

    # Retourner aussi les features PCA
    # (utilisées à la place des codes STEGO
    #  pour la visualisation)
    return seg_full, feat_pca


# ============================================================
#  6. VISUALISATION DES CODES STEGO (PCA 2D)
# ============================================================

def visualize_stego_codes(stego_codes, seg_map,
                           img_orig,
                           save_path=None):
    """
    Visualise les codes STEGO dans l'espace 2D (PCA).

    Montre comment STEGO a organisé les patches :
    les patches du même segment forment des clusters
    bien séparés dans l'espace STEGO.
    """
    h_p = w_p = int(len(stego_codes) ** 0.5)
    seg_flat  = seg_map.flatten()

    # Downscale seg_map vers espace patches
    from scipy.ndimage import zoom
    H, W = img_orig.shape
    zh   = h_p / H
    zw   = w_p / W
    seg_patch = zoom(
        seg_map.astype(float),
        (zh, zw), order=0).astype(int).flatten()

    pca = PCA(n_components=2, random_state=42)
    codes_2d = pca.fit_transform(stego_codes)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5),
        facecolor='black')

    # PCA des codes STEGO
    ax = axes[0]
    ax.set_facecolor('#111')
    cmap_s = plt.get_cmap('tab10')
    for k in range(N_REGIONS):
        mask = seg_patch == k
        if mask.sum() == 0:
            continue
        c = cmap_s(k / N_REGIONS)
        ax.scatter(
            codes_2d[mask, 0],
            codes_2d[mask, 1],
            c=[c], s=5, alpha=0.6,
            label=REGIONS[k][0])
    ax.set_title('Codes STEGO (PCA 2D)',
                 color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#222',
              labelcolor='white',
              fontsize=6,
              markerscale=3)
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    # Carte des codes STEGO (1ère composante PCA)
    ax = axes[1]
    codes_map = pca.transform(
        stego_codes)[:, 0].reshape(h_p, w_p)
    from scipy.ndimage import zoom as zm
    codes_full = zm(codes_map,
                    (H/h_p, W/w_p), order=1)
    im = ax.imshow(codes_full, cmap='RdYlGn')
    ax.set_title('Carte STEGO (PC1)',
                 color='white', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax,
                 fraction=0.046, pad=0.04)

    fig.suptitle('Espace de représentation STEGO',
                 color='white', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight',
                    facecolor='black')
    else:
        plt.show()
    plt.close(fig)


# ============================================================
#  7. ATTRIBUTION ANATOMIQUE
# ============================================================

def assign_anatomy(seg_map, img_orig):
    """Attribution labels anatomiques IRM T2."""
    H, W   = img_orig.shape
    cy, cx = H / 2, W / 2
    n_segs = int(seg_map.max()) + 1

    props = []
    for k in range(n_segs):
        m = seg_map == k
        n = m.sum()
        if n == 0:
            props.append({
                'k': k, 'n': 0,
                'mean_i': 0,
                'cy': cy, 'cx': cx,
                'dist_c': 0})
            continue
        ys, xs = np.where(m)
        my = float(ys.mean())
        mx = float(xs.mean())
        dc = np.sqrt(((my-cy)/H)**2 +
                     ((mx-cx)/W)**2)
        props.append({
            'k': k, 'n': n,
            'mean_i': float(img_orig[m].mean()),
            'cy': my, 'cx': mx,
            'dist_c': dc})

    rem = list(props)

    def pop_best(fn):
        b = min(rem, key=fn)
        rem.remove(b)
        return b['k']

    fond_k = pop_best(lambda p: p['mean_i'])
    sac_k  = pop_best(
        lambda p: -p['mean_i']*3 + p['dist_c']*5)
    disc_k = pop_best(
        lambda p: -p['mean_i']*2 +
                  abs(p['cx']-cx)/W*4 +
                  max(0, p['cy']-cy)/H*3)
    emin_k = pop_best(
        lambda p: p['mean_i']*2 +
                  abs(p['cx']-cx)/W*4 -
                  max(0, p['cy']-cy)/H*3)

    gauche = sorted(
        [p for p in rem if p['cx'] < cx],
        key=lambda p: p['cy'])
    droite = sorted(
        [p for p in rem if p['cx'] >= cx],
        key=lambda p: p['cy'])

    def assign_muscles(sl, side):
        res = {}
        ip = 4 if side == 'G' else 5
        im = 6 if side == 'G' else 7
        ie = 8 if side == 'G' else 9
        n  = len(sl)
        if n == 0:
            return res
        if n == 1:
            res[sl[0]['k']] = ip
        elif n == 2:
            res[sl[0]['k']] = ip
            res[sl[1]['k']] = ie
        else:
            res[sl[0]['k']] = ip
            rest = sorted(
                sl[1:],
                key=lambda p: abs(p['cx']-cx))
            res[rest[0]['k']] = im
            for p in rest[1:]:
                res[p['k']] = ie
        return res

    assignment = {
        fond_k: 0, disc_k: 1,
        sac_k:  2, emin_k: 3}
    assignment.update(assign_muscles(gauche, 'G'))
    assignment.update(assign_muscles(droite, 'D'))

    anat = np.zeros((H, W), dtype=np.int8)
    for k in range(n_segs):
        anat[seg_map == k] = assignment.get(k, 0)

    for idx in range(1, N_REGIONS):
        m = anat == idx
        m = morphology.remove_small_objects(
            m, min_size=20)
        m = binary_closing(m, morphology.disk(2))
        anat[anat == idx] = 0
        anat[m] = idx

    unknown = anat == 0
    if unknown.any():
        _, idx_map = distance_transform_edt(
            unknown, return_indices=True)
        filled = anat[idx_map[0], idx_map[1]]
        anat[unknown] = filled[unknown]

    return anat


# ============================================================
#  8. ANALYSE TEXTURE MANNIL
# ============================================================

def extract_texture_mannil(image, mask):
    """Features texture Mannil 2018."""
    pixels = image[mask].astype(np.float32)
    if len(pixels) < 20:
        return None

    counts, _ = np.histogram(
        pixels, bins=256, range=(0, 1))
    probs = counts / (counts.sum() + 1e-8)
    pnz   = probs[probs > 0]

    feat = {
        'n_pixels'      : int(len(pixels)),
        'hist_mean'     : float(np.mean(pixels)),
        'hist_variance' : float(np.var(pixels)),
        'hist_std'      : float(np.std(pixels)),
        'hist_skewness' : float(skew(pixels)),
        'hist_kurtosis' : float(kurtosis(pixels)),
        'hist_entropy'  : float(
            -np.sum(pnz * np.log2(pnz))),
        'hist_p10'      : float(
            np.percentile(pixels, 10)),
        'hist_p25'      : float(
            np.percentile(pixels, 25)),
        'hist_p50'      : float(
            np.percentile(pixels, 50)),
        'hist_p75'      : float(
            np.percentile(pixels, 75)),
        'hist_p90'      : float(
            np.percentile(pixels, 90)),
    }

    try:
        img_u8 = (image * 255).astype(np.uint8)
        glcm   = graycomatrix(
            img_u8, distances=[1, 2, 3],
            angles=[0, np.pi/4,
                    np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True, normed=True)
        feat['glcm_contrast']    = float(
            graycoprops(glcm, 'contrast').mean())
        feat['glcm_energy']      = float(
            graycoprops(glcm, 'energy').mean())
        feat['glcm_homogeneity'] = float(
            graycoprops(glcm, 'homogeneity').mean())
        feat['glcm_correlation'] = float(
            graycoprops(glcm, 'correlation').mean())
        p_nz = glcm[:, :, 0, 0]
        p_nz = p_nz[p_nz > 0]
        feat['glcm_entropy'] = float(
            -np.sum(p_nz * np.log2(p_nz + 1e-10)))
    except Exception:
        for k in ['glcm_contrast', 'glcm_energy',
                  'glcm_homogeneity',
                  'glcm_correlation',
                  'glcm_entropy']:
            feat[k] = 0.0

    return feat


# ============================================================
#  9. VISUALISATION
# ============================================================

def make_color_map(anat_map):
    H, W = anat_map.shape
    c    = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (_, color) in REGIONS.items():
        c[anat_map == idx] = color
    return c


def visualize(img_orig, seg_stego, anat_map,
              stego_codes, all_feat,
              img_name, save_path=None):
    """Figure complète STEGO."""
    from scipy.ndimage import zoom as zm

    fig = plt.figure(figsize=(28, 16),
                     facecolor='black')
    fig.suptitle(
        f"DINOv2 seul + Texture Mannil 2018 "
        f"— {img_name}",
        color='white', fontsize=13,
        fontweight='bold')

    img01     = np.clip(img_orig, 0, 1)
    color_seg = make_color_map(anat_map)
    H, W      = img_orig.shape

    # Carte PCA des codes STEGO
    h_p = w_p = int(len(stego_codes) ** 0.5)
    pca        = PCA(n_components=3, random_state=42)
    codes_pca  = pca.fit_transform(stego_codes)
    codes_rgb  = codes_pca.reshape(h_p, w_p, 3)
    # Normalisation pour affichage
    for c in range(3):
        ch = codes_rgb[:, :, c]
        codes_rgb[:, :, c] = (
            ch - ch.min()) / (
            ch.max() - ch.min() + 1e-8)
    zh = H / h_p
    zw = W / w_p
    codes_full = zm(codes_rgb,
                    (zh, zw, 1), order=1)

    gs1 = fig.add_gridspec(
        1, 5, left=0.01, right=0.99,
        top=0.92, bottom=0.60, wspace=0.05)
    gs2 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.56, bottom=0.30, wspace=0.12)
    gs3 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.26, bottom=0.01, wspace=0.12)

    # P1 : IRM originale
    ax = fig.add_subplot(gs1[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('IRM T2 originale',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # P2 : Espace DINOv2 (PCA RGB)
    ax = fig.add_subplot(gs1[1])
    ax.imshow(np.clip(codes_full, 0, 1))
    ax.set_title('Espace DINOv2\n(PCA RGB)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # P3 : Segmentation DINOv2 brute
    ax = fig.add_subplot(gs1[2])
    ax.imshow(seg_stego, cmap='tab10')
    ax.set_title('K-Means\nsur features DINOv2',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # P4 : Segmentation anatomique
    ax = fig.add_subplot(gs1[3])
    ax.imshow(color_seg)
    pts = [
        mpatches.Patch(
            color=np.array(REGIONS[j][1]) / 255,
            label=REGIONS[j][0])
        for j in range(N_REGIONS)]
    ax.legend(
        handles=pts,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.28),
        ncol=5, fontsize=5.5,
        facecolor='#222',
        labelcolor='white',
        framealpha=0.85)
    ax.set_title('Segmentation\nanatomique DINOv2',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # P5 : Overlay + contours
    ax = fig.add_subplot(gs1[4])
    overlay = np.stack([img01] * 3, axis=-1)
    c_f     = color_seg.astype(np.float32) / 255
    ov      = np.clip(0.45*overlay + 0.55*c_f, 0, 1)
    ax.imshow(ov)
    from skimage import segmentation as sg
    for idx in range(1, N_REGIONS):
        m = anat_map == idx
        c = np.array(REGIONS[idx][1]) / 255
        if m.sum() > 0:
            bd = sg.find_boundaries(m, mode='outer')
            ax.contour(bd, colors=[c],
                       linewidths=0.8)
    ax.set_title('Overlay + contours',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Histogrammes + tableaux
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
            ax_h.axvline(
                feat['hist_mean'],
                color='white', linestyle='--',
                linewidth=1.0)
        else:
            ax_h.text(0.5, 0.5, 'vide',
                      ha='center', va='center',
                      color='gray', fontsize=7,
                      transform=ax_h.transAxes)
        ax_h.set_title(name, color=color,
                       fontsize=6,
                       fontweight='bold')
        ax_h.tick_params(colors='white',
                         labelsize=5)
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
            rows = [
                [lbl, f"{feat.get(k, 0):.3f}"]
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
                        '#2a2a2a' if r % 2 == 0
                        else '#111')
                    cell.set_text_props(
                        color='white')
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


# ============================================================
#  10. MAIN
# ============================================================

if __name__ == '__main__':

    import torch

    device = ('cuda'
              if torch.cuda.is_available()
              else 'cpu')
    print(f"[INFO] Device : {device}")

    dino_model = load_dino(device)
    print(f"[INFO] Modèle : DINOv2 vits14 seul "
          f"(sans STEGO)")

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
    print(f"[INFO] → {OUTPUT_FOLDER}\n")

    all_rows = []

    for img_path in image_paths:
        name = img_path.stem
        print(f"\n{'='*60}")
        print(f"[IMAGE] {name}")
        print(f"{'='*60}")

        # Prétraitement
        try:
            img_orig, img_clahe, tensor, sz = \
                preprocess(img_path)
        except Exception as e:
            print(f"  [ERREUR] {e}")
            continue

        # Features DINO
        try:
            feat_map = extract_dino_features(
                dino_model, tensor, device)
        except Exception as e:
            print(f"  [ERREUR DINO] {e}")
            continue

        # DINOv2 seul : segmentation directe
        try:
            seg_stego, stego_codes = \
                dinov2_segment(
                    feat_map, img_orig)
        except Exception as e:
            print(f"  [ERREUR DINOv2] {e}")
            continue

        # Attribution anatomique
        try:
            anat_map = assign_anatomy(
                seg_stego, img_orig)
        except Exception as e:
            print(f"  [ERREUR anatomie] {e}")
            continue

        # Analyse texture Mannil
        all_feat = {}
        print("  Texture Mannil :")
        for idx in range(N_REGIONS):
            mask_r = anat_map == idx
            feat   = extract_texture_mannil(
                img_orig, mask_r)
            all_feat[idx] = feat
            rname  = REGIONS[idx][0]
            if feat:
                print(
                    f"    {rname:22s} "
                    f"{feat['n_pixels']:5d}px "
                    f"mean={feat['hist_mean']:.3f}")
            else:
                print(f"    {rname:22s} → vide")

        # Visualisation principale
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_dinov2_only_mannil.png")
        visualize(img_orig, seg_stego,
                  anat_map, stego_codes,
                  all_feat, name,
                  save_path=fig_path)

        # Visualisation espace STEGO
        stego_fig = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_stego_space.png")
        visualize_stego_codes(
            stego_codes, seg_stego,
            img_orig, save_path=stego_fig)

        # Sauvegardes
        np.save(os.path.join(
            OUTPUT_FOLDER,
            f"{name}_anat_map.npy"), anat_map)
        np.save(os.path.join(
            OUTPUT_FOLDER,
            f"{name}_dinov2_codes.npy"),
            stego_codes)
        Image.fromarray(
            make_color_map(anat_map)).save(
            os.path.join(OUTPUT_FOLDER,
                         f"{name}_color_seg.png"))

        # CSV
        row = {'patient': name}
        for idx in range(N_REGIONS):
            rname = REGIONS[idx][0].replace(
                ' ', '_')
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
            'all_patients_dinov2_only_mannil.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[CSV] → {csv_path}")
        print(f"  {df.shape[0]} patients × "
              f"{df.shape[1]} features")

        print("\n[RÉSUMÉ] hist_mean/région :")
        for idx in range(N_REGIONS):
            rname = REGIONS[idx][0]
            col   = (f"{rname.replace(' ', '_')}"
                     f"_hist_mean")
            if col in df.columns:
                print(
                    f"  {rname:25s} "
                    f"{df[col].mean():.4f} "
                    f"± {df[col].std():.4f}")

    print(f"\n[TERMINÉ] → {OUTPUT_FOLDER}")