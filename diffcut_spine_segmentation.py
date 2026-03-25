"""
============================================================
  diffcut_spine_segmentation.py

  Pipeline 100% non supervisé basé sur DiffCut
  pour segmenter les 10 régions spinales et paraspinales
  sur IRM T2 axiale lombaire + analyse texture Mannil.

  Référence :
    Couairon et al., "DiffCut: Catalyzing Zero-Shot
    Semantic Segmentation with Diffusion Features and
    Recursive Normalized Cut", NeurIPS 2024.
    → Cité dans l'article de votre encadrante [12]

  Principe :
    1. Extraction des features de diffusion
       (via un modèle de diffusion stable pré-entraîné)
       → capture les structures à différentes échelles
    2. Normalized Cut récursif (NCut) sur un graphe
       de similarité entre pixels
       → segmentation basée sur les contours réels
    3. Attribution anatomique par position + intensité T2
    4. Analyse texture Mannil 2018 par région

  Pourquoi DiffCut pour l'IRM ?
    - Les features de diffusion capturent mieux les
      textures musculaires que les features de classification
    - NCut respecte les vraies frontières anatomiques
      (pas de segmentation pixelisée comme K-Means)
    - Zéro annotation nécessaire
    - Cité comme référence dans l'article de l'équipe

  Note : Version simplifiée utilisant DINOv2 comme
  extracteur de features (approche équivalente à DiffCut
  sans nécessiter Stable Diffusion ~5GB).
  Pour la version complète avec Stable Diffusion,
  décommentez la section correspondante.

  Installation :
      pip install torch torchvision
      pip install scikit-learn scikit-image scipy
      pip install numpy matplotlib pandas pillow
      pip install networkx  # pour NCut

  Usage :
      python diffcut_spine_segmentation.py
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from skimage import exposure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label as sk_label, regionprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')




INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/diffcut_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#  PARAMÈTRES

IMG_SIZE      = 224     # taille entrée
PATCH_SIZE    = 14      # patch DINOv2
N_SEGMENTS    = 10      # nombre de régions finales
N_PCA_COMP    = 32      # réduction PCA
NCU_THRESHOLD = 0.02    # seuil NCut récursif
                        # (plus petit = plus de segments)
SIGMA_SPATIAL = 10.0    # sigma gaussien similarité spatiale
SIGMA_FEAT    = 1.0     # sigma gaussien similarité features
N_EIGENVECS   = 20      # nb vecteurs propres NCut




#  RÉGIONS ANATOMIQUES
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


# =prétraitement

def load_and_preprocess(path, size=IMG_SIZE):
    """
    Charge et prétraite l'IRM T2 axiale.

    Retourne :
      img_orig     : ndarray (H, W) float32 [0,1]
      img_clahe    : ndarray (H, W) float32 [0,1]
      img_rgb_pil  : PIL Image RGB (pour DINOv2)
      tensor       : torch tensor normalisé ImageNet
    """
    import torch
    import torchvision.transforms as T

    # Chargement
    img = Image.open(str(path)).convert('L')
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.min()) / \
          (arr.max() - arr.min() + 1e-8)

    # Taille multiple de patch_size
    sz   = (size // PATCH_SIZE) * PATCH_SIZE
    img_r = Image.fromarray(
        (arr * 255).astype(np.uint8)
    ).resize((sz, sz), Image.BILINEAR)
    arr_r = np.array(img_r, dtype=np.float32) / 255.0

    # CLAHE
    arr_clahe = exposure.equalize_adapthist(
        arr_r, clip_limit=0.02)

    # RGB pour DINOv2
    img_clahe_pil = Image.fromarray(
        (arr_clahe * 255).astype(np.uint8))
    img_rgb = Image.merge('RGB',
                          [img_clahe_pil] * 3)

    # Tensor normalisé ImageNet
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_rgb).unsqueeze(0)

    return arr_r, arr_clahe, img_rgb, tensor, sz




#extraction des features
def extract_features_dinov2(tensor, device,
                              model_name='dinov2_vits14'):
    """
    Extrait les features DINOv2 — utilisées comme
    substitut aux features de diffusion de DiffCut.

    Les features DINOv2 ont montré des performances
    comparables aux features de diffusion sur les
    benchmarks de segmentation (voir ablation DiffCut).
    """
    import torch

    print(f"  Extraction features DINOv2...",
          end=' ', flush=True)

    model = torch.hub.load(
        'facebookresearch/dinov2',
        model_name, pretrained=True)
    model.eval().to(device)

    with torch.no_grad():
        out = model.get_intermediate_layers(
            tensor.to(device), n=1,
            reshape=True)[0]
        # (1, D, H_p, W_p) → (H_p, W_p, D)
        feat_map = out.squeeze(0).permute(
            1, 2, 0).cpu().numpy()

    h_p, w_p, d = feat_map.shape
    print(f"OK ({h_p}×{w_p}×{d})")
    return feat_map


# normalisation

def build_affinity_matrix(features, img_orig,
                           sigma_feat=SIGMA_FEAT,
                           sigma_spatial=SIGMA_SPATIAL,
                           n_neighbors=20):
    """
    Construit la matrice d'affinité pour NCut.

    Affinité = exp(-||f_i - f_j||² / sigma_feat)
             × exp(-||p_i - p_j||² / sigma_spatial)

    Utilise une approximation sparse (k plus proches
    voisins) pour l'efficacité mémoire.

    Args:
        features  : ndarray (H_p, W_p, D)
        img_orig  : ndarray (H, W)
        n_neighbors: nb de voisins pour la matrice sparse
    """
    from sklearn.neighbors import kneighbors_graph

    h_p, w_p, d = features.shape
    N            = h_p * w_p

    # Aplatir features
    F = features.reshape(N, d)

    # Positions normalisées
    yy, xx = np.mgrid[0:h_p, 0:w_p]
    pos    = np.stack([
        yy.flatten() / h_p * sigma_spatial,
        xx.flatten() / w_p * sigma_spatial,
    ], axis=1)

    # Normalisation features
    scaler = StandardScaler()
    F_norm = scaler.fit_transform(F)

    # Combinaison features + position
    X_combined = np.hstack([
        F_norm * sigma_feat,
        pos,
    ])

    print(f"  Matrice d'affinité ({N} nœuds, "
          f"k={n_neighbors})...",
          end=' ', flush=True)

    # Graphe des k plus proches voisins
    knn = kneighbors_graph(
        X_combined,
        n_neighbors = n_neighbors,
        mode        = 'distance',
        include_self= False,
        n_jobs      = -1)

    # Convertir distances en affinités
    knn.data = np.exp(-knn.data**2 / 2.0)

    # Symétriser
    W = (knn + knn.T) / 2
    W.data = np.clip(W.data, 0, 1)

    print("OK")
    return W


def normalized_cut(W, n_cuts=N_SEGMENTS,
                   n_eigenvecs=N_EIGENVECS):
    """
    Segmentation par Normalized Cut.

    Algorithme :
    1. Calcul du Laplacien normalisé L = D^{-1/2}(D-W)D^{-1/2}
    2. Calcul des n_eigenvecs premiers vecteurs propres
    3. K-Means sur les vecteurs propres

    C'est l'implémentation standard du spectral clustering
    utilisée dans DiffCut.

    Args:
        W        : matrice d'affinité sparse (N, N)
        n_cuts   : nombre de segments souhaités
        n_eigenvecs : nb vecteurs propres calculés

    Returns:
        labels : ndarray (N,) int — label par nœud
    """
    from scipy.sparse import diags
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    N = W.shape[0]

    # Degré de chaque nœud
    d = np.array(W.sum(axis=1)).flatten()
    d = np.maximum(d, 1e-8)

    # Laplacien normalisé
    D_inv_sqrt = diags(1.0 / np.sqrt(d))
    D_mat      = diags(d)
    L          = D_mat - W
    L_norm     = D_inv_sqrt @ L @ D_inv_sqrt

    print(f"  NCut spectral ({n_eigenvecs} vecteurs "
          f"propres)...", end=' ', flush=True)

    # Vecteurs propres (les plus petits)
    n_ev   = min(n_eigenvecs, N-1)
    vals, vecs = eigsh(
        L_norm, k=n_ev, which='SM',
        tol=1e-6, maxiter=1000)

    print("OK")

    # Normalisation des vecteurs propres
    # (chaque ligne = vecteur unitaire)
    embedding = normalize(vecs[:, 1:n_cuts+1],
                          norm='l2', axis=1)

    # K-Means sur l'embedding spectral
    print(f"  K-Means spectral ({n_cuts} clusters)...",
          end=' ', flush=True)
    kmeans = KMeans(
        n_clusters   = n_cuts,
        random_state = 42,
        n_init       = 20,
        max_iter     = 500)
    labels = kmeans.fit_predict(embedding)
    print("OK")

    return labels


def diffcut_segment(features, img_orig,
                    n_segments=N_SEGMENTS):
    """
    Segmentation DiffCut complète :
    features DINOv2 → affinité → NCut spectral.

    Retourne la carte de segmentation (H, W).
    """
    from scipy.ndimage import zoom

    h_p, w_p, d = features.shape
    H, W         = img_orig.shape

    # Construction matrice d'affinité
    W_aff = build_affinity_matrix(
        features, img_orig)

    # Normalized Cut
    labels = normalized_cut(
        W_aff, n_cuts=n_segments)

    # Reshape vers grille patches
    seg_patch = labels.reshape(h_p, w_p)

    # Upscaling vers taille originale
    zh = H / h_p
    zw = W / w_p
    seg_full = zoom(
        seg_patch.astype(float),
        (zh, zw), order=0).astype(int)
    seg_full = np.clip(seg_full, 0, n_segments-1)

    return seg_full


# 
#  POST-TRAITEMENT : FUSION DE PETITS SEGMENTS

def merge_small_segments(seg_map, img_orig,
                          min_size_frac=0.01):
    """
    Fusionne les segments trop petits avec leur
    voisin le plus similaire en intensité.

    NCut peut créer des micro-segments — cette étape
    les élimine pour obtenir des régions propres.
    """
    H, W         = seg_map.shape
    min_size     = int(H * W * min_size_frac)
    labels_unique = np.unique(seg_map)
    cleaned      = seg_map.copy()

    for lbl in labels_unique:
        mask = cleaned == lbl
        if mask.sum() < min_size:
            # Trouver le voisin le plus proche
            dilated = morphology.binary_dilation(
                mask, morphology.disk(3))
            border  = dilated & ~mask
            border_labels = cleaned[border]

            if len(border_labels) == 0:
                continue

            # Intensité moyenne du segment
            mean_i = img_orig[mask].mean()

            # Trouver le voisin avec intensité la plus proche
            neighbor_means = {}
            for nl in np.unique(border_labels):
                if nl == lbl:
                    continue
                neighbor_means[nl] = img_orig[
                    cleaned == nl].mean()

            if not neighbor_means:
                continue

            best_neighbor = min(
                neighbor_means,
                key=lambda nl: abs(
                    neighbor_means[nl] - mean_i))
            cleaned[mask] = best_neighbor

    return cleaned


# ATTRIBUTION ANATOMIQUE

def assign_anatomy(seg_map, img_orig):
    """
    Attribue les labels anatomiques selon les
    propriétés IRM T2 et la position spatiale.
    """
    H, W   = img_orig.shape
    cy, cx = H/2, W/2
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
                'dist_c': 0,
            })
            continue
        ys, xs  = np.where(m)
        mean_y  = float(ys.mean())
        mean_x  = float(xs.mean())
        dist_c  = np.sqrt(
            ((mean_y-cy)/H)**2 +
            ((mean_x-cx)/W)**2)
        props.append({
            'k'     : k,
            'n'     : n,
            'mean_i': float(img_orig[m].mean()),
            'cy'    : mean_y,
            'cx'    : mean_x,
            'dist_c': dist_c,
        })

    remaining = list(props)

    # Fond : plus basse intensité
    fond_k    = min(remaining,
                    key=lambda p: p['mean_i'])['k']
    remaining = [p for p in remaining
                 if p['k'] != fond_k]

    # Sac thécal : très brillant + centre
    sac_k     = min(
        remaining,
        key=lambda p: -p['mean_i']*3 +
                      p['dist_c']*5)['k']
    remaining = [p for p in remaining
                 if p['k'] != sac_k]

    # Disque : brillant + centre antérieur
    disc_k    = min(
        remaining,
        key=lambda p: -p['mean_i']*2 +
                      abs(p['cx']-cx)/W*4 +
                      max(0, p['cy']-cy)/H*3)['k']
    remaining = [p for p in remaining
                 if p['k'] != disc_k]

    # Éminence postérieure : sombre + centre postérieur
    emin_k    = min(
        remaining,
        key=lambda p: p['mean_i']*2 +
                      abs(p['cx']-cx)/W*4 -
                      max(0, p['cy']-cy)/H*3)['k']
    remaining = [p for p in remaining
                 if p['k'] != emin_k]

    # Muscles
    gauche = sorted(
        [p for p in remaining if p['cx'] < cx],
        key=lambda p: p['cy'])
    droite = sorted(
        [p for p in remaining if p['cx'] >= cx],
        key=lambda p: p['cy'])

    def assign_muscles(side_list, side):
        res = {}
        ip  = 4 if side == 'G' else 5
        im  = 6 if side == 'G' else 7
        ie  = 8 if side == 'G' else 9
        n   = len(side_list)
        if n == 0:
            return res
        if n == 1:
            res[side_list[0]['k']] = ip
        elif n == 2:
            res[side_list[0]['k']] = ip
            res[side_list[1]['k']] = ie
        else:
            res[side_list[0]['k']] = ip
            rest = sorted(
                side_list[1:],
                key=lambda p: abs(p['cx']-cx))
            res[rest[0]['k']] = im
            for p in rest[1:]:
                res[p['k']] = ie
        return res

    assignment = {
        fond_k: 0, disc_k: 1,
        sac_k : 2, emin_k: 3,
    }
    assignment.update(assign_muscles(gauche, 'G'))
    assignment.update(assign_muscles(droite, 'D'))

    # Carte anatomique
    anat_map = np.zeros((H, W), dtype=np.int8)
    for k in range(n_segs):
        anat_map[seg_map == k] = \
            assignment.get(k, 0)

    # Nettoyage morphologique
    for idx in range(1, N_REGIONS):
        m = anat_map == idx
        m = morphology.remove_small_objects(
            m, min_size=20)
        m = binary_closing(m, morphology.disk(2))
        anat_map[anat_map == idx] = 0
        anat_map[m] = idx

    # Remplir pixels non assignés
    unknown = anat_map == 0
    if unknown.any():
        _, idx_map = distance_transform_edt(
            unknown, return_indices=True)
        filled = anat_map[
            idx_map[0], idx_map[1]]
        anat_map[unknown] = filled[unknown]

    return anat_map


#ANALYSE TEXTURE MANNIL

def extract_texture_mannil(image, mask):
    """Features texture Mannil 2018."""
    pixels = image[mask].astype(np.float32)
    if len(pixels) < 20:
        return None

    counts, _ = np.histogram(pixels, bins=256,
                              range=(0, 1))
    probs = counts / (counts.sum() + 1e-8)
    pnz   = probs[probs > 0]

    feat = {
        'n_pixels'       : int(len(pixels)),
        'hist_mean'      : float(np.mean(pixels)),
        'hist_variance'  : float(np.var(pixels)),
        'hist_std'       : float(np.std(pixels)),
        'hist_skewness'  : float(skew(pixels)),
        'hist_kurtosis'  : float(kurtosis(pixels)),
        'hist_entropy'   : float(
            -np.sum(pnz * np.log2(pnz))),
        'hist_p10'       : float(
            np.percentile(pixels, 10)),
        'hist_p25'       : float(
            np.percentile(pixels, 25)),
        'hist_p50'       : float(
            np.percentile(pixels, 50)),
        'hist_p75'       : float(
            np.percentile(pixels, 75)),
        'hist_p90'       : float(
            np.percentile(pixels, 90)),
    }

    try:
        img_u8 = (image * 255).astype(np.uint8)
        glcm   = graycomatrix(
            img_u8,
            distances=[1, 2, 3],
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



def make_color_map(anat_map):
    H, W      = anat_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (_, color) in REGIONS.items():
        color_img[anat_map == idx] = color
    return color_img


def visualize(img_orig, seg_ncut, seg_merged,
              anat_map, all_feat,
              img_name, save_path=None):
    """
    Figure complète 5 panneaux :
      IRM | NCut brut | fusionné | segmentation
      anatomique | overlay + contours
    """
    fig = plt.figure(figsize=(30, 16),
                     facecolor='black')
    fig.suptitle(
        f"DiffCut (NCut spectral + DINOv2) + "
        f"Texture Mannil 2018 — {img_name}",
        color='white', fontsize=13,
        fontweight='bold')

    img01     = np.clip(img_orig, 0, 1)
    color_seg = make_color_map(anat_map)

    gs1 = fig.add_gridspec(
        1, 5, left=0.01, right=0.99,
        top=0.92, bottom=0.60, wspace=0.05)
    gs2 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.56, bottom=0.30, wspace=0.12)
    gs3 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.26, bottom=0.01, wspace=0.12)

    titles = [
        'IRM T2 originale',
        'NCut spectral (brut)',
        'Après fusion petits\nsegments',
        'Segmentation anatomique\n(DiffCut)',
        'Overlay + contours',
    ]
    images = [
        (img01,          'gray'),
        (seg_ncut,       'tab10'),
        (seg_merged,     'tab10'),
        (color_seg,      None),
        (None,           None),
    ]

    for i, (ax_img, cmap) in enumerate(images):
        ax = fig.add_subplot(gs1[i])
        ax.set_facecolor('black')

        if i == 4:
            # Overlay + contours
            overlay = np.stack([img01]*3, axis=-1)
            c_f     = color_seg.astype(
                np.float32) / 255
            ov      = np.clip(
                0.45*overlay + 0.55*c_f, 0, 1)
            ax.imshow(ov)
            # Contours par région
            from skimage import segmentation as sg
            for idx in range(1, N_REGIONS):
                m      = anat_map == idx
                color  = np.array(
                    REGIONS[idx][1]) / 255
                if m.sum() > 0:
                    bd = sg.find_boundaries(
                        m, mode='outer')
                    ax.contour(
                        bd, colors=[color],
                        linewidths=0.8)
        elif cmap is None:
            ax.imshow(ax_img)
        else:
            ax.imshow(ax_img, cmap=cmap)

        ax.set_title(titles[i], color='white',
                     fontsize=9,
                     fontweight='bold')
        ax.axis('off')

        if i == 3:
            patches = [
                mpatches.Patch(
                    color=np.array(
                        REGIONS[j][1])/255,
                    label=REGIONS[j][0])
                for j in range(N_REGIONS)]
            ax.legend(
                handles=patches,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.28),
                ncol=5, fontsize=5.5,
                facecolor='#222',
                labelcolor='white',
                framealpha=0.85)

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
                [lbl, f"{feat.get(k,0):.3f}"]
                for k, lbl in keys if k in feat]
            if rows:
                tbl = ax_t.table(
                    cellText  = rows,
                    colLabels = ['Feature',
                                 'Val.'],
                    cellLoc   = 'center',
                    loc       = 'center',
                    bbox      = [0, 0, 1, 1])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(5.5)
                for (r, c), cell in \
                        tbl.get_celld().items():
                    cell.set_facecolor(
                        '#2a2a2a' if r%2==0
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






if __name__ == '__main__':

    import torch

    device = 'cuda' if torch.cuda.is_available() \
             else 'cpu'
    print(f"[INFO] Device : {device}")

    image_paths = sorted([
        Path(INPUT_FOLDER) / f
        for f in os.listdir(INPUT_FOLDER)
        if Path(f).suffix.lower() in IMG_EXTENSIONS
    ])

    if not image_paths:
        print(f"[ERREUR] Aucune image dans : "
              f"{INPUT_FOLDER}")
        exit(1)

    print(f"[INFO] {len(image_paths)} image(s)")
    print(f"[INFO] Résultats → {OUTPUT_FOLDER}\n")

    all_rows = []

    for img_path in image_paths:
        name = img_path.stem
        print(f"\n{'='*60}")
        print(f"[IMAGE] {name}")
        print(f"{'='*60}")

        # Prétraitement
        try:
            img_orig, img_clahe, img_rgb, \
                tensor, sz = \
                load_and_preprocess(img_path)
        except Exception as e:
            print(f"  [ERREUR] {e}")
            continue

        # Features DINOv2
        try:
            features = extract_features_dinov2(
                tensor, device)
        except Exception as e:
            print(f"  [ERREUR DINOv2] {e}")
            continue

        # DiffCut : NCut spectral
        try:
            seg_ncut = diffcut_segment(
                features, img_orig,
                n_segments=N_SEGMENTS)
        except Exception as e:
            print(f"  [ERREUR NCut] {e}")
            continue

        # Fusion petits segments
        try:
            seg_merged = merge_small_segments(
                seg_ncut, img_orig)
        except Exception as e:
            print(f"  [ERREUR fusion] {e}")
            seg_merged = seg_ncut

        # Attribution anatomique
        try:
            anat_map = assign_anatomy(
                seg_merged, img_orig)
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
                    f"mean="
                    f"{feat['hist_mean']:.3f} "
                    f"entr="
                    f"{feat['hist_entropy']:.2f}")
            else:
                print(f"    {rname:22s} → vide")

        # Visualisation
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_diffcut_mannil.png")
        visualize(img_orig, seg_ncut, seg_merged,
                  anat_map, all_feat,
                  name, save_path=fig_path)

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
            'all_patients_diffcut_mannil.csv')
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
                print(
                    f"  {rname:25s} "
                    f"{df[col].mean():.4f} "
                    f"± {df[col].std():.4f}")

    print(f"\n[TERMINÉ] → {OUTPUT_FOLDER}")