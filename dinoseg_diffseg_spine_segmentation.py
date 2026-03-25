"""
============================================================
  dinoseg_diffseg_spine_segmentation.py

  Pipeline 100% non supervisé combinant DinoSeg et DiffSeg
  pour segmenter les 10 régions spinales et paraspinales
  sur IRM T2 axiale lombaire + analyse texture Mannil.

  Corrections v2 :
    - get_intermediate_layers : reshape=False
      (compatibilité DINOv2 toutes versions)
    - attn_map initialisé par défaut avant la boucle
    - seg_dino / seg_diff initialisés par défaut

  Installation :
      pip install torch torchvision
      pip install scikit-learn scikit-image scipy
      pip install numpy matplotlib pandas pillow

  Usage :
      python dinoseg_diffseg_spine_segmentation.py
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
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from skimage import exposure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler,
                                    normalize)
import warnings
warnings.filterwarnings('ignore')




INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/dinoseg_diffseg_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#paramètres
IMG_SIZE     = 224
PATCH_SIZE   = 14
N_REGIONS    = 10
N_LAYERS     = 4
N_EIGENVECS  = 15
FUSION_ALPHA = 0.6


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



# CHARGEMENT DINOV2

def load_dinov2(device, model_name='dinov2_vits14'):
    import torch
    print(f"[DINOv2] Chargement {model_name}...",
          end=' ', flush=True)
    model = torch.hub.load(
        'facebookresearch/dinov2',
        model_name, pretrained=True)
    model.eval().to(device)
    print(f"OK ({device})")
    return model


# PRÉTRAITEMENT

def preprocess(path, size=IMG_SIZE):
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




#  DINOSEG = EXTRACTION FEATURES 

def extract_dinoseg_features(model, tensor,
                              device,
                              n_layers=N_LAYERS):
    """
    CORRECTION reshape=False :
    Retourne (N_patches, D) → reshape manuel en (H_p, W_p, D)
    """
    import torch

    with torch.no_grad():
        tensor = tensor.to(device)

        # reshape=False → (1, N_patches, D)
        layer_feats = model.get_intermediate_layers(
            tensor, n=n_layers, reshape=False)

        all_feats = []
        h_p = w_p = d = None
        for lf in layer_feats:
            f         = lf.squeeze(0).cpu().numpy()
            n_patches = f.shape[0]
            _d        = f.shape[1]
            _h = _w   = int(n_patches ** 0.5)
            f_map     = f.reshape(_h, _w, _d)
            all_feats.append(f_map)
            h_p, w_p, d = _h, _w, _d

        feat_map = np.mean(all_feats, axis=0)

        # Cartes d'attention
        attn_maps = []
        hooks     = []

        def make_hook():
            def hook_fn(module, inp, out):
                attn = out.detach()
                if isinstance(attn, tuple):
                    attn = attn[0]
                if attn.dim() == 4:
                    cls_attn = attn[0, :, 0, 1:]
                    attn_maps.append(
                        cls_attn.cpu().numpy())
            return hook_fn

        n_blocks = len(model.blocks)
        for i in range(
                max(0, n_blocks - n_layers),
                n_blocks):
            h = model.blocks[i].attn\
                     .register_forward_hook(
                make_hook())
            hooks.append(h)

        _ = model(tensor)
        for h in hooks:
            h.remove()

    if attn_maps and h_p is not None:
        attn_concat = np.concatenate(
            attn_maps, axis=0)
        attn_mean = attn_concat.mean(axis=0)
        n_p = attn_mean.shape[0]
        _h = _w = int(n_p ** 0.5)
        if _h * _w == n_p:
            attn_2d = attn_mean.reshape(_h, _w)
        else:
            attn_2d = np.ones((h_p, w_p))
        attn_2d = (attn_2d - attn_2d.min()) / \
                  (attn_2d.max() -
                   attn_2d.min() + 1e-8)
    else:
        attn_2d = np.ones((h_p or 16, w_p or 16))

    print(f"  DinoSeg features : {feat_map.shape}"
          f", attention : {attn_2d.shape}")
    return feat_map, attn_2d


def dinoseg_segment(feat_map, attn_map,
                    img_orig,
                    n_segments=N_REGIONS,
                    n_eigenvecs=N_EIGENVECS):
    from scipy.ndimage import zoom

    h_p, w_p, d = feat_map.shape
    H, W         = img_orig.shape
    N            = h_p * w_p

    F   = feat_map.reshape(N, d)
    F_n = normalize(F, norm='l2')

    W_cos = F_n @ F_n.T
    W_cos = np.maximum(W_cos, 0)
    np.fill_diagonal(W_cos, 0)

    attn_flat = attn_map.flatten()
    if len(attn_flat) != N:
        from scipy.ndimage import zoom as zm
        zh = h_p / attn_map.shape[0]
        zw = w_p / attn_map.shape[1]
        attn_map = zm(attn_map, (zh, zw), order=1)
        attn_flat = attn_map.flatten()[:N]

    attn_w = np.sqrt(np.outer(
        attn_flat[:N], attn_flat[:N]))
    W_att  = W_cos * attn_w
    np.fill_diagonal(W_att, 0)

    print(f"  DinoSeg SpectralClustering "
          f"({n_segments} segments)...",
          end=' ', flush=True)

    try:
        sc = SpectralClustering(
            n_clusters    = n_segments,
            affinity      = 'precomputed',
            random_state  = 42,
            n_init        = 10,
            assign_labels = 'kmeans')
        labels = sc.fit_predict(W_att)
        print("OK")
    except Exception as e:
        print(f"ERREUR ({e}) → K-Means fallback")
        km     = KMeans(n_clusters=n_segments,
                        random_state=42, n_init=10)
        labels = km.fit_predict(F_n)

    seg_patch = labels.reshape(h_p, w_p)
    zh = H / h_p
    zw = W / w_p
    seg_full = zoom(seg_patch.astype(float),
                    (zh, zw), order=0).astype(int)
    return np.clip(seg_full, 0, n_segments - 1)


# DIFFSEG — EXTRACTION FEATURES 

def extract_diffseg_features(model, tensor, device):
    """
    CORRECTION reshape=False :
    3 niveaux de profondeur DINOv2.
    """
    import torch

    with torch.no_grad():
        tensor   = tensor.to(device)
        n_blocks = len(model.blocks)

        levels = [
            max(0, n_blocks // 4),
            max(0, n_blocks // 2),
            max(0, 3 * n_blocks // 4),
        ]

        # reshape=False → (1, N_patches, D)
        layer_features = model.get_intermediate_layers(
            tensor, n=n_blocks, reshape=False)

        feat_levels = []
        for lvl in levels:
            lf  = layer_features[lvl]
            f   = lf.squeeze(0).cpu().numpy()
            n_p = f.shape[0]
            _d  = f.shape[1]
            _h = _w = int(n_p ** 0.5)
            feat_levels.append(f.reshape(_h, _w, _d))

    print(f"  DiffSeg : 3 niveaux "
          f"{[f.shape for f in feat_levels]}")
    return feat_levels


def diffseg_iterative_merge(feat_levels,
                             img_orig,
                             n_segments=N_REGIONS):
    from scipy.ndimage import zoom

    H, W         = img_orig.shape
    seg_maps_all = []

    for lvl_idx, feat in enumerate(feat_levels):
        h_p, w_p, d = feat.shape
        F   = feat.reshape(-1, d)
        F_n = normalize(F, norm='l2')

        n_comp = min(32, F_n.shape[0]-1,
                     F_n.shape[1])
        if n_comp > 1:
            pca   = PCA(n_components=n_comp,
                        random_state=42)
            F_pca = pca.fit_transform(F_n)
        else:
            F_pca = F_n

        km  = KMeans(n_clusters=n_segments,
                     random_state=42, n_init=10)
        lbl = km.fit_predict(F_pca)
        seg = lbl.reshape(h_p, w_p)

        zh = H / h_p
        zw = W / w_p
        seg_full = zoom(seg.astype(float),
                        (zh, zw), order=0).astype(int)
        seg_maps_all.append(seg_full)
        print(f"    Niveau {lvl_idx+1} : OK")

    print(f"  Consensus iteratif...",
          end=' ', flush=True)

    vote_matrix = np.stack(
        [s.flatten() for s in seg_maps_all], axis=1)
    scaler      = StandardScaler()
    vote_scaled = scaler.fit_transform(
        vote_matrix.astype(float))

    yy, xx = np.mgrid[0:H, 0:W]
    pos    = np.stack([
        yy.flatten() / H,
        xx.flatten() / W,
    ], axis=1) * 0.3

    combined     = np.hstack([vote_scaled, pos])
    km_final     = KMeans(n_clusters=n_segments,
                          random_state=42, n_init=15)
    final_labels = km_final.fit_predict(combined)
    print("OK")
    return final_labels.reshape(H, W), seg_maps_all



#  5. FUSION DINOSEG + DIFFSEG

def fuse_dinoseg_diffseg(seg_dino, seg_diff,
                          img_orig,
                          alpha=FUSION_ALPHA,
                          n_segments=N_REGIONS):
    H, W = img_orig.shape
    print(f"  Fusion DinoSeg(α={alpha}) + "
          f"DiffSeg(α={1-alpha:.1f})...",
          end=' ', flush=True)

    feat_fusion = np.stack([
        seg_dino.flatten().astype(float)
        / n_segments * alpha,
        seg_diff.flatten().astype(float)
        / n_segments * (1 - alpha),
        img_orig.flatten(),
        np.mgrid[0:H, 0:W][0].flatten() / H * 0.3,
        np.mgrid[0:H, 0:W][1].flatten() / W * 0.3,
    ], axis=1)

    scaler    = StandardScaler()
    feat_s    = scaler.fit_transform(feat_fusion)
    km        = KMeans(n_clusters=n_segments,
                       random_state=42, n_init=15)
    labels    = km.fit_predict(feat_s)
    seg_fused = labels.reshape(H, W)

    print("OK")
    return seg_fused


# ATTRIBUTION ANATOMIQUE

def assign_anatomy(seg_map, img_orig):
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
            'cy': my, 'cx': mx, 'dist_c': dc})

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


# ANALYSE TEXTURE MANNIL

def extract_texture_mannil(image, mask):
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



#visualisation

def make_color_map(anat_map):
    H, W = anat_map.shape
    c    = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (_, color) in REGIONS.items():
        c[anat_map == idx] = color
    return c


def visualize(img_orig, attn_map,
              seg_dino, seg_diff,
              seg_fused, anat_map,
              all_feat, img_name,
              save_path=None):

    from scipy.ndimage import zoom as zm

    fig = plt.figure(figsize=(32, 16),
                     facecolor='black')
    fig.suptitle(
        f"DinoSeg + DiffSeg + Texture "
        f"Mannil 2018 — {img_name}",
        color='white', fontsize=13,
        fontweight='bold')

    img01     = np.clip(img_orig, 0, 1)
    color_seg = make_color_map(anat_map)
    H, W      = img_orig.shape

    # Upscale attn_map vers taille image
    if attn_map.shape != (H, W):
        zh = H / attn_map.shape[0]
        zw = W / attn_map.shape[1]
        attn_full = zm(attn_map, (zh, zw), order=1)
    else:
        attn_full = attn_map

    gs1 = fig.add_gridspec(
        1, 6, left=0.01, right=0.99,
        top=0.92, bottom=0.60, wspace=0.05)
    gs2 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.56, bottom=0.30, wspace=0.12)
    gs3 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.26, bottom=0.01, wspace=0.12)

    panels = [
        (img01,     'gray',   'IRM T2\noriginale'),
        (attn_full, 'inferno','Attention\nDINOv2'),
        (seg_dino,  'tab10',  'DinoSeg\n(spectral)'),
        (seg_diff,  'tab10',  'DiffSeg\n(consensus)'),
        (seg_fused, 'tab10',  'Fusion\nDinoSeg+DiffSeg'),
        (color_seg, None,     'Segmentation\nanatomique'),
    ]

    for i, (im, cmap, title) in enumerate(panels):
        ax = fig.add_subplot(gs1[i])
        if cmap:
            ax.imshow(im, cmap=cmap)
        else:
            ax.imshow(im)
            pts = [
                mpatches.Patch(
                    color=np.array(
                        REGIONS[j][1]) / 255,
                    label=REGIONS[j][0])
                for j in range(N_REGIONS)]
            ax.legend(
                handles=pts,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.30),
                ncol=5, fontsize=5,
                facecolor='#222',
                labelcolor='white',
                framealpha=0.85)
        ax.set_title(title, color='white',
                     fontsize=9,
                     fontweight='bold')
        ax.axis('off')

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
#  9. MAIN
# ============================================================

if __name__ == '__main__':

    import torch

    device = ('cuda'
              if torch.cuda.is_available()
              else 'cpu')
    print(f"[INFO] Device : {device}")

    model = load_dinov2(device)

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

        # Valeurs par défaut
        _hp      = IMG_SIZE // PATCH_SIZE
        attn_map = np.ones((_hp, _hp))
        seg_dino = np.zeros(
            (IMG_SIZE, IMG_SIZE), dtype=int)
        seg_diff = np.zeros(
            (IMG_SIZE, IMG_SIZE), dtype=int)

        # Prétraitement
        try:
            img_orig, img_clahe, tensor, sz = \
                preprocess(img_path)
            H, W     = img_orig.shape
            _hp      = H // PATCH_SIZE
            attn_map = np.ones((_hp, _hp))
            seg_dino = np.zeros((H, W), dtype=int)
            seg_diff = np.zeros((H, W), dtype=int)
        except Exception as e:
            print(f"  [ERREUR prétraitement] {e}")
            continue

        # DinoSeg
        print("\n  [DinoSeg]")
        try:
            feat_dino, attn_map = \
                extract_dinoseg_features(
                    model, tensor, device)
            seg_dino = dinoseg_segment(
                feat_dino, attn_map, img_orig)
        except Exception as e:
            print(f"  [ERREUR DinoSeg] {e}")

        # DiffSeg
        print("\n  [DiffSeg]")
        try:
            feat_levels = extract_diffseg_features(
                model, tensor, device)
            seg_diff, _ = diffseg_iterative_merge(
                feat_levels, img_orig)
        except Exception as e:
            print(f"  [ERREUR DiffSeg] {e}")
            seg_diff = seg_dino.copy()

        #Fusion
        print("\n  [Fusion]")
        try:
            seg_fused = fuse_dinoseg_diffseg(
                seg_dino, seg_diff, img_orig)
        except Exception as e:
            print(f"  [ERREUR Fusion] {e}")
            seg_fused = seg_dino.copy()

        #Attribution anatomique
        try:
            anat_map = assign_anatomy(
                seg_fused, img_orig)
        except Exception as e:
            print(f"  [ERREUR anatomie] {e}")
            continue

        # Texture Mannil
        all_feat = {}
        print("\n  Texture Mannil :")
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

        # Visualisation
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_dinoseg_diffseg_mannil.png")
        visualize(
            img_orig, attn_map,
            seg_dino, seg_diff,
            seg_fused, anat_map,
            all_feat, name,
            save_path=fig_path)

        #  Sauvegardes
        np.save(os.path.join(
            OUTPUT_FOLDER,
            f"{name}_anat_map.npy"), anat_map)
        Image.fromarray(
            make_color_map(anat_map)).save(
            os.path.join(OUTPUT_FOLDER,
                         f"{name}_color_seg.png"))

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
            'all_patients_dinoseg_diffseg.csv')
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