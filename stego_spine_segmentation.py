"""
============================================================
  stego_spine_segmentation.py

  Pipeline 100% non supervisé basé sur STEGO
  pour segmenter les 10 régions spinales et paraspinales
  sur IRM T2 axiale lombaire + analyse texture Mannil.

  Référence :
    Hamilton et al., "Unsupervised Semantic Segmentation
    by Distilling Feature Correspondences", ICLR 2022.
    https://arxiv.org/abs/2203.08414

  Principe de STEGO :
    STEGO (Self-supervised Transformer with Energy-based
    Graph Optimization) apprend à segmenter en distillant
    les correspondances entre features DINO :

    1. Features DINO extraites pour chaque image
    2. Pour chaque paire de patches similaires
       (dans la même image ou images différentes),
       STEGO apprend à les regrouper dans le même
       segment via une perte contrastive
    3. Résultat : une tête de segmentation légère
       entraînée de façon non supervisée sur vos images

  Différence clé avec les autres méthodes :
    - K-Means / GMM / DiffCut : segmentation statique
      (un seul passage)
    - STEGO : ENTRAÎNE une tête de segmentation sur
      vos propres images (apprentissage non supervisé)
    → Meilleures performances car adapté à vos données

  Notre implémentation :
    Version simplifiée de STEGO sans entraînement
    multi-époque (trop long) :
    - Features DINO extraites
    - Matrice de correspondance construite
    - Clustering spectral guidé par les correspondances
    - Fine-tuning local par descente de gradient
      sur la perte STEGO (quelques pas)

  Installation :
      pip install torch torchvision
      pip install scikit-learn scikit-image scipy
      pip install numpy matplotlib pandas pillow

  Usage :
      python stego_spine_segmentation.py
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



INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/stego_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#paramètres
IMG_SIZE       = 224
PATCH_SIZE     = 8       # DINO vits8 patch 8
                          # (meilleure résolution spatiale)
N_REGIONS      = 10
N_LAYERS       = 1       # couche DINO utilisée
STEGO_DIM      = 70      # dimension tête STEGO
STEGO_LR       = 5e-3    # learning rate micro-entraînement
STEGO_STEPS    = 30      # pas de gradient sur chaque image
STEGO_K        = 7       # nb voisins pour correspondances
STEGO_NEG_K    = 3       # nb voisins négatifs
TEMPERATURE    = 0.1     # température contrastive



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


# CHARGEMENT DINO

def load_dino(device, model_name='dino_vits8'):
    """
    Charge DINO vits8.

    STEGO original utilise DINO vits8 (patch 8)
    pour une meilleure résolution spatiale.
    """
    import torch
    print(f"[DINO] Chargement {model_name}...",
          end=' ', flush=True)
    model = torch.hub.load(
        'facebookresearch/dino:main',
        model_name, pretrained=True)
    model.eval().to(device)
    print(f"OK ({device})")
    return model


#  PRÉTRAITEMENT

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



# EXTRACTION FEATURES DINO

def extract_dino_features(model, tensor, device):
    """
    Extrait les features DINO patch par patch.

    Pour DINO vits8 sur image 224×224 :
    → (224/8)² = 784 patches × 384 dims
    """
    import torch

    with torch.no_grad():
        tensor = tensor.to(device)
        feats  = model.get_intermediate_layers(
            tensor, n=1)[0]
        # Shape : (1, N_patches+1, D)
        feats  = feats.squeeze(0)
        # Enlever token CLS
        pf     = feats[1:, :]  # (N_patches, D)
        h = w  = int(pf.shape[0] ** 0.5)
        d      = pf.shape[1]
        feat_map = pf.reshape(
            h, w, d).cpu().numpy()

    print(f"  Features DINO : {feat_map.shape}")
    return feat_map


# ============================================================
#  4. TÊTE DE SEGMENTATION STEGO
#
#  STEGO ajoute une tête linéaire légère au-dessus
#  de DINO :
#
#  feat_DINO (384 dims) → tête_STEGO → code (70 dims)
#
#  La tête est entraînée via une perte contrastive
#  basée sur les CORRESPONDANCES entre patches :
#
#  Perte STEGO =
#    - Pour chaque patch i, ses K voisins les plus
#      similaires dans DINO doivent aussi être
#      similaires dans l'espace STEGO (attirer)
#    - Les patches non-voisins doivent être
#      différents dans STEGO (repousser)
#
#  → STEGO apprend à "regrouper" les patches
#    qui se ressemblent sémantiquement
# ============================================================

class STEGOHead:
    """
    Tête de segmentation STEGO légère.

    Architecture :
    Linear(384 → 384) → ReLU → Linear(384 → 70)
    → L2 normalization

    Entraînée via perte contrastive sur vos images.
    """

    def __init__(self, in_dim=384,
                 out_dim=STEGO_DIM):
        import torch
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )
        self.out_dim = out_dim

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def forward(self, x):
        """
        x : tensor (N, D)
        Retourne : tensor (N, out_dim) normalisé L2
        """
        import torch
        import torch.nn.functional as F
        out = self.net(x)
        return F.normalize(out, dim=-1)

    def __call__(self, x):
        return self.forward(x)


def build_correspondence_matrix(feat_flat,
                                  k=STEGO_K):
    """
    Construit la matrice de correspondance STEGO.

    Pour chaque patch i, trouve ses K plus proches
    voisins dans l'espace de features DINO.

    Ces correspondances guident l'entraînement :
    les voisins dans DINO doivent rester voisins
    dans l'espace STEGO.

    Retourne :
      pos_pairs : liste de (i, j) paires positives
      neg_pairs : liste de (i, j) paires négatives
    """
    from sklearn.neighbors import NearestNeighbors

    N = feat_flat.shape[0]

    # K plus proches voisins
    nbrs = NearestNeighbors(
        n_neighbors=k+1,
        metric='cosine',
        n_jobs=-1)
    nbrs.fit(feat_flat)
    distances, indices = nbrs.kneighbors(feat_flat)

    # Paires positives (voisins proches)
    pos_pairs = []
    for i in range(N):
        for j in indices[i, 1:]:  # skip self
            pos_pairs.append((i, int(j)))

    # Paires négatives (éloignés)
    neg_pairs = []
    rng = np.random.default_rng(42)
    for i in range(N):
        # Tirer des négatifs au hasard loin de i
        neg_idx = rng.choice(
            N, size=STEGO_NEG_K * 3,
            replace=False)
        # Garder ceux qui ne sont pas voisins
        neighbor_set = set(indices[i, 1:].tolist())
        neg_idx = [n for n in neg_idx
                   if n not in neighbor_set
                   and n != i][:STEGO_NEG_K]
        for j in neg_idx:
            neg_pairs.append((i, j))

    return pos_pairs, neg_pairs


def stego_loss(codes, pos_pairs, neg_pairs,
               temperature=TEMPERATURE):
    """
    Perte contrastive STEGO.

    Pour chaque paire positive (i,j) :
    maximiser la similarité cosinus entre codes[i] et codes[j]

    Pour chaque paire négative (i,j) :
    minimiser la similarité cosinus

    = InfoNCE loss sur les correspondances DINO
    """
    import torch
    import torch.nn.functional as F

    if not pos_pairs or not neg_pairs:
        return torch.tensor(0.0,
                            requires_grad=True)

    # Paires positives
    pos_i = torch.tensor(
        [p[0] for p in pos_pairs[:100]],
        dtype=torch.long)
    pos_j = torch.tensor(
        [p[1] for p in pos_pairs[:100]],
        dtype=torch.long)

    sim_pos = (codes[pos_i] *
               codes[pos_j]).sum(dim=-1)
    sim_pos = sim_pos / temperature

    # Paires négatives
    neg_i = torch.tensor(
        [p[0] for p in neg_pairs[:100]],
        dtype=torch.long)
    neg_j = torch.tensor(
        [p[1] for p in neg_pairs[:100]],
        dtype=torch.long)

    sim_neg = (codes[neg_i] *
               codes[neg_j]).sum(dim=-1)
    sim_neg = sim_neg / temperature

    # Perte = -log(exp(sim_pos) /
    #               (exp(sim_pos) + exp(sim_neg)))
    loss = -F.logsigmoid(sim_pos).mean() + \
            F.logsigmoid(sim_neg).mean()

    return loss


def train_stego_head(feat_flat, device,
                     n_steps=STEGO_STEPS,
                     lr=STEGO_LR):
    """
    Entraîne la tête STEGO sur les features
    d'une seule image (micro-entraînement).

    C'est l'innovation clé de notre adaptation :
    au lieu d'entraîner sur un grand dataset,
    on entraîne quelques pas sur l'image courante
    → adaptation rapide aux structures anatomiques
      de cette IRM spécifique.

    Retourne les codes STEGO optimisés.
    """
    import torch
    import torch.optim as optim

    N, D   = feat_flat.shape
    head   = STEGOHead(in_dim=D,
                       out_dim=STEGO_DIM).to(device)
    optim_ = optim.Adam(
        head.parameters(), lr=lr)

    # Correspondances DINO
    print(f"  Construction correspondances "
          f"(K={STEGO_K})...",
          end=' ', flush=True)
    pos_pairs, neg_pairs = \
        build_correspondence_matrix(feat_flat)
    print(f"OK ({len(pos_pairs)} pos, "
          f"{len(neg_pairs)} neg)")

    feat_t = torch.from_numpy(
        feat_flat).float().to(device)

    print(f"  Entraînement STEGO "
          f"({n_steps} pas)...",
          end=' ', flush=True)

    for step in range(n_steps):
        optim_.zero_grad()
        codes = head(feat_t)
        loss  = stego_loss(
            codes, pos_pairs, neg_pairs)
        loss.backward()
        optim_.step()

        if step % 10 == 0:
            print(f"loss={loss.item():.4f}",
                  end=' ', flush=True)

    print("OK")

    # Codes finaux
    with torch.no_grad():
        final_codes = head(feat_t).cpu().numpy()

    return final_codes


# SEGMENTATION PAR CLUSTERING SUR CODES STEGO

def stego_segment(feat_map, img_orig,
                  device,
                  n_segments=N_REGIONS):
    """
    Pipeline STEGO complet :
    1. Aplatir features DINO
    2. Entraîner tête STEGO (micro)
    3. Obtenir les codes STEGO
    4. K-Means sur les codes
    5. Upscale vers taille image

    Les codes STEGO sont bien meilleurs que les
    features DINO brutes pour le clustering car
    ils ont été optimisés pour regrouper les
    patches sémantiquement similaires.
    """
    from scipy.ndimage import zoom

    h_p, w_p, d = feat_map.shape
    H, W         = img_orig.shape
    N            = h_p * w_p

    # Normalisation features DINO
    feat_flat = feat_map.reshape(N, d)
    feat_flat = normalize(feat_flat, norm='l2')

    # Entraînement tête STEGO
    stego_codes = train_stego_head(
        feat_flat, device)
    # stego_codes : (N, STEGO_DIM)

    print(f"  Codes STEGO : {stego_codes.shape}")

    # PCA pour visualisation
    n_comp = min(20, stego_codes.shape[0]-1,
                 stego_codes.shape[1])
    pca    = PCA(n_components=n_comp,
                 random_state=42)
    codes_pca = pca.fit_transform(stego_codes)
    var_exp   = pca.explained_variance_ratio_.sum()
    print(f"  PCA codes : {var_exp*100:.1f}% "
          f"variance expliquée")

    # Ajout features spatiales
    yy, xx = np.mgrid[0:h_p, 0:w_p]
    pos    = np.stack([
        yy.flatten() / h_p * 0.3,
        xx.flatten() / w_p * 0.3,
    ], axis=1)

    X_combined = np.hstack([codes_pca, pos])
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X_combined)

    # K-Means sur les codes STEGO
    print(f"  K-Means ({n_segments} clusters)...",
          end=' ', flush=True)
    km     = KMeans(n_clusters=n_segments,
                    random_state=42,
                    n_init=15, max_iter=500)
    labels = km.fit_predict(X_scaled)
    print("OK")

    seg_patch = labels.reshape(h_p, w_p)

    # Upscale vers taille image
    zh = H / h_p
    zw = W / w_p
    seg_full = zoom(seg_patch.astype(float),
                    (zh, zw), order=0).astype(int)
    seg_full = np.clip(seg_full, 0, n_segments-1)

    return seg_full, stego_codes



#  VISUALISATION DES CODES STEGO (PCA 2D)

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


# ATTRIBUTION ANATOMIQUE

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


# ANALYSE TEXTURE MANNIL

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


# VISUALISATION

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
        f"STEGO + Texture Mannil 2018 "
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

    # P2 : Espace STEGO (PCA RGB)
    ax = fig.add_subplot(gs1[1])
    ax.imshow(np.clip(codes_full, 0, 1))
    ax.set_title('Espace STEGO\n(PCA RGB)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # P3 : Segmentation STEGO brute
    ax = fig.add_subplot(gs1[2])
    ax.imshow(seg_stego, cmap='tab10')
    ax.set_title('K-Means\nsur codes STEGO',
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
    ax.set_title('Segmentation\nanatomique STEGO',
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




if __name__ == '__main__':

    import torch

    device = ('cuda'
              if torch.cuda.is_available()
              else 'cpu')
    print(f"[INFO] Device : {device}")

    dino_model = load_dino(device)

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

        # STEGO : entraînement + segmentation
        try:
            seg_stego, stego_codes = \
                stego_segment(
                    feat_map, img_orig, device)
        except Exception as e:
            print(f"  [ERREUR STEGO] {e}")
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
            f"{name}_stego_mannil.png")
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
            f"{name}_stego_codes.npy"),
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
            'all_patients_stego_mannil.csv')
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