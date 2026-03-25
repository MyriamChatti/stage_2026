"""
============================================================
  unsam_spine_segmentation.py

  Pipeline 100% non supervisé basé sur UNSAM
  pour segmenter les 10 régions spinales et paraspinales
  sur IRM T2 axiale lombaire + analyse texture Mannil.

  Référence :
    Wang et al., "Segment Anything Without Supervision",
    NeurIPS 2024.
    → Cité dans l'article de votre encadrante [14]
    "CutLER within a divide-and-conquer algorithm
    and the Segment Anything model (SAM) to develop
    UNSAM"

  Principe :
    1. CutLER (Cut and Learn) détecte automatiquement
       les objets/régions sans annotation
       → utilise NCut récursif sur features DINO
       pour proposer des masques candidats
    2. SAM (Segment Anything Model) raffine chaque
       masque candidat avec précision
    3. Sélection et fusion des meilleurs masques
    4. Attribution anatomique par position + intensité T2
    5. Analyse texture Mannil 2018 par région

  Note : Cette implémentation reproduit le principe
  de UNSAM sans nécessiter l'installation complète
  de CutLER (complexe). On utilise :
    - DINO v1 pour les propositions de masques
      (comme dans CutLER original)
    - SAM pour le raffinement
    - Algorithme divide-and-conquer pour les régions

  Installation :
      # SAM (Meta)
      pip install git+https://github.com/facebookresearch/segment-anything.git
      # Poids SAM : télécharger vit_b
      # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

      # Autres
      pip install torch torchvision
      pip install scikit-learn scikit-image scipy
      pip install numpy matplotlib pandas pillow

  Usage :
      python unsam_spine_segmentation.py
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
from scipy.sparse.linalg import eigsh
from skimage import exposure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label as sk_label, regionprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import warnings
warnings.filterwarnings('ignore')



INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/unsam_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

# Chemin poids SAM
# Télécharger : wget https://dl.fbaipublicfiles.com/
#               segment_anything/sam_vit_b_01ec64.pth
SAM_CHECKPOINT = os.path.expanduser(
    "~/sam_checkpoints/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"   # vit_b (léger) ou vit_h (précis)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(SAM_CHECKPOINT),
            exist_ok=True)



#paramètres
IMG_SIZE       = 224
PATCH_SIZE     = 8      # DINO v1 patch 8 (meilleure résolution)
N_REGIONS      = 10
N_PROPOSALS    = 20     # nb masques candidats CutLER
MIN_MASK_SIZE  = 0.005  # fraction min de l'image
MAX_MASK_SIZE  = 0.6    # fraction max de l'image
N_EIGENVECS    = 10     # vecteurs propres pour NCut





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



# TÉLÉCHARGEMENT AUTOMATIQUE DES POIDS SAM

def download_sam_weights(checkpoint=SAM_CHECKPOINT):
    """Télécharge automatiquement les poids SAM si absents."""
    if os.path.exists(checkpoint):
        print(f"[SAM] Poids trouvés : {checkpoint}")
        return True

    url = ("https://dl.fbaipublicfiles.com/"
           "segment_anything/sam_vit_b_01ec64.pth")

    print(f"[SAM] Téléchargement poids (~375MB)...")
    print(f"  URL : {url}")

    try:
        import urllib.request

        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = int(
                    count * block_size * 100
                    / total_size)
                print(f"\r  {pct}%",
                      end='', flush=True)

        urllib.request.urlretrieve(
            url, checkpoint,
            reporthook=progress)
        print(f"\n[SAM] Téléchargé !")
        return True

    except Exception as e:
        print(f"\n[SAM] Échec : {e}")
        print(f"  Téléchargez manuellement :")
        print(f"  {url}")
        print(f"  → {checkpoint}")
        return False


#  CHARGEMENT DES MODÈLES

def load_dino_model(device):
    """Charge DINO vits8 pour les propositions CutLER."""
    import torch
    print("[DINO] Chargement vits8...",
          end=' ', flush=True)
    model = torch.hub.load(
        'facebookresearch/dino:main',
        'dino_vits8', pretrained=True)
    model.eval().to(device)
    print("OK")
    return model


def load_sam_model(checkpoint=SAM_CHECKPOINT,
                   model_type=SAM_MODEL_TYPE):
    """Charge SAM pour le raffinement des masques."""
    try:
        from segment_anything import (
            sam_model_registry,
            SamPredictor,
            SamAutomaticMaskGenerator)

        print(f"[SAM] Chargement {model_type}...",
              end=' ', flush=True)
        sam   = sam_model_registry[model_type](
            checkpoint=checkpoint)
        device = 'cuda' \
            if __import__('torch').cuda.is_available() \
            else 'cpu'
        sam.to(device)

        predictor = SamPredictor(sam)
        generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side       = 16,
            pred_iou_thresh       = 0.7,
            stability_score_thresh= 0.8,
            min_mask_region_area  = 100,
        )
        print(f"OK ({device})")
        return predictor, generator

    except ImportError:
        print("\n[SAM] Non installé.")
        print("  pip install git+https://github.com/"
              "facebookresearch/"
              "segment-anything.git")
        return None, None

    except Exception as e:
        print(f"\n[SAM] Erreur : {e}")
        return None, None


# PRÉTRAITEMENT

def preprocess(path, size=IMG_SIZE):
    """Prétraite l'IRM T2 pour DINO + SAM."""
    import torch
    import torchvision.transforms as T

    img = Image.open(str(path)).convert('L')
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.min()) / \
          (arr.max() - arr.min() + 1e-8)

    sz   = (size // PATCH_SIZE) * PATCH_SIZE
    img_r = Image.fromarray(
        (arr * 255).astype(np.uint8)
    ).resize((sz, sz), Image.BILINEAR)
    arr_r = np.array(img_r, dtype=np.float32) / 255

    # CLAHE
    arr_c = exposure.equalize_adapthist(
        arr_r, clip_limit=0.02)

    # RGB pour DINO et SAM
    img_rgb = Image.merge('RGB', [
        Image.fromarray(
            (arr_c * 255).astype(np.uint8))
    ] * 3)

    # Tensor ImageNet
    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    tensor = t(img_rgb).unsqueeze(0)

    # RGB uint8 pour SAM
    rgb_uint8 = np.array(img_rgb, dtype=np.uint8)

    return arr_r, arr_c, rgb_uint8, tensor, sz



# CUTLER : PROPOSITIONS DE MASQUES PAR NCUT RÉCURSIF
#
#  CutLER (Cut and Learn) génère des masques candidats
#  sans supervision en appliquant NCut récursivement
#  sur les features DINO :
#
#  1. Extraire les features DINO de l'image
#  2. NCut divise l'image en 2 régions
#  3. Pour chaque région, recommencer (récursif)
#  4. Conserver toutes les régions intermédiaires
#     comme masques candidats
#  5. Filtrer par taille et qualité

def extract_dino_features(model, tensor, device,
                            patch_size=PATCH_SIZE):
    """Extrait les features DINO par patch."""
    import torch
    with torch.no_grad():
        tensor = tensor.to(device)
        feats  = model.get_intermediate_layers(
            tensor, n=1)[0]
        feats  = feats.squeeze(0)
        # Enlever token CLS
        patch_feats = feats[1:, :]
        h = w = int(patch_feats.shape[0]**0.5)
        feat_map = patch_feats.reshape(
            h, w, -1).cpu().numpy()
    return feat_map


def ncut_bipartition(feat_map, tau=0.15):
    """
    Bipartition NCut d'une carte de features.

    Divise les patches en 2 groupes selon le
    2ème vecteur propre du Laplacien normalisé.

    Retourne :
      mask_a, mask_b : masques booléens (h_p, w_p)
      score          : score de qualité de la coupure
    """
    from scipy.sparse import csr_matrix, diags

    h_p, w_p, d = feat_map.shape
    N = h_p * w_p
    F = feat_map.reshape(N, d)

    # Normalisation
    F_norm = normalize(F, norm='l2')

    # Matrice de similarité (cosinus)
    W = F_norm @ F_norm.T
    W = np.maximum(W, 0)
    np.fill_diagonal(W, 0)

    # Laplacien normalisé
    deg      = W.sum(axis=1)
    deg      = np.maximum(deg, 1e-8)
    D_inv_sq = diags(1.0 / np.sqrt(deg))
    L_sym    = D_inv_sq @ (diags(deg) -
               csr_matrix(W)) @ D_inv_sq

    # 2ème vecteur propre
    try:
        vals, vecs = eigsh(
            L_sym, k=2, which='SM', tol=1e-5)
        v2 = vecs[:, 1]
    except Exception:
        return None, None, 0

    # Bipartition par signe
    mask_a = (v2 >= 0).reshape(h_p, w_p)
    mask_b = ~mask_a

    # Score NCut (qualité de la coupure)
    labels = mask_a.flatten().astype(int)
    assoc_a = W[labels==0, :][:, labels==0].sum()
    assoc_b = W[labels==1, :][:, labels==1].sum()
    total   = W.sum() + 1e-8
    cut     = W[labels==0, :][:, labels==1].sum()
    score   = cut / (assoc_a + cut) + \
              cut / (assoc_b + cut)

    return mask_a, mask_b, float(score)


def cutler_proposals(feat_map, img_shape,
                     n_proposals=N_PROPOSALS,
                     max_depth=4):
    """
    Génère N masques candidats par NCut récursif.

    Algorithme divide-and-conquer :
    1. NCut sur l'image entière → 2 régions
    2. NCut sur chaque région → 2 sous-régions
    3. Continuer jusqu'à max_depth niveaux
    4. Conserver toutes les régions comme candidats

    Retourne :
      proposals : liste de masques (H, W) bool
    """
    from scipy.ndimage import zoom

    H, W  = img_shape
    h_p   = feat_map.shape[0]
    w_p   = feat_map.shape[1]
    zh, zw = H/h_p, W/w_p

    proposals = []
    # File de traitement : (feat_region, mask_global)
    queue     = [(feat_map,
                  np.ones((h_p, w_p), dtype=bool),
                  0)]

    while queue and len(proposals) < n_proposals:
        feat_reg, mask_reg, depth = queue.pop(0)

        if depth >= max_depth:
            continue
        if mask_reg.sum() < 4:
            continue

        # NCut bipartition
        mask_a, mask_b, score = ncut_bipartition(
            feat_reg)

        if mask_a is None:
            continue

        # Masques dans l'espace global
        global_a = mask_reg & mask_a
        global_b = mask_reg & mask_b

        for gm in [global_a, global_b]:
            if gm.sum() < 2:
                continue

            # Upscale vers taille image
            gm_full = zoom(
                gm.astype(float),
                (zh, zw), order=0
            ).astype(bool)

            size_frac = gm_full.sum() / (H*W)
            if (MIN_MASK_SIZE < size_frac
                    < MAX_MASK_SIZE):
                proposals.append(gm_full)

            # Features de la sous-région
            feat_sub = feat_reg.copy()
            feat_sub[~gm] = 0

            queue.append((feat_sub, gm, depth+1))

    print(f"  CutLER : {len(proposals)} "
          f"masques candidats générés")
    return proposals



# SAM RAFFINEMENT

def sam_refine_proposals(predictor, generator,
                          rgb_uint8, proposals,
                          img_orig):
    """
    Deux modes de SAM selon disponibilité :

    Mode 1 — Avec propositions CutLER :
      Pour chaque masque CutLER, extrait le point
      central et laisse SAM raffiner le contour.

    Mode 2 — SAM automatique (fallback) :
      SAM génère automatiquement tous les masques
      sans aucun prompt.
    """
    H, W = img_orig.shape

    if predictor is None and generator is None:
        print("  SAM non disponible → "
              "utilisation masques CutLER bruts")
        return proposals

    refined = []

    # Mode 2 : SAM automatique
    if generator is not None and len(proposals) == 0:
        print("  SAM automatique...",
              end=' ', flush=True)
        try:
            predictor.set_image(rgb_uint8)
            sam_masks = generator.generate(rgb_uint8)
            for m in sam_masks:
                mask = m['segmentation']
                size_f = mask.sum() / (H*W)
                if MIN_MASK_SIZE < size_f \
                        < MAX_MASK_SIZE:
                    refined.append(mask)
            print(f"OK ({len(refined)} masques)")
            return refined
        except Exception as e:
            print(f"ERREUR ({e})")
            return proposals

    # Mode 1 : raffinement des propositions CutLER
    if predictor is not None:
        print(f"  SAM raffine "
              f"{len(proposals)} masques...",
              end=' ', flush=True)
        try:
            predictor.set_image(rgb_uint8)
        except Exception as e:
            print(f"ERREUR set_image ({e})")
            return proposals

        for prop in proposals:
            ys, xs = np.where(prop)
            if len(ys) == 0:
                continue
            # Point central du masque
            cy = int(ys.mean())
            cx = int(xs.mean())
            # Vérifier que le point est dans le masque
            if not prop[cy, cx]:
                dists = (ys-cy)**2 + (xs-cx)**2
                n_idx = np.argmin(dists)
                cy, cx = int(ys[n_idx]), int(xs[n_idx])

            try:
                masks, scores, _ = \
                    predictor.predict(
                        point_coords=np.array(
                            [[cx, cy]]),
                        point_labels=np.array([1]),
                        multimask_output=True)
                best     = np.argmax(scores)
                best_m   = masks[best]
                size_f   = best_m.sum() / (H*W)
                if MIN_MASK_SIZE < size_f \
                        < MAX_MASK_SIZE:
                    refined.append(best_m)
            except Exception:
                refined.append(prop)

        print(f"OK ({len(refined)} masques raffinés)")

    return refined if refined else proposals


# SÉLECTION ET FUSION DES MASQUES --> CARTE DE SEGMENTS

def masks_to_segmap(masks, img_orig,
                    n_final=N_REGIONS):
    """
    Convertit une liste de masques en carte de
    segmentation (H, W) avec N régions.

    Stratégie :
    1. Trier les masques par score de qualité
       (uniformité d'intensité dans le masque)
    2. Appliquer les masques par ordre décroissant
       de qualité (les meilleurs en dernier
       = priorité haute)
    3. K-Means final pour regrouper en N_REGIONS
    """
    H, W = img_orig.shape

    if not masks:
        # Fallback : K-Means sur intensité + position
        print("  Fallback K-Means...")
        yy, xx = np.mgrid[0:H, 0:W]
        feat   = np.stack([
            img_orig,
            yy/H, xx/W], axis=-1
        ).reshape(-1, 3)
        km     = KMeans(n_clusters=n_final,
                        random_state=42,
                        n_init=10)
        labels = km.fit_predict(feat)
        return labels.reshape(H, W)

    # Score de qualité : homogénéité du masque
    def mask_score(m):
        if m.sum() == 0:
            return -1
        px  = img_orig[m]
        return -float(px.std())  # plus uniforme = meilleur

    masks_sorted = sorted(
        masks, key=mask_score, reverse=False)

    # Construction de la carte
    seg_map    = np.zeros((H, W), dtype=np.int32)
    next_label = 1

    for m in masks_sorted:
        if next_label > 50:   # max labels intermédiaires
            break
        region = m & (seg_map == 0)
        if region.sum() > 0:
            seg_map[region] = next_label
            next_label     += 1

    # Remplir les pixels non assignés
    unknown = seg_map == 0
    if unknown.any():
        _, idx_map = distance_transform_edt(
            unknown, return_indices=True)
        seg_map[unknown] = seg_map[
            idx_map[0][unknown],
            idx_map[1][unknown]]

    # Regrouper en N_REGIONS par K-Means
    # sur les features (intensité + position + label)
    yy, xx = np.mgrid[0:H, 0:W]
    feat   = np.stack([
        img_orig,
        yy.astype(float)/H,
        xx.astype(float)/W,
        seg_map.astype(float) / (next_label+1),
    ], axis=-1).reshape(-1, 4)

    scaler = StandardScaler()
    feat_s = scaler.fit_transform(feat)

    km = KMeans(n_clusters=n_final,
                random_state=42, n_init=15)
    final_labels = km.fit_predict(feat_s)
    final_map    = final_labels.reshape(H, W)

    return final_map


# ATTRIBUTION ANATOMIQUE

def assign_anatomy(seg_map, img_orig):
    """Attribution des labels anatomiques."""
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
                'dist_c': 0})
            continue
        ys, xs = np.where(m)
        mean_y = float(ys.mean())
        mean_x = float(xs.mean())
        dist_c = np.sqrt(
            ((mean_y-cy)/H)**2 +
            ((mean_x-cx)/W)**2)
        props.append({
            'k'     : k, 'n': n,
            'mean_i': float(img_orig[m].mean()),
            'cy'    : mean_y, 'cx': mean_x,
            'dist_c': dist_c})

    remaining = list(props)

    def pop_best(score_fn):
        best = min(remaining, key=score_fn)
        remaining.remove(best)
        return best['k']

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
        [p for p in remaining if p['cx'] < cx],
        key=lambda p: p['cy'])
    droite = sorted(
        [p for p in remaining if p['cx'] >= cx],
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
            rest = sorted(sl[1:],
                key=lambda p: abs(p['cx']-cx))
            res[rest[0]['k']] = im
            for p in rest[1:]:
                res[p['k']] = ie
        return res

    assignment = {
        fond_k: 0, disc_k: 1,
        sac_k : 2, emin_k: 3}
    assignment.update(assign_muscles(gauche, 'G'))
    assignment.update(assign_muscles(droite, 'D'))

    anat_map = np.zeros((H, W), dtype=np.int8)
    for k in range(n_segs):
        anat_map[seg_map == k] = \
            assignment.get(k, 0)

    for idx in range(1, N_REGIONS):
        m = anat_map == idx
        m = morphology.remove_small_objects(
            m, min_size=20)
        m = binary_closing(m, morphology.disk(2))
        anat_map[anat_map == idx] = 0
        anat_map[m] = idx

    unknown = anat_map == 0
    if unknown.any():
        _, idx_map = distance_transform_edt(
            unknown, return_indices=True)
        filled = anat_map[idx_map[0], idx_map[1]]
        anat_map[unknown] = filled[unknown]

    return anat_map


# ANALYSE TEXTURE MANNIL

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



def make_color_map(anat_map):
    H, W = anat_map.shape
    c    = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (_, color) in REGIONS.items():
        c[anat_map == idx] = color
    return c


def visualize(img_orig, proposals_vis,
              seg_intermediate, anat_map,
              all_feat, img_name,
              save_path=None):
    """Figure complète UNSAM."""

    fig = plt.figure(figsize=(30, 16),
                     facecolor='black')
    fig.suptitle(
        f"UNSAM (CutLER + SAM) + Texture "
        f"Mannil 2018 — {img_name}",
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

    # Panneau 1 : IRM originale
    ax = fig.add_subplot(gs1[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('IRM T2 originale',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Panneau 2 : Masques CutLER (propositions)
    ax = fig.add_subplot(gs1[1])
    ax.imshow(img01, cmap='gray')
    cmap_p = plt.get_cmap('tab20')
    for i, prop in enumerate(
            proposals_vis[:15]):
        color = cmap_p(i/15)[:3]
        overlay_p = np.zeros(
            (*img01.shape, 4))
        overlay_p[prop, :3] = color
        overlay_p[prop, 3]  = 0.4
        ax.imshow(overlay_p)
    ax.set_title(
        f'CutLER : {len(proposals_vis)}\n'
        f'masques candidats',
        color='white', fontsize=10,
        fontweight='bold')
    ax.axis('off')

    # Panneau 3 : Segmentation intermédiaire
    ax = fig.add_subplot(gs1[2])
    ax.imshow(seg_intermediate, cmap='tab10')
    ax.set_title('Segmentation\nintermédiaire',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Panneau 4 : Segmentation anatomique finale
    ax = fig.add_subplot(gs1[3])
    ax.imshow(color_seg)
    patches = [
        mpatches.Patch(
            color=np.array(REGIONS[i][1])/255,
            label=REGIONS[i][0])
        for i in range(N_REGIONS)]
    ax.legend(handles=patches,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.28),
              ncol=5, fontsize=5.5,
              facecolor='#222',
              labelcolor='white',
              framealpha=0.85)
    ax.set_title('UNSAM\n(segmentation finale)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Panneau 5 : Overlay + contours
    ax = fig.add_subplot(gs1[4])
    overlay = np.stack([img01]*3, axis=-1)
    c_f = color_seg.astype(np.float32)/255
    ov  = np.clip(0.45*overlay + 0.55*c_f, 0, 1)
    ax.imshow(ov)
    from skimage import segmentation as sg
    for idx in range(1, N_REGIONS):
        m     = anat_map == idx
        color = np.array(REGIONS[idx][1])/255
        if m.sum() > 0:
            bd = sg.find_boundaries(
                m, mode='outer')
            ax.contour(bd, colors=[color],
                       linewidths=0.8)
    ax.set_title('Overlay + contours',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Histogrammes + tableaux
    for i in range(N_REGIONS):
        name  = REGIONS[i][0]
        color = np.array(REGIONS[i][1])/255
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
                    colLabels = ['Feature', 'Val.'],
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

    device = 'cuda' \
        if torch.cuda.is_available() \
        else 'cpu'
    print(f"[INFO] Device : {device}")

    # Téléchargement poids SAM
    sam_ok = download_sam_weights()

    # Chargement modèles
    dino_model = load_dino_model(device)

    sam_predictor, sam_generator = None, None
    if sam_ok:
        sam_predictor, sam_generator = \
            load_sam_model()

    if sam_predictor is None:
        print("[INFO] SAM non disponible → "
              "UNSAM en mode CutLER seul")

    # Images
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
            img_orig, img_clahe, rgb_uint8, \
                tensor, sz = preprocess(img_path)
        except Exception as e:
            print(f"  [ERREUR] {e}")
            continue

        # Features DINO
        try:
            feat_map = extract_dino_features(
                dino_model, tensor, device)
            print(f"  Features DINO : "
                  f"{feat_map.shape}")
        except Exception as e:
            print(f"  [ERREUR DINO] {e}")
            continue

        # CutLER : propositions de masques
        try:
            proposals = cutler_proposals(
                feat_map, img_orig.shape)
        except Exception as e:
            print(f"  [ERREUR CutLER] {e}")
            proposals = []

        # SAM : raffinement
        try:
            refined_masks = sam_refine_proposals(
                sam_predictor, sam_generator,
                rgb_uint8, proposals, img_orig)
        except Exception as e:
            print(f"  [ERREUR SAM] {e}")
            refined_masks = proposals

        # Carte de segmentation
        try:
            seg_map = masks_to_segmap(
                refined_masks, img_orig)
        except Exception as e:
            print(f"  [ERREUR segmap] {e}")
            continue

        # Attribution anatomique
        try:
            anat_map = assign_anatomy(
                seg_map, img_orig)
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
                    f"{feat['hist_mean']:.3f}")
            else:
                print(f"    {rname:22s} → vide")

        # Visualisation
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_unsam_mannil.png")
        visualize(img_orig,
                  refined_masks, seg_map,
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
            'all_patients_unsam_mannil.csv')
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