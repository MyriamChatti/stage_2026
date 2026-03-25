"""
============================================================
  cutler_sam_spine_segmentation.py

  Pipeline 100% non supervisé basé sur CutLER + SAM
  pour segmenter les 10 régions spinales et paraspinales
  sur IRM T2 axiale lombaire + analyse texture Mannil.

  Référence :
    Wang et al., "Cut and Learn for Unsupervised
    Object Detection and Instance Segmentation",
    CVPR 2023.

  Différence clé avec UNSAM :
    - UNSAM  : CutLER génère des masques → SAM raffine
    - CutLER + SAM : CutLER détecte les OBJETS
      (boîtes englobantes) → SAM segmente chaque objet
      avec précision → fusion en carte anatomique

  Principe :
    1. MaskCut (cœur de CutLER) :
       NCut multi-rond sur features DINO
       → détecte plusieurs objets simultanément
       → produit des boîtes englobantes + masques grossiers
    2. SAM utilise ces boîtes comme prompts (box prompt)
       → segmentation précise de chaque objet
    3. Suppression des doublons (NMS)
    4. Attribution anatomique
    5. Analyse texture Mannil 2018

  Avantage par rapport à UNSAM :
    - Box prompts SAM = plus stable que point prompts
    - MaskCut détecte mieux les objets multiples
      (psoas gauche ET droit séparément)
    - NMS élimine les masques redondants

  Installation :
      pip install git+https://github.com/facebookresearch/segment-anything.git
      pip install torch torchvision
      pip install scikit-learn scikit-image scipy
      pip install numpy matplotlib pandas pillow

  Usage :
      python cutler_sam_spine_segmentation.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
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
from skimage.measure import (label as sk_label,
                              regionprops)
from sklearn.cluster import KMeans
from sklearn.preprocessing import (StandardScaler,
                                    normalize)
import warnings
warnings.filterwarnings('ignore')




INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/cutler_sam_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

SAM_CHECKPOINT = os.path.expanduser(
    "~/sam_checkpoints/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(SAM_CHECKPOINT),
            exist_ok=True)




#  PARAMÈTRES

IMG_SIZE        = 224
PATCH_SIZE      = 8        # DINO patch 8
N_REGIONS       = 10
N_MASKCUT_ROUNDS= 3        # nb de rounds MaskCut
                           # (= nb d'objets détectés)
NMS_THRESHOLD   = 0.5      # seuil IoU pour NMS
MIN_MASK_FRAC   = 0.003    # taille min masque
MAX_MASK_FRAC   = 0.65     # taille max masque
TAU             = 0.2      # seuil NCut bipartition

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


# TÉLÉCHARGEMENT POIDS SAM

def download_sam_weights(ckpt=SAM_CHECKPOINT):
    """Télécharge automatiquement les poids SAM."""
    if os.path.exists(ckpt):
        print(f"[SAM] Poids trouvés : {ckpt}")
        return True

    url = ("https://dl.fbaipublicfiles.com/"
           "segment_anything/"
           "sam_vit_b_01ec64.pth")
    print(f"[SAM] Téléchargement (~375MB)...")

    try:
        import urllib.request

        def progress(count, bs, total):
            if total > 0:
                pct = int(count*bs*100/total)
                print(f"\r  {pct}%",
                      end='', flush=True)

        urllib.request.urlretrieve(
            url, ckpt, reporthook=progress)
        print(f"\n[SAM] OK")
        return True
    except Exception as e:
        print(f"\n[SAM] Échec : {e}")
        print(f"  Téléchargez manuellement :")
        print(f"  {url} → {ckpt}")
        return False


#  CHARGEMENT MODÈLES

def load_dino(device):
    """Charge DINO vits8."""
    import torch
    print("[DINO] Chargement...",
          end=' ', flush=True)
    m = torch.hub.load(
        'facebookresearch/dino:main',
        'dino_vits8', pretrained=True)
    m.eval().to(device)
    print("OK")
    return m


def load_sam(ckpt=SAM_CHECKPOINT,
             model_type=SAM_MODEL_TYPE):
    """Charge SAM avec support box prompts."""
    try:
        from segment_anything import (
            sam_model_registry,
            SamPredictor)

        print(f"[SAM] Chargement {model_type}...",
              end=' ', flush=True)
        sam  = sam_model_registry[model_type](
            checkpoint=ckpt)
        dev  = 'cuda' \
            if __import__('torch') \
                   .cuda.is_available() \
            else 'cpu'
        sam.to(dev)
        pred = SamPredictor(sam)
        print(f"OK ({dev})")
        return pred

    except ImportError:
        print("\n[SAM] Non installé.")
        print("  pip install git+https://github.com/"
              "facebookresearch/"
              "segment-anything.git")
        return None
    except Exception as e:
        print(f"\n[SAM] Erreur : {e}")
        return None


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
    arr_r = np.array(img_r,
                     dtype=np.float32) / 255.0

    arr_c = exposure.equalize_adapthist(
        arr_r, clip_limit=0.02)

    img_clahe = Image.fromarray(
        (arr_c * 255).astype(np.uint8))
    img_rgb   = Image.merge(
        'RGB', [img_clahe] * 3)

    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    tensor    = t(img_rgb).unsqueeze(0)
    rgb_uint8 = np.array(img_rgb,
                         dtype=np.uint8)

    return arr_r, arr_c, rgb_uint8, tensor, sz


# ============================================================
#  4. MASKCUT — Cœur de CutLER
#
#  MaskCut détecte N objets en N rounds :
#
#  Round 1 :
#    - NCut sur toute l'image → objet principal
#    - Masquer cet objet
#  Round 2 :
#    - NCut sur l'image masquée → 2ème objet
#    - Masquer aussi
#  Round K :
#    - NCut sur image avec K-1 objets masqués
#
#  Résultat : K masques d'objets distincts
#  + leurs boîtes englobantes
# ============================================================

def extract_dino_features(model, tensor, device):
    """Extrait les features DINO patch."""
    import torch
    with torch.no_grad():
        tensor = tensor.to(device)
        feats  = model.get_intermediate_layers(
            tensor, n=1)[0].squeeze(0)
        pf     = feats[1:, :]   # sans CLS
        h = w  = int(pf.shape[0]**0.5)
        fm     = pf.reshape(h, w, -1).cpu().numpy()
    return fm


def compute_affinity(feat_map, mask=None):
    """
    Calcule la matrice d'affinité cosinus
    entre patches, avec masque optionnel.

    Si mask fourni, met à zéro les affinités
    des patches masqués → NCut ignore ces régions.
    """
    h_p, w_p, d = feat_map.shape
    N = h_p * w_p
    F = feat_map.reshape(N, d)
    F = normalize(F, norm='l2')

    W = F @ F.T
    W = np.maximum(W, 0)
    np.fill_diagonal(W, 0)

    # Appliquer le masque
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        W[~mask_flat, :] = 0
        W[:, ~mask_flat] = 0

    return W


def ncut_single(W):
    """
    NCut bipartition → retourne le masque
    de l'objet principal (plus grande région
    cohérente).
    """
    N   = W.shape[0]
    deg = np.array(W.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1e-8)

    D_inv_sq = diags(1.0 / np.sqrt(deg))
    L_sym    = D_inv_sq @ (
        diags(deg) - csr_matrix(W)
    ) @ D_inv_sq

    try:
        _, vecs = eigsh(
            L_sym, k=2, which='SM', tol=1e-4)
        v2 = vecs[:, 1]
    except Exception:
        return np.zeros(N, dtype=bool)

    # Bipartition par signe
    mask_pos = v2 >= 0
    mask_neg = ~mask_pos

    # Choisir la région la plus cohérente
    # = la plus petite (objet vs fond)
    if mask_pos.sum() < mask_neg.sum():
        return mask_pos
    return mask_neg


def maskcut(feat_map, img_shape,
            n_rounds=N_MASKCUT_ROUNDS,
            patch_size=PATCH_SIZE):
    """
    MaskCut : détection de N objets par NCut
    multi-round avec masquage progressif.

    Retourne :
      masks    : liste de masques (H, W) bool
      boxes    : liste de boîtes [x1,y1,x2,y2]
      scores   : liste de scores de qualité
    """
    from scipy.ndimage import zoom

    H, W         = img_shape
    h_p, w_p, _  = feat_map.shape
    zh, zw       = H/h_p, W/w_p

    masks_patches = []  # masques espace patches
    masks_full    = []  # masques espace image
    boxes         = []
    scores        = []

    # Masque des patches déjà utilisés
    used = np.zeros((h_p, w_p), dtype=bool)

    print(f"  MaskCut ({n_rounds} rounds) :",
          end=' ', flush=True)

    for r in range(n_rounds):
        # Masque des patches disponibles
        available = ~used

        if available.sum() < 4:
            break

        # Affinité avec masque
        W_aff = compute_affinity(
            feat_map,
            mask=available)

        # NCut bipartition
        obj_flat = ncut_single(W_aff)
        obj_map  = obj_flat.reshape(h_p, w_p)

        # Conserver uniquement les patches
        # disponibles
        obj_map  = obj_map & available

        if obj_map.sum() < 2:
            continue

        # Upscale vers image
        obj_full = zoom(
            obj_map.astype(float),
            (zh, zw), order=0
        ).astype(bool)

        # Filtrer par taille
        size_f = obj_full.sum() / (H*W)
        if not (MIN_MASK_FRAC < size_f
                < MAX_MASK_FRAC):
            used |= obj_map
            continue

        # Boîte englobante
        ys, xs = np.where(obj_full)
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())

        # Score = uniformité du masque
        # (sera utilisé plus tard)
        score = float(obj_map.sum()) / \
                float(available.sum() + 1e-8)

        masks_patches.append(obj_map)
        masks_full.append(obj_full)
        boxes.append([x1, y1, x2, y2])
        scores.append(score)

        # Masquer les patches détectés
        used |= obj_map

        print(f"R{r+1}({obj_full.sum()}px)",
              end=' ', flush=True)

    print(f"→ {len(masks_full)} masques")
    return masks_full, boxes, scores


# ============================================================
#  SAM RAFFINEMENT AVEC BOX PROMPTS
#
#  Différence clé vs UNSAM :
#  On utilise les BOÎTES ENGLOBANTES comme prompts
#  SAM → bien plus stable que les points centraux
#  → SAM segmente précisément tout ce qui est
#    dans la boîte
# ============================================================

def sam_refine_with_boxes(predictor, rgb_uint8,
                           masks_coarse, boxes,
                           scores, img_orig):
    """
    Raffine les masques CutLER avec SAM
    en utilisant les boîtes englobantes.

    Box prompt = plus robuste que point prompt :
    SAM sait exactement quelle région segmenter.
    """
    H, W = img_orig.shape

    if predictor is None:
        print("  SAM non disponible → "
              "masques CutLER bruts")
        return masks_coarse, scores

    print(f"  SAM box-prompts "
          f"({len(boxes)} boîtes)...",
          end=' ', flush=True)

    try:
        predictor.set_image(rgb_uint8)
    except Exception as e:
        print(f"ERREUR set_image ({e})")
        return masks_coarse, scores

    refined_masks  = []
    refined_scores = []

    for i, (box, score_c) in enumerate(
            zip(boxes, scores)):
        try:
            x1, y1, x2, y2 = box

            # Box prompt pour SAM
            box_arr = np.array(
                [x1, y1, x2, y2],
                dtype=np.float32)

            masks, scores_sam, _ = \
                predictor.predict(
                    box              = box_arr,
                    multimask_output = True)

            # Meilleur masque SAM
            best_idx  = np.argmax(scores_sam)
            best_mask = masks[best_idx]
            best_score= float(
                scores_sam[best_idx])

            # Filtrer par taille
            size_f = best_mask.sum() / (H*W)
            if MIN_MASK_FRAC < size_f \
                    < MAX_MASK_FRAC:
                refined_masks.append(best_mask)
                refined_scores.append(best_score)
            else:
                # Garder le masque CutLER
                refined_masks.append(
                    masks_coarse[i])
                refined_scores.append(score_c)

        except Exception:
            refined_masks.append(masks_coarse[i])
            refined_scores.append(score_c)

    print(f"OK ({len(refined_masks)} masques)")
    return refined_masks, refined_scores


# ============================================================
#  NON-MAXIMUM SUPPRESSION (NMS)
#
#  Élimine les masques redondants / qui se chevauchent
#  trop (IoU > seuil) en gardant le meilleur.
# ============================================================

def compute_iou(mask_a, mask_b):
    """Calcule l'IoU entre deux masques binaires."""
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter) / (float(union) + 1e-8)


def nms_masks(masks, scores,
              iou_thresh=NMS_THRESHOLD):
    """
    Non-Maximum Suppression sur les masques.

    Garde les masques avec le meilleur score
    et supprime ceux qui se chevauchent trop.
    """
    if not masks:
        return []

    # Trier par score décroissant
    order   = np.argsort(-np.array(scores))
    kept    = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        kept.append(masks[i])

        # Supprimer les masques similaires
        for j in order:
            if j == i or j in suppressed:
                continue
            iou = compute_iou(masks[i], masks[j])
            if iou > iou_thresh:
                suppressed.add(j)

    print(f"  NMS : {len(masks)} → "
          f"{len(kept)} masques conservés")
    return kept


# ============================================================
#  7. CONSTRUCTION CARTE DE SEGMENTATION
# ============================================================

def masks_to_segmap(masks, img_orig,
                    n_final=N_REGIONS):
    """
    Construit la carte de segmentation finale
    à partir des masques CutLER + SAM.

    Si pas assez de masques → K-Means fallback.
    """
    H, W = img_orig.shape

    if not masks:
        print("  Fallback K-Means...")
        yy, xx = np.mgrid[0:H, 0:W]
        feat   = np.stack([
            img_orig, yy/H, xx/W
        ], axis=-1).reshape(-1, 3)
        km = KMeans(n_clusters=n_final,
                    random_state=42, n_init=10)
        return km.fit_predict(feat).reshape(H, W)

    # Trier masques par taille décroissante
    # (petits objets = priorité haute)
    masks_sorted = sorted(
        masks,
        key=lambda m: m.sum(),
        reverse=True)

    seg_map    = np.zeros((H, W), dtype=np.int32)
    next_label = 1

    for m in masks_sorted:
        if next_label > 60:
            break
        # Assigner uniquement les pixels libres
        free = m & (seg_map == 0)
        if free.sum() > 20:
            seg_map[free] = next_label
            next_label   += 1

    # Remplir les pixels non assignés
    unknown = seg_map == 0
    if unknown.any():
        _, idx_map = distance_transform_edt(
            unknown, return_indices=True)
        seg_map[unknown] = seg_map[
            idx_map[0][unknown],
            idx_map[1][unknown]]

    # Regrouper en N_REGIONS
    yy, xx = np.mgrid[0:H, 0:W]
    feat   = np.stack([
        img_orig,
        yy.astype(float)/H,
        xx.astype(float)/W,
        seg_map.astype(float) / (next_label+1),
    ], axis=-1).reshape(-1, 4)

    scaler = StandardScaler()
    feat_s = scaler.fit_transform(feat)

    km     = KMeans(n_clusters=n_final,
                    random_state=42, n_init=15)
    labels = km.fit_predict(feat_s)
    return labels.reshape(H, W)



# ATTRIBUTION ANATOMIQUE

def assign_anatomy(seg_map, img_orig):
    """Attribution labels anatomiques IRM T2."""
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
        lambda p: -p['mean_i']*3 +
                  p['dist_c']*5)
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
        sac_k : 2, emin_k: 3}
    assignment.update(
        assign_muscles(gauche, 'G'))
    assignment.update(
        assign_muscles(droite, 'D'))

    anat = np.zeros((H, W), dtype=np.int8)
    for k in range(n_segs):
        anat[seg_map == k] = \
            assignment.get(k, 0)

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
            graycoprops(glcm,'contrast').mean())
        feat['glcm_energy']      = float(
            graycoprops(glcm,'energy').mean())
        feat['glcm_homogeneity'] = float(
            graycoprops(glcm,'homogeneity').mean())
        feat['glcm_correlation'] = float(
            graycoprops(glcm,'correlation').mean())
        p_nz = glcm[:,:,0,0]
        p_nz = p_nz[p_nz > 0]
        feat['glcm_entropy'] = float(
            -np.sum(p_nz*np.log2(p_nz+1e-10)))
    except Exception:
        for k in ['glcm_contrast','glcm_energy',
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


def visualize(img_orig, masks_cutler,
              boxes, masks_sam, anat_map,
              all_feat, img_name,
              save_path=None):
    """Figure complète CutLER + SAM."""

    fig = plt.figure(figsize=(30, 16),
                     facecolor='black')
    fig.suptitle(
        f"CutLER + SAM (box prompts) + "
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

    # P1 : IRM originale
    ax = fig.add_subplot(gs1[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('IRM T2 originale',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # P2 : Masques CutLER + boîtes
    ax = fig.add_subplot(gs1[1])
    ax.imshow(img01, cmap='gray')
    cmap_p = plt.get_cmap('Set1')
    for i, (m, box) in enumerate(
            zip(masks_cutler, boxes)):
        c = cmap_p(i/max(len(boxes),1))[:3]
        ov = np.zeros((*img01.shape, 4))
        ov[m, :3] = c
        ov[m,  3] = 0.45
        ax.imshow(ov)
        # Boîte englobante
        x1,y1,x2,y2 = box
        rect = patches.Rectangle(
            (x1,y1), x2-x1, y2-y1,
            linewidth=1.5,
            edgecolor=c, facecolor='none')
        ax.add_patch(rect)
    ax.set_title(
        f'MaskCut : {len(masks_cutler)}\n'
        f'objets + boîtes',
        color='white', fontsize=10,
        fontweight='bold')
    ax.axis('off')

    # P3 : Masques SAM raffinés
    ax = fig.add_subplot(gs1[2])
    ax.imshow(img01, cmap='gray')
    for i, m in enumerate(masks_sam[:12]):
        c = cmap_p(i/12)[:3]
        ov = np.zeros((*img01.shape, 4))
        ov[m, :3] = c
        ov[m,  3] = 0.45
        ax.imshow(ov)
    ax.set_title(
        f'SAM raffinement\n'
        f'(box prompts)',
        color='white', fontsize=10,
        fontweight='bold')
    ax.axis('off')

    # P4 : Segmentation anatomique finale
    ax = fig.add_subplot(gs1[3])
    ax.imshow(color_seg)
    pts = [
        mpatches.Patch(
            color=np.array(REGIONS[i][1])/255,
            label=REGIONS[i][0])
        for i in range(N_REGIONS)]
    ax.legend(handles=pts,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.28),
              ncol=5, fontsize=5.5,
              facecolor='#222',
              labelcolor='white',
              framealpha=0.85)
    ax.set_title(
        'Segmentation anatomique\nfinale',
        color='white', fontsize=10,
        fontweight='bold')
    ax.axis('off')

    # P5 : Overlay + contours SAM
    ax = fig.add_subplot(gs1[4])
    overlay = np.stack([img01]*3, axis=-1)
    c_f     = color_seg.astype(
        np.float32) / 255
    ov      = np.clip(
        0.45*overlay + 0.55*c_f, 0, 1)
    ax.imshow(ov)
    from skimage import segmentation as sg
    for idx in range(1, N_REGIONS):
        m = anat_map == idx
        c = np.array(REGIONS[idx][1])/255
        if m.sum() > 0:
            bd = sg.find_boundaries(
                m, mode='outer')
            ax.contour(bd, colors=[c],
                       linewidths=0.8)
    ax.set_title(
        'Overlay + contours\nanatomiques',
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
            ax_h.hist(px, bins=30,
                      color=color,
                      edgecolor='none',
                      alpha=0.9)
            ax_h.axvline(
                feat['hist_mean'],
                color='white',
                linestyle='--',
                linewidth=1.0)
        else:
            ax_h.text(
                0.5, 0.5, 'vide',
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
                for k, lbl in keys
                if k in feat]
            if rows:
                tbl = ax_t.table(
                    cellText  = rows,
                    colLabels = [
                        'Feature', 'Val.'],
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

    # Téléchargement + chargement
    sam_ok        = download_sam_weights()
    dino_model    = load_dino(device)
    sam_predictor = load_sam() if sam_ok else None

    if sam_predictor is None:
        print("[INFO] SAM non disponible → "
              "CutLER seul (sans raffinement)")

    image_paths = sorted([
        Path(INPUT_FOLDER) / f
        for f in os.listdir(INPUT_FOLDER)
        if Path(f).suffix.lower()
        in IMG_EXTENSIONS
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
            img_orig, img_clahe, \
                rgb_uint8, tensor, sz = \
                preprocess(img_path)
        except Exception as e:
            print(f"  [ERREUR] {e}")
            continue

        # Features DINO
        try:
            feat_map = extract_dino_features(
                dino_model, tensor, device)
            print(f"  DINO : {feat_map.shape}")
        except Exception as e:
            print(f"  [ERREUR DINO] {e}")
            continue

        # MaskCut
        try:
            masks_c, boxes, scores = maskcut(
                feat_map, img_orig.shape)
        except Exception as e:
            print(f"  [ERREUR MaskCut] {e}")
            masks_c, boxes, scores = [], [], []

        # SAM raffinement avec box prompts
        try:
            masks_s, scores_s = \
                sam_refine_with_boxes(
                    sam_predictor,
                    rgb_uint8,
                    masks_c, boxes,
                    scores, img_orig)
        except Exception as e:
            print(f"  [ERREUR SAM] {e}")
            masks_s, scores_s = \
                masks_c, scores

        # NMS
        try:
            masks_final = nms_masks(
                masks_s, scores_s)
        except Exception as e:
            print(f"  [ERREUR NMS] {e}")
            masks_final = masks_s

        # Carte de segmentation
        try:
            seg_map = masks_to_segmap(
                masks_final, img_orig)
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
                print(
                    f"    {rname:22s} → vide")

        # Visualisation
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_cutler_sam_mannil.png")
        visualize(img_orig,
                  masks_c, boxes,
                  masks_final,
                  anat_map, all_feat,
                  name, save_path=fig_path)

        # Sauvegardes
        np.save(os.path.join(
            OUTPUT_FOLDER,
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
            'all_patients_cutler_sam.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[CSV] → {csv_path}")
        print(f"  {df.shape[0]} patients × "
              f"{df.shape[1]} features")

        print("\n[RÉSUMÉ] hist_mean/région :")
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