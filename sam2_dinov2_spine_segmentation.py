"""
============================================================
  sam2_dinov2_spine_segmentation.py

  Pipeline 100% non supervisé basé sur SAM2 + DINOv2
  pour segmenter les 10 régions spinales et paraspinales
  sur IRM T2 axiale lombaire + analyse texture Mannil.

  Principe :
    1. DINOv2 (ViT-S/14) extrait des features sémantiques
       profondes par patch (768 dims) — bien meilleures
       que DINO v1 pour distinguer les tissus anatomiques
    2. K-Means sur les features DINOv2 → génère des
       points "seeds" au centre de chaque cluster
    3. SAM2 utilise ces seeds pour produire des masques
       précis avec des contours anatomiques nets
    4. Attribution anatomique par position + intensité T2
    5. Analyse texture Mannil 2018 par région

  Pourquoi SAM2 + DINOv2 ?
    - DINOv2 = features bien plus discriminantes que DINO v1
      (entraîné sur 142M images, patch 14×14)
    - SAM2 = contours anatomiques précis à partir de seeds
    - Combinaison = sémantique profonde + précision spatiale
    - Meilleure reconnaissance des structures anatomiques
      fines (multifidus, éminence postérieure...)

  Installation :
      # SAM2
      pip install git+https://github.com/facebookresearch/sam2.git
      # ou : pip install sam2

      # DINOv2 via torch.hub
      pip install torch torchvision

      # Autres
      pip install scikit-learn scikit-image
                  numpy matplotlib pandas scipy pillow

      # Télécharger les poids SAM2 :
      # https://dl.fbaipublicfiles.com/segment_anything_2/
      #         092824/sam2.1_hiera_small.pt
      # Mettre dans : ~/sam2_checkpoints/

  Usage :
      python sam2_dinov2_spine_segmentation.py
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
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/sam2_dinov2_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}

# Chemin vers les poids SAM2
# Télécharger depuis :
# https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
SAM2_CHECKPOINT = os.path.expanduser(
    "~/sam2_checkpoints/sam2.1_hiera_small.pt")
SAM2_CONFIG     = "sam2.1_hiera_s.yaml"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(SAM2_CHECKPOINT),
            exist_ok=True)





DINOV2_MODEL = 'dinov2_vits14'  # ViT-S/14 (768 dims)
                                  # Options : dinov2_vits14
                                  #           dinov2_vitb14
                                  #           dinov2_vitl14
PATCH_SIZE   = 14                 # patch DINOv2
IMG_SIZE     = 224                # taille entrée
N_CLUSTERS   = 10                 # nb de régions
N_PCA_COMP   = 64                 # réduction PCA



#  DÉFINITION DES 10 RÉGIONS ANATOMIQUES

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



# TÉLÉCHARGEMENT AUTOMATIQUE DES POIDS SAM2

def download_sam2_weights(checkpoint_path=SAM2_CHECKPOINT):
    """Télécharge automatiquement les poids SAM2 si absents."""
    if os.path.exists(checkpoint_path):
        print(f"[SAM2] Poids trouvés : {checkpoint_path}")
        return True

    url = ("https://dl.fbaipublicfiles.com/"
           "segment_anything_2/092824/"
           "sam2.1_hiera_small.pt")

    print(f"[SAM2] Téléchargement des poids...")
    print(f"  URL : {url}")
    print(f"  Destination : {checkpoint_path}")

    try:
        import urllib.request
        os.makedirs(os.path.dirname(checkpoint_path),
                    exist_ok=True)

        def progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size)
            print(f"\r  Progression : {pct}%",
                  end='', flush=True)

        urllib.request.urlretrieve(
            url, checkpoint_path, reporthook=progress)
        print(f"\n[SAM2] Poids téléchargés !")
        return True

    except Exception as e:
        print(f"\n[SAM2] Échec téléchargement : {e}")
        print(f"  Téléchargez manuellement depuis :")
        print(f"  {url}")
        print(f"  et placez le fichier dans :")
        print(f"  {checkpoint_path}")
        return False


# CHARGEMENT DES MODÈLES

def load_dinov2(model_name=DINOV2_MODEL):
    """Charge DINOv2 depuis torch.hub."""
    import torch

    print(f"[DINOv2] Chargement {model_name}...",
          end=' ', flush=True)
    model  = torch.hub.load(
        'facebookresearch/dinov2',
        model_name,
        pretrained=True)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() \
             else 'cpu'
    model  = model.to(device)
    print(f"OK ({device})")
    return model, device


def load_sam2(checkpoint=SAM2_CHECKPOINT,
              config=SAM2_CONFIG):
    """Charge SAM2."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import (
            SAM2ImagePredictor)

        print(f"[SAM2] Chargement...", end=' ', flush=True)
        sam2_model = build_sam2(config, checkpoint)
        predictor  = SAM2ImagePredictor(sam2_model)
        print("OK")
        return predictor

    except ImportError:
        print("\n[SAM2] Package non installé.")
        print("  Installez avec :")
        print("  pip install git+https://github.com/"
              "facebookresearch/sam2.git")
        return None

    except Exception as e:
        print(f"\n[SAM2] Erreur chargement : {e}")
        return None


#pré traitement images
def preprocess_image(path, size=IMG_SIZE):
    """
    Prétraite l'IRM T2 pour DINOv2 + SAM2.

    - Conversion niveaux de gris → RGB
    - CLAHE pour améliorer le contraste musculaire
    - Normalisation ImageNet pour DINOv2
    """
    import torch
    import torchvision.transforms as T

    # Chargement
    img = Image.open(str(path)).convert('L')
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.min()) / \
          (arr.max() - arr.min() + 1e-8)

    # CLAHE
    arr_clahe = exposure.equalize_adapthist(
        arr, clip_limit=0.02)

    # Taille multiple de patch_size
    sz      = (size // PATCH_SIZE) * PATCH_SIZE
    img_pil = Image.fromarray(
        (arr_clahe * 255).astype(np.uint8))
    img_pil = img_pil.resize((sz, sz), Image.BILINEAR)

    # RGB pour DINOv2
    img_rgb = Image.merge('RGB', [img_pil] * 3)

    # Tensor normalisé ImageNet pour DINOv2
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_rgb).unsqueeze(0)

    # Image originale normalisée
    img_orig = np.array(
        Image.open(str(path))
             .convert('L')
             .resize((sz, sz), Image.BILINEAR),
        dtype=np.float32) / 255.0

    # Image RGB uint8 pour SAM2
    img_rgb_uint8 = np.array(img_rgb, dtype=np.uint8)

    return tensor, img_orig, img_rgb_uint8, sz


# EXTRACTION FEATURES DINOv2

def extract_dinov2_features(model, tensor, device):
    """
    Extrait les features DINOv2 par patch.

    DINOv2 ViT-S/14 sur image 224×224 :
      - Patches : (224/14)² = 256 patches
      - Dimensions : 384 (ViT-S) ou 768 (ViT-B)

    Les features DINOv2 sont bien meilleures que DINO v1
    pour distinguer les tissus anatomiques car :
      - Entraîné sur 142M images (LVD-142M)
      - Patch 14×14 = contexte plus large
      - Supervision distillée + auto-supervisée
    """
    import torch

    with torch.no_grad():
        tensor = tensor.to(device)

        # Extraction features intermédiaires
        # get_intermediate_layers retourne les features
        # de la dernière couche
        out = model.get_intermediate_layers(
            tensor, n=1, reshape=True)[0]
        # Shape : (1, D, H_p, W_p)

        # Réorganiser en (H_p, W_p, D)
        feat_map = out.squeeze(0).permute(
            1, 2, 0).cpu().numpy()

    h_p, w_p, d = feat_map.shape
    print(f"  Features DINOv2 : {feat_map.shape} "
          f"({h_p}×{w_p} patches × {d} dims)")

    return feat_map


# CLUSTERING DES FEATURES DINOv2

def cluster_dinov2_features(feat_map, img_orig,
                              n_clusters=N_CLUSTERS,
                              n_pca=N_PCA_COMP):
    """
    Clustering K-Means sur les features DINOv2.

    Étapes :
    1. PCA (384/768 → 64 dims)
    2. Ajout features spatiales pondérées
    3. K-Means
    4. Upscaling vers taille originale
    5. Calcul des points seeds (centroïdes des clusters)
       → utilisés comme prompts pour SAM2

    Retourne :
      seg_coarse : carte segmentation grossière (H, W)
      seeds      : liste de (x, y, cluster_idx)
    """
    from scipy.ndimage import zoom

    h_p, w_p, d = feat_map.shape
    H, W         = img_orig.shape
    X            = feat_map.reshape(-1, d)

    # PCA
    n_comp = min(n_pca, X.shape[0]-1, X.shape[1])
    print(f"  PCA ({d}→{n_comp} dims)...",
          end=' ', flush=True)
    pca    = PCA(n_components=n_comp,
                 random_state=42)
    X_pca  = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_.sum()
    print(f"OK ({var*100:.1f}% variance)")

    # Features spatiales
    yy, xx = np.mgrid[0:h_p, 0:w_p]
    yy_n   = (yy.flatten() / h_p).reshape(-1, 1)
    xx_n   = (xx.flatten() / w_p).reshape(-1, 1)

    X_combined = np.hstack([
        X_pca,
        yy_n * 0.4 * n_comp,
        xx_n * 0.4 * n_comp,
    ])

    scaler     = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    # K-Means
    print(f"  K-Means ({n_clusters} clusters)...",
          end=' ', flush=True)
    kmeans  = KMeans(n_clusters=n_clusters,
                     random_state=42,
                     n_init=15, max_iter=500)
    labels  = kmeans.fit_predict(X_combined)
    seg_p   = labels.reshape(h_p, w_p)
    print("OK")

    # Upscaling vers taille originale
    zh = H / h_p
    zw = W / w_p
    seg_coarse = zoom(seg_p.astype(float),
                      (zh, zw), order=0).astype(int)
    seg_coarse = np.clip(seg_coarse, 0, n_clusters-1)

    # Calcul des seeds (centroïdes dans image originale)
    seeds = []
    for k in range(n_clusters):
        mask_k = seg_coarse == k
        if mask_k.sum() < 10:
            continue
        ys, xs = np.where(mask_k)
        # Point central du cluster
        cy = int(ys.mean())
        cx = int(xs.mean())
        # Vérifier que le point est bien dans le cluster
        # (le centroïde peut être hors du masque)
        if not mask_k[cy, cx]:
            # Trouver le point le plus proche du centroïde
            dists = (ys - cy)**2 + (xs - cx)**2
            nearest = np.argmin(dists)
            cy, cx = int(ys[nearest]), int(xs[nearest])
        seeds.append((cx, cy, k))

    return seg_coarse, seeds, kmeans


# RAFFINEMENT SAM2

def refine_with_sam2(predictor, img_rgb_uint8,
                     seeds, seg_coarse, img_orig):
    """
    Utilise SAM2 pour raffiner les masques à partir
    des seeds DINOv2.

    Pour chaque cluster :
    1. Donne le point seed comme prompt à SAM2
    2. SAM2 génère un masque précis avec contours nets
    3. Le masque SAM2 remplace le masque K-Means grossier

    SAM2 produit des contours anatomiques réels
    là où K-Means produisait des frontières pixelisées.
    """
    import torch

    H, W       = img_orig.shape
    n_clusters = seg_coarse.max() + 1

    # Initialiser SAM2 avec l'image
    predictor.set_image(img_rgb_uint8)

    refined_masks = {}

    print(f"  SAM2 raffinement ({len(seeds)} seeds)...")

    for cx, cy, k in seeds:
        try:
            # Point seed comme prompt
            point_coords  = np.array([[cx, cy]])
            point_labels  = np.array([1])  # 1 = foreground

            # Prédiction SAM2
            masks, scores, logits = predictor.predict(
                point_coords  = point_coords,
                point_labels  = point_labels,
                multimask_output = True,
            )

            # Choisir le masque avec le meilleur score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]

            # Intersectionner avec le cluster DINOv2
            # pour éviter de déborder sur d'autres régions
            cluster_mask = seg_coarse == k
            refined      = best_mask & cluster_mask

            # Si l'intersection est trop petite,
            # garder le masque SAM2 complet
            if refined.sum() < 0.3 * best_mask.sum():
                refined = best_mask

            refined_masks[k] = refined
            print(f"    Cluster {k:2d} → "
                  f"{refined.sum():5d} px "
                  f"(score={scores[best_idx]:.3f})")

        except Exception as e:
            print(f"    Cluster {k:2d} → ERREUR ({e})")
            # Fallback : garder le masque K-Means
            refined_masks[k] = seg_coarse == k

    # Reconstruire la carte de segmentation
    seg_refined = np.zeros((H, W), dtype=np.int8)

    # Trier par taille décroissante pour gérer les
    # chevauchements (plus grand = priorité basse)
    sorted_clusters = sorted(
        refined_masks.items(),
        key=lambda x: x[1].sum(),
        reverse=True)

    for k, mask in sorted_clusters:
        seg_refined[mask] = k

    return seg_refined


# ATTRIBUTION ANATOMIQUE

def assign_anatomy(seg_map, img_orig):
    """
    Attribue les labels anatomiques aux clusters.

    Règles IRM T2 :
      Fond            → très basse intensité + hors corps
      Sac thécal      → très brillant + centre absolu
      Disque          → brillant + centre antérieur
      Éminence post.  → sombre + centre postérieur
      Psoas           → latéral haut, intensité moyenne
      Multifidus      → postéro-central
      Érecteur        → postéro-latéral
    """
    H, W = img_orig.shape
    cy, cx = H/2, W/2
    n_clusters = int(seg_map.max()) + 1

    # Propriétés par cluster
    props = []
    for k in range(n_clusters):
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
            'k'      : k,
            'n'      : n,
            'mean_i' : float(img_orig[m].mean()),
            'cy'     : mean_y,
            'cx'     : mean_x,
            'dist_c' : dist_c,
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

    # Disque : brillant + centre antérieur (y < cy)
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

    # Muscles : gauche / droite
    gauche = sorted(
        [p for p in remaining if p['cx'] < cx],
        key=lambda p: p['cy'])
    droite = sorted(
        [p for p in remaining if p['cx'] >= cx],
        key=lambda p: p['cy'])

    def assign_muscles(side_list, side):
        res = {}
        n   = len(side_list)
        ip  = 4 if side == 'G' else 5
        im  = 6 if side == 'G' else 7
        ie  = 8 if side == 'G' else 9
        if n == 0:
            return res
        if n == 1:
            res[side_list[0]['k']] = ip
        elif n == 2:
            res[side_list[0]['k']] = ip
            res[side_list[1]['k']] = ie
        else:
            res[side_list[0]['k']] = ip
            rest = sorted(side_list[1:],
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

    # Carte anatomique finale
    anat_map = np.zeros((H, W), dtype=np.int8)
    for k in range(n_clusters):
        anat_map[seg_map == k] = assignment.get(k, 0)

    # Nettoyage
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
        p_nz = glcm[:,:,0,0]
        p_nz = p_nz[p_nz > 0]
        feat['glcm_entropy'] = float(
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


def visualize(img_orig, seg_coarse, anat_map,
              seeds, all_feat, img_name,
              save_path=None):
    """Figure complète 5 colonnes + histogrammes."""

    fig = plt.figure(figsize=(30, 16),
                     facecolor='black')
    fig.suptitle(
        f"SAM2 + DINOv2 — Segmentation non supervisée "
        f"+ Texture Mannil 2018 — {img_name}",
        color='white', fontsize=13, fontweight='bold')

    img01     = np.clip(img_orig, 0, 1)
    color_seg = make_color_map(anat_map)

    # Layout
    gs1 = fig.add_gridspec(
        1, 5, left=0.01, right=0.99,
        top=0.92, bottom=0.60, wspace=0.05)
    gs2 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.56, bottom=0.30, wspace=0.12)
    gs3 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.26, bottom=0.01, wspace=0.12)

    # IRM originale
    ax = fig.add_subplot(gs1[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('IRM T2 originale', color='white',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    # Segmentation grossière DINOv2
    ax = fig.add_subplot(gs1[1])
    ax.imshow(seg_coarse, cmap='tab10')
    # Points seeds
    for cx, cy, k in seeds:
        ax.plot(cx, cy, 'w+', markersize=8,
                markeredgewidth=1.5)
    ax.set_title('DINOv2 K-Means\n(seeds pour SAM2)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Segmentation raffinée SAM2
    ax = fig.add_subplot(gs1[2])
    ax.imshow(color_seg)
    patches = [
        mpatches.Patch(
            color=np.array(REGIONS[i][1])/255,
            label=REGIONS[i][0])
        for i in range(N_REGIONS)]
    ax.legend(handles=patches,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.26),
              ncol=5, fontsize=6,
              facecolor='#222',
              labelcolor='white',
              framealpha=0.85)
    ax.set_title('SAM2 + DINOv2\n(segmentation raffinée)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Overlay
    ax = fig.add_subplot(gs1[3])
    overlay = np.stack([img01]*3, axis=-1)
    c_f     = color_seg.astype(np.float32)/255
    ov      = np.clip(0.45*overlay + 0.55*c_f, 0, 1)
    ax.imshow(ov)
    ax.set_title('Overlay\nIRM + segmentation',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Contours sur IRM
    ax = fig.add_subplot(gs1[4])
    ax.imshow(img01, cmap='gray')
    for idx in range(1, N_REGIONS):
        m     = anat_map == idx
        color = np.array(REGIONS[idx][1]) / 255
        # Contour
        from skimage import segmentation as seg_sk
        boundary = seg_sk.find_boundaries(m, mode='outer')
        ax.contour(boundary, colors=[color],
                   linewidths=1.0)
    ax.set_title('Contours anatomiques\n(SAM2)',
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
            rows = [[lbl, f"{feat.get(k,0):.3f}"]
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

    # Téléchargement poids SAM2 si nécessaire
    sam2_ok = download_sam2_weights()

    # Chargement modèles
    dinov2_model, device = load_dinov2()

    sam2_predictor = None
    if sam2_ok:
        sam2_predictor = load_sam2()

    if sam2_predictor is None:
        print("\n[INFO] SAM2 non disponible → "
              "utilisation DINOv2 seul (sans raffinement)")

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
            tensor, img_orig, img_rgb_uint8, sz = \
                preprocess_image(img_path)
        except Exception as e:
            print(f"  [ERREUR] {e}")
            continue

        # Features DINOv2
        try:
            feat_map = extract_dinov2_features(
                dinov2_model, tensor, device)
        except Exception as e:
            print(f"  [ERREUR DINOv2] {e}")
            continue

        # Clustering DINOv2
        try:
            seg_coarse, seeds, _ = \
                cluster_dinov2_features(
                    feat_map, img_orig)
        except Exception as e:
            print(f"  [ERREUR clustering] {e}")
            continue

        # Raffinement SAM2 (si disponible)
        if sam2_predictor is not None:
            try:
                seg_refined = refine_with_sam2(
                    sam2_predictor,
                    img_rgb_uint8,
                    seeds, seg_coarse, img_orig)
            except Exception as e:
                print(f"  [ERREUR SAM2] {e} "
                      f"→ fallback DINOv2")
                seg_refined = seg_coarse
        else:
            seg_refined = seg_coarse

        # Attribution anatomique
        try:
            anat_map = assign_anatomy(
                seg_refined, img_orig)
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
                print(f"    {rname:22s} "
                      f"{feat['n_pixels']:5d}px "
                      f"mean={feat['hist_mean']:.3f}")
            else:
                print(f"    {rname:22s} → vide")

        # Visualisation
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_sam2_dinov2_mannil.png")
        visualize(img_orig, seg_coarse, anat_map,
                  seeds, all_feat, name,
                  save_path=fig_path)

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
            'all_patients_sam2_dinov2.csv')
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