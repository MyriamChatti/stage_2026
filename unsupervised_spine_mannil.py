"""
============================================================
  unsupervised_spine_mannil.py

  Pipeline 100% non supervisé pour IRM T2 axiale lombaire.
  Segmente les 10 régions d'intérêt :
    1. Disque intervertébral
    2. Sac thécal
    3. Éminence postérieure
    4. Psoas gauche
    5. Psoas droit
    6. Multifidus gauche
    7. Multifidus droit
    8. Érecteur des érecteurs gauche
    9. Érecteur des érecteurs droit
   10. Fond (background)

  Méthode :
    - Propriétés physiques IRM T2 pour guider
      la segmentation (pas d'annotation)
    - GMM (Gaussian Mixture Model) sur intensité
      + features spatiales + texture locale
    - Post-traitement anatomique par position
    - Analyse de texture Mannil 2018 par région

  Installation :
      pip install scikit-image scikit-learn numpy
                  matplotlib pandas scipy pillow

  Usage :
      python unsupervised_spine_mannil.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from scipy.ndimage import (gaussian_filter, median_filter,
                            binary_fill_holes,
                            binary_opening, binary_closing,
                            distance_transform_edt)
from scipy.stats import skew, kurtosis
from skimage import exposure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label as sk_label, regionprops
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/unsupervised_mannil_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                  '.tiff', '.tif'}
IMG_SIZE = (256, 256)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#  DÉFINITION DES RÉGIONS
#  Couleurs inspirées de la Figure 1 de l'article

REGIONS = {
    # Idx : (nom affiché, couleur RGB, description T2)
    0: ('Fond',                  [30,  30,  30],  'très sombre'),
    1: ('Disque intervert.',     [255, 200,   0],  'brillant T2'),
    2: ('Sac thécal',            [0,   180, 255],  'très brillant T2 (LCR)'),
    3: ('Éminence postérieure',  [180,  90,   0],  'sombre, os cortical'),
    4: ('Psoas gauche',          [220,  60, 180],  'gris moyen, grand'),
    5: ('Psoas droit',           [ 80, 200,  60],  'gris moyen, grand'),
    6: ('Multifidus gauche',     [ 50, 160,  80],  'gris moyen, postérieur'),
    7: ('Multifidus droit',      [160,  60, 200],  'gris moyen, postérieur'),
    8: ('Érecteur gauche',       [ 40,  80, 200],  'gris moyen, latéral'),
    9: ('Érecteur droit',        [210, 190,  40],  'gris moyen, latéral'),
}

N_REGIONS = len(REGIONS)


#  CHARGEMENT ET PRÉTRAITEMENT IRM T2

def load_and_preprocess(path, size=IMG_SIZE):
    """
    Charge + prétraite une IRM T2 axiale lombaire.

    Prétraitements :
    - Conversion niveaux de gris
    - Redimensionnement
    - Débruitage médian (conserve les bords)
    - CLAHE (amélioration contraste local)
    - Normalisation [0, 1]
    """
    img = Image.open(str(path)).convert('L')
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Normalisation [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Débruitage médian (préserve les contours anatomiques)
    arr_med = median_filter(arr, size=3)

    # CLAHE — révèle les structures musculaires
    arr_clahe = exposure.equalize_adapthist(
        arr_med, clip_limit=0.02)

    return arr, arr_clahe


#  EXTRACTION DE LA RÉGION D'INTÉRÊT GLOBALE
#     (éliminer le fond noir et la peau)

def extract_body_mask(image):
    """
    Extrait le masque du corps (élimine fond noir).

    En IRM T2, le fond est quasi noir (intensité < 0.05).
    On garde uniquement les pixels du corps.
    """
    # Seuil bas pour éliminer le fond
    body = image > 0.05

    # Fermeture morphologique pour combler les trous
    body = binary_closing(body,
                          morphology.disk(5))
    body = binary_fill_holes(body)

    # Garder uniquement le plus grand composant connexe
    labeled = sk_label(body)
    if labeled.max() == 0:
        return body
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    body    = labeled == largest.label

    return body


#  SEGMENTATION EN 10 RÉGIONS — APPROCHE NON SUPERVISÉE
#
#  Principe basé sur les propriétés IRM T2 :
#
#  Intensité T2 (brillance) :
#    - LCR / Sac thécal   → très brillant (blanc)
#    - Disque intervert.  → brillant (gris clair)
#    - Graisse            → brillant
#    - Muscles            → gris moyen
#    - Ligaments / os     → sombre
#    - Fond               → noir
#
#  Position spatiale :
#    - Sac thécal         → centre absolu
#    - Disque             → centre, juste devant sac
#    - Éminence post.     → centre, derrière sac
#    - Psoas              → latéral haut
#    - Multifidus         → postéro-central
#    - Érecteur           → postéro-latéral
# ============================================================

def build_feature_vector(image, clahe, body_mask):
    """
    Construit le vecteur de features par pixel pour GMM.

    Features (9 au total) :
    1. Intensité originale
    2. Intensité CLAHE
    3. Gradient magnitude (contours)
    4. Variance locale 5×5 (texture)
    5. Laplacien (netteté locale)
    6. Position y normalisée
    7. Position x normalisée
    8. Distance au centre normalisée
    9. Angle par rapport au centre
    """
    H, W = image.shape

    # Coordonnées
    yy, xx   = np.mgrid[0:H, 0:W]
    cy, cx   = H / 2, W / 2
    yy_n     = (yy - cy) / H
    xx_n     = (xx - cx) / W
    dist_c   = np.sqrt(yy_n**2 + xx_n**2)
    angle    = np.arctan2(yy_n, xx_n) / np.pi

    # Gradient
    grad = filters.sobel(clahe)

    # Variance locale
    from scipy.ndimage import uniform_filter
    mu   = uniform_filter(clahe, size=5)
    mu2  = uniform_filter(clahe**2, size=5)
    var  = np.sqrt(np.maximum(mu2 - mu**2, 0))

    # Laplacien
    lap  = np.abs(filters.laplace(clahe))

    feat = np.stack([
        image,          # 0 : intensité brute
        clahe,          # 1 : intensité CLAHE
        grad,           # 2 : gradient
        var,            # 3 : variance locale
        lap,            # 4 : laplacien
        yy_n,           # 5 : position verticale
        xx_n,           # 6 : position horizontale
        dist_c,         # 7 : distance au centre
        angle,          # 8 : angle
    ], axis=-1)

    return feat


def segment_unsupervised(image, clahe, body_mask):
    """
    Segmentation GMM non supervisée en 10 régions.

    Étape 1 : GMM sur les features (10 composantes)
    Étape 2 : Attribution anatomique par règles T2
    """
    H, W  = image.shape
    feat  = build_feature_vector(image, clahe, body_mask)
    X_all = feat.reshape(-1, feat.shape[-1])

    # Normalisation
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    #  GMM
    print("  GMM (10 composantes)...", end=' ', flush=True)
    gmm = GaussianMixture(
        n_components    = N_REGIONS,
        covariance_type = 'full',
        max_iter        = 200,
        n_init          = 5,
        random_state    = 42,
    )
    labels_flat = gmm.fit_predict(X_scaled)
    seg_map     = labels_flat.reshape(H, W)
    print("OK")

    # Attribution anatomique 
    seg_anatomical = assign_anatomy(seg_map, image, body_mask)

    return seg_anatomical


def assign_anatomy(seg_map, image, body_mask):
    """
    Attribue les labels anatomiques aux clusters GMM
    en utilisant les propriétés physiques IRM T2 et
    la position spatiale.

    Règles :
    ┌─────────────────────────────────────────────────────┐
    │ Région          │ Intensité T2  │ Position          │
    ├─────────────────────────────────────────────────────┤
    │ Fond            │ très basse    │ hors corps        │
    │ Sac thécal      │ très haute    │ centre            │
    │ Disque          │ haute         │ centre ant.       │
    │ Éminence post.  │ basse         │ centre post.      │
    │ Psoas G/D       │ moyenne       │ ant. latéral      │
    │ Multifidus G/D  │ moyenne-basse │ post. central     │
    │ Érecteur G/D    │ moyenne       │ post. latéral     │
    └─────────────────────────────────────────────────────┘
    """
    H, W = image.shape
    cy, cx = H / 2, W / 2
    n_clusters = seg_map.max() + 1

    # Calculer les propriétés de chaque cluster
    props = []
    for k in range(n_clusters):
        m = seg_map == k
        n = m.sum()
        if n == 0:
            props.append({
                'k': k, 'n': 0,
                'mean_i': 0, 'std_i': 0,
                'cy': cy, 'cx': cx,
                'dist_c': 0,
            })
            continue
        ys, xs = np.where(m)
        mean_y = float(ys.mean())
        mean_x = float(xs.mean())
        dist   = np.sqrt(
            ((mean_y - cy)/H)**2 +
            ((mean_x - cx)/W)**2)
        props.append({
            'k'      : k,
            'n'      : n,
            'mean_i' : float(image[m].mean()),
            'std_i'  : float(image[m].std()),
            'cy'     : mean_y,
            'cx'     : mean_x,
            'dist_c' : dist,
            'body_frac': float(
                (m & body_mask).sum() / (n + 1e-8)),
        })

    # Règle 0 : Fond 
    # Plus bas intensité ET hors corps
    fond_candidates = sorted(
        [p for p in props if p['body_frac'] < 0.3],
        key=lambda p: p['mean_i'])
    fond_k = fond_candidates[0]['k'] \
        if fond_candidates else \
        min(props, key=lambda p: p['mean_i'])['k']

    remaining = [p for p in props if p['k'] != fond_k]

    #  Règle 1 : Sac thécal 
    # Plus haute intensité + proche du centre
    sac_candidates = sorted(
        remaining,
        key=lambda p: -p['mean_i'] + p['dist_c'] * 2)
    sac_k = sac_candidates[0]['k']
    remaining = [p for p in remaining
                 if p['k'] != sac_k]

    #  Règle 2 : Disque intervertébral 
    # Haute intensité + centre + légèrement antérieur (y < cy)
    disc_candidates = sorted(
        remaining,
        key=lambda p: -p['mean_i'] +
                      abs(p['cx'] - cx)/W * 3 +
                      max(0, p['cy'] - cy)/H * 2)
    disc_k = disc_candidates[0]['k']
    remaining = [p for p in remaining
                 if p['k'] != disc_k]

    # Règle 3 : Éminence postérieure
    # Basse intensité + centre + postérieur (y > cy)
    emin_candidates = sorted(
        remaining,
        key=lambda p: p['mean_i'] +
                      abs(p['cx'] - cx)/W * 3 -
                      max(0, p['cy'] - cy)/H * 2)
    emin_k = emin_candidates[0]['k']
    remaining = [p for p in remaining
                 if p['k'] != emin_k]

    # Règles 4-9 : Muscles (6 régions)
    # Trier par position x pour gauche/droite
    # Trier par position y pour haut/bas
    muscles = sorted(remaining, key=lambda p: p['cx'])

    # Séparer gauche / droite selon cx vs cx de l'image
    gauche = [p for p in muscles if p['cx'] < cx]
    droite = [p for p in muscles if p['cx'] >= cx]

    # Dans chaque côté, trier par y (haut → bas)
    gauche_s = sorted(gauche, key=lambda p: p['cy'])
    droite_s = sorted(droite, key=lambda p: p['cy'])

    # Attribution :
    # Haut latéral   → psoas
    # Bas central    → multifidus
    # Bas latéral    → érecteur
    def assign_muscle_side(sorted_list, side='gauche'):
        """Attribue psoas/multifidus/érecteur selon position."""
        result = {}
        n = len(sorted_list)
        if n == 0:
            return result
        if n == 1:
            if side == 'gauche':
                result[sorted_list[0]['k']] = 4
            else:
                result[sorted_list[0]['k']] = 5
            return result
        if n == 2:
            # Haut = psoas, bas = érecteur
            if side == 'gauche':
                result[sorted_list[0]['k']] = 4
                result[sorted_list[1]['k']] = 8
            else:
                result[sorted_list[0]['k']] = 5
                result[sorted_list[1]['k']] = 9
            return result
        # n >= 3
        # Plus haut = psoas
        # Plus bas + plus central = multifidus
        # Plus bas + plus latéral = érecteur
        sorted_by_y = sorted_list
        psoas_p     = sorted_by_y[0]
        rest        = sorted(
            sorted_by_y[1:],
            key=lambda p: abs(p['cx'] - cx))
        # Plus proche du centre = multifidus
        multifidus_p = rest[0]
        erecteur_p   = rest[1] if len(rest) > 1 \
                       else rest[0]

        if side == 'gauche':
            result[psoas_p['k']]     = 4  # psoas G
            result[multifidus_p['k']]= 6  # multifidus G
            result[erecteur_p['k']]  = 8  # érecteur G
        else:
            result[psoas_p['k']]     = 5  # psoas D
            result[multifidus_p['k']]= 7  # multifidus D
            result[erecteur_p['k']]  = 9  # érecteur D
        # Clusters supplémentaires → érecteur
        for p in rest[2:]:
            result[p['k']] = 8 if side == 'gauche' else 9
        return result

    muscle_assignment = {}
    muscle_assignment.update(
        assign_muscle_side(gauche_s, 'gauche'))
    muscle_assignment.update(
        assign_muscle_side(droite_s, 'droite'))

    # Construction carte anatomique finale 
    assignment = {
        fond_k: 0,
        sac_k : 2,
        disc_k: 1,
        emin_k: 3,
    }
    assignment.update(muscle_assignment)

    # Carte finale
    anat_map = np.zeros((H, W), dtype=np.int8)
    for k in range(n_clusters):
        region_idx        = assignment.get(k, 0)
        anat_map[seg_map == k] = region_idx

    # Post-traitement : nettoyage morphologique
    anat_map = postprocess_map(anat_map)

    return anat_map


def postprocess_map(anat_map):
    """
    Nettoyage morphologique de la carte anatomique.

    - Fermeture pour combler les trous dans les régions
    - Suppression des petits artefacts
    - Remplissage des pixels non assignés
    """
    cleaned = anat_map.copy()

    for region_idx in range(1, N_REGIONS):
        mask = cleaned == region_idx

        # Supprimer petits artefacts
        mask = morphology.remove_small_objects(
            mask, min_size=30)

        # Fermeture morphologique légère
        mask = binary_closing(mask, morphology.disk(2))

        cleaned[anat_map == region_idx] = 0
        cleaned[mask]                   = region_idx

    return cleaned


# ANALYSE TEXTURE MANNIL PAR RÉGION

def extract_texture_mannil(image, mask):
    """
    Extrait les features de texture Mannil 2018.

    Features histogramme + GLCM comme dans l'article.
    """
    pixels = image[mask].astype(np.float32)
    if len(pixels) < 20:
        return None

    # Histogramme
    counts, _ = np.histogram(pixels, bins=256,
                              range=(0, 1))
    probs     = counts / (counts.sum() + 1e-8)
    probs_nz  = probs[probs > 0]

    features = {
        'n_pixels'       : int(len(pixels)),
        'hist_mean'      : float(np.mean(pixels)),
        'hist_variance'  : float(np.var(pixels)),
        'hist_std'       : float(np.std(pixels)),
        'hist_skewness'  : float(skew(pixels)),
        'hist_kurtosis'  : float(kurtosis(pixels)),
        'hist_entropy'   : float(
            -np.sum(probs_nz * np.log2(probs_nz))),
        'hist_p10'       : float(np.percentile(pixels, 10)),
        'hist_p25'       : float(np.percentile(pixels, 25)),
        'hist_p50'       : float(np.percentile(pixels, 50)),
        'hist_p75'       : float(np.percentile(pixels, 75)),
        'hist_p90'       : float(np.percentile(pixels, 90)),
        'hist_iqr'       : float(
            np.percentile(pixels, 75) -
            np.percentile(pixels, 25)),
    }

    # GLCM
    try:
        img_u8 = (image * 255).astype(np.uint8)
        glcm   = graycomatrix(
            img_u8,
            distances=[1, 2, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256, symmetric=True, normed=True)

        features['glcm_contrast']    = float(
            graycoprops(glcm, 'contrast').mean())
        features['glcm_energy']      = float(
            graycoprops(glcm, 'energy').mean())
        features['glcm_homogeneity'] = float(
            graycoprops(glcm, 'homogeneity').mean())
        features['glcm_correlation'] = float(
            graycoprops(glcm, 'correlation').mean())

        p_nz = glcm[:, :, 0, 0]
        p_nz = p_nz[p_nz > 0]
        features['glcm_entropy'] = float(
            -np.sum(p_nz * np.log2(p_nz + 1e-10)))

    except Exception:
        for k in ['glcm_contrast', 'glcm_energy',
                  'glcm_homogeneity', 'glcm_correlation',
                  'glcm_entropy']:
            features[k] = 0.0

    return features


#visusalisation

def make_color_map(anat_map):
    """Convertit la carte anatomique en image RGB."""
    H, W      = anat_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (name, color, _) in REGIONS.items():
        color_img[anat_map == idx] = color
    return color_img


def visualize(image, anat_map, all_feat,
              img_name, save_path=None):
    """
    Figure complète :
      - Ligne 1 : IRM | segmentation | overlay
      - Ligne 2 : histogramme par région (10 colonnes)
      - Ligne 3 : tableau features Mannil par région
    """
    fig = plt.figure(figsize=(26, 14),
                     facecolor='black')
    fig.suptitle(
        f"Segmentation non supervisée + Texture Mannil 2018"
        f" — {img_name}",
        color='white', fontsize=13,
        fontweight='bold')

    img01     = np.clip(image, 0, 1)
    color_seg = make_color_map(anat_map)

    # Layout
    gs1 = fig.add_gridspec(
        1, 3, left=0.01, right=0.99,
        top=0.91, bottom=0.62, wspace=0.05)
    gs2 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.58, bottom=0.32, wspace=0.15)
    gs3 = fig.add_gridspec(
        1, N_REGIONS, left=0.01, right=0.99,
        top=0.28, bottom=0.01, wspace=0.15)

    # --- Ligne 1 ---
    ax = fig.add_subplot(gs1[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('IRM T2 originale',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs1[1])
    ax.imshow(color_seg)
    patches = [
        mpatches.Patch(
            color=np.array(REGIONS[i][1])/255,
            label=REGIONS[i][0])
        for i in range(N_REGIONS)
    ]
    ax.legend(handles=patches,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.22),
              ncol=5, fontsize=6.5,
              facecolor='#222',
              labelcolor='white',
              framealpha=0.85)
    ax.set_title('Segmentation non supervisée (GMM)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs1[2])
    overlay = np.stack([img01]*3, axis=-1)
    c_f     = color_seg.astype(np.float32) / 255
    ov      = np.clip(0.45*overlay + 0.55*c_f, 0, 1)
    ax.imshow(ov)
    ax.set_title('Overlay IRM + segmentation',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # --- Lignes 2 & 3 : par région ---
    for i in range(N_REGIONS):
        name  = REGIONS[i][0]
        color = np.array(REGIONS[i][1]) / 255
        feat  = all_feat.get(i)

        # Histogramme
        ax_h = fig.add_subplot(gs2[i])
        ax_h.set_facecolor('#111')
        mask_r = anat_map == i

        if feat and feat['n_pixels'] > 20:
            px = image[mask_r]
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
                       fontsize=6.5, fontweight='bold')
        ax_h.tick_params(colors='white', labelsize=5)
        for sp in ax_h.spines.values():
            sp.set_edgecolor('#333')

        # Tableau features
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
                    for k, lbl in keys
                    if k in feat]
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

        # Chargement
        try:
            image, clahe = load_and_preprocess(img_path)
        except Exception as e:
            print(f"  [ERREUR] {e}")
            continue

        # Masque corps
        body_mask = extract_body_mask(image)

        # Segmentation non supervisée
        try:
            anat_map = segment_unsupervised(
                image, clahe, body_mask)
        except Exception as e:
            print(f"  [ERREUR segmentation] {e}")
            continue

        # Analyse texture Mannil par région
        all_feat = {}
        print("  Texture Mannil par région :")
        for idx in range(N_REGIONS):
            mask_r = anat_map == idx
            feat   = extract_texture_mannil(
                image, mask_r)
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
            f"{name}_unsup_mannil.png")
        visualize(image, anat_map, all_feat,
                  name, save_path=fig_path)

        # Sauvegardes
        np.save(os.path.join(OUTPUT_FOLDER,
                             f"{name}_anat_map.npy"),
                anat_map)

        color_img = make_color_map(anat_map)
        Image.fromarray(color_img).save(
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
            'all_patients_unsup_mannil.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[CSV] → {csv_path}")
        print(f"  {df.shape[0]} patients × "
              f"{df.shape[1]} features")

        print("\n[RÉSUMÉ] hist_mean par région :")
        for idx in range(N_REGIONS):
            rname = REGIONS[idx][0]
            col   = (f"{rname.replace(' ', '_')}"
                     f"_hist_mean")
            if col in df.columns:
                print(f"  {rname:25s} "
                      f"{df[col].mean():.4f} "
                      f"± {df[col].std():.4f}")

    print(f"\n[TERMINÉ] → {OUTPUT_FOLDER}")