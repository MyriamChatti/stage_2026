"""
============================================================
  spine_segmentation_mannil.py

  Pipeline complet :
  1. Segmentation automatique des 6 régions paraspinales
     (psoas G/D, érecteur G/D, multifidus G/D)
     sur IRM axiale lombaire T2 — sans annotation
  2. Analyse de texture Mannil 2018 sur chaque région

  Méthode de segmentation :
    - Prétraitement IRM (CLAHE, débruitage)
    - K-Means spatial + intensité (6 clusters)
    - Post-traitement anatomique (gauche/droite,
      haut/bas) pour affecter les labels corrects
    - Visualisation colorée comme la Figure 1 de
      l'article

  Installation :
      pip install scikit-image scikit-learn numpy
                  matplotlib pandas scipy pillow

  Usage :
      python spine_segmentation_mannil.py
============================================================
"""
 ###### code semi-supervisée ou non supervisée avec contraintes anatomiques a priori
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, label as nd_label
from skimage import exposure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
#  CHEMINS
# ============================================================

INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/segmentation_mannil_results"
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
#  DÉFINITION DES 6 RÉGIONS ANATOMIQUES

REGIONS = {
    'psoas_gauche'    : {'color': [220,  60, 180], 'side': 'right',  'position': 'top'},
    'psoas_droit'     : {'color': [ 80, 180,  60], 'side': 'left',   'position': 'top'},
    'erecteur_gauche' : {'color': [ 40,  80, 200], 'side': 'right',  'position': 'bottom'},
    'multifidus_gauche': {'color': [ 50, 160,  80], 'side': 'center', 'position': 'bottom'},
    'multifidus_droit': {'color': [160,  60, 200], 'side': 'center', 'position': 'bottom'},
    'erecteur_droit'  : {'color': [210, 190,  40], 'side': 'left',   'position': 'bottom'},
}

REGION_NAMES = list(REGIONS.keys())
REGION_COLORS = np.array([v['color'] for v in REGIONS.values()],
                          dtype=np.uint8)


# CHARGEMENT

def load_and_preprocess(path, target_size=(256, 256)):
    """Charge l'image IRM et applique les prétraitements.

    - Conversion niveaux de gris
    - Redimensionnement
    - CLAHE (amélioration du contraste local)
    - Débruitage gaussien léger
    """
    img = Image.open(str(path)).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Normalisation [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # CLAHE — améliore la visibilité des structures musculaires
    arr_clahe = exposure.equalize_adapthist(arr, clip_limit=0.03)

    # Débruitage léger
    arr_denoised = gaussian_filter(arr_clahe, sigma=0.8)

    return arr, arr_denoised


#  SEGMENTATION AUTOMATIQUE EN 6 RÉGIONS

def build_feature_map(image, denoised):
    """Construit la carte de features pour K-Means.

    Features par pixel :
    - Intensité originale
    - Intensité débruitée
    - Gradient (texture locale)
    - Position normalisée (x, y) — contrainte spatiale
    - Texture locale (variance dans voisinage 5x5)
    """
    H, W = image.shape

    # Coordonnées spatiales normalisées
    yy, xx = np.mgrid[0:H, 0:W]
    yy_n = yy.astype(np.float32) / H
    xx_n = xx.astype(np.float32) / W

    # Gradient (magnitude des bords)
    grad = filters.sobel(denoised)

    # Variance locale (texture)
    from scipy.ndimage import uniform_filter
    mean_local = uniform_filter(denoised, size=5)
    mean2_local = uniform_filter(denoised**2, size=5)
    var_local = np.sqrt(np.maximum(mean2_local - mean_local**2, 0))

    # Empilement des features (H, W, 6)
    feat = np.stack([
        image,          # intensité brute
        denoised,       # intensité filtrée
        grad,           # gradient
        var_local,      # variance locale
        yy_n * 1.5,     # position verticale (pondérée)
        xx_n * 1.5,     # position horizontale (pondérée)
    ], axis=-1)

    return feat


def segment_6_regions(image, denoised, n_clusters=6,
                       random_state=42):
    """Segmente l'IRM en 6 régions par K-Means spatiotemporel.

    Retourne :
        seg_map  : ndarray (H, W) int — label 0..5 par pixel
        centers  : ndarray (6, n_features) — centres des clusters
    """
    H, W = image.shape

    # Construction de la carte de features
    feat = build_feature_map(image, denoised)
    X    = feat.reshape(-1, feat.shape[-1])

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    print(f"  K-Means ({n_clusters} clusters)...",
          end=' ', flush=True)
    kmeans = KMeans(
        n_clusters  = n_clusters,
        random_state= random_state,
        n_init      = 20,
        max_iter    = 500,
    )
    labels = kmeans.fit_predict(X_scaled)
    seg_map = labels.reshape(H, W)
    print("OK")

    return seg_map, kmeans.cluster_centers_


def assign_anatomical_labels(seg_map, image):
    """Affecte les labels anatomiques aux clusters K-Means.

    Stratégie basée sur la position spatiale et l'intensité :
    - Les psoas sont en HAUT de l'image (y < H/2), grands,
      intensité moyenne-haute
    - Les érecteurs sont en BAS à gauche et droite,
      intensité moyenne
    - Les multifidus sont en BAS au centre,
      petits, intensité moyenne-basse

    Retourne :
        anatomical_map : ndarray (H, W) int (0..5)
            correspondant à REGION_NAMES
    """
    H, W = seg_map.shape
    n_clusters = seg_map.max() + 1

    # Calculer les propriétés de chaque cluster
    props = []
    for k in range(n_clusters):
        mask = seg_map == k
        n_pixels = mask.sum()
        if n_pixels == 0:
            props.append({
                'label': k, 'size': 0,
                'cx': W/2, 'cy': H/2,
                'mean_intensity': 0.5,
                'mean_y': H/2, 'mean_x': W/2
            })
            continue

        ys, xs = np.where(mask)
        props.append({
            'label'         : k,
            'size'          : n_pixels,
            'cx'            : float(xs.mean()),
            'cy'            : float(ys.mean()),
            'mean_intensity': float(image[mask].mean()),
            'mean_y'        : float(ys.mean()),
            'mean_x'        : float(xs.mean()),
        })

    # Trier par position verticale (y croissant = haut → bas)
    props_sorted_y = sorted(props, key=lambda p: p['cy'])

    # Les 2 clusters les plus hauts → psoas
    top2    = sorted(props_sorted_y[:3],
                     key=lambda p: p['cx'])
    bottom4 = sorted(props_sorted_y[3:],
                     key=lambda p: p['cx'])

    # Si moins de 2 clusters en haut, ajuster
    if len(top2) < 2:
        top2    = sorted(props[:2], key=lambda p: p['cx'])
        bottom4 = sorted(props[2:], key=lambda p: p['cx'])

    # Attribution anatomique :
    # top2[0] = psoas gauche (x < W/2 dans image = droite anatomique)
    # top2[1] = psoas droit
    # bottom4 : de gauche à droite
    #   [0] = érecteur gauche
    #   [1] = multifidus gauche
    #   [2] = multifidus droit
    #   [3] = érecteur droit

    assignment = {}  # cluster_label → index dans REGION_NAMES

    if len(top2) >= 2:
        # Psoas : celui de droite de l'image = psoas gauche (anatomie)
        if top2[0]['cx'] > top2[1]['cx']:
            top2[0], top2[1] = top2[1], top2[0]
        assignment[top2[0]['label']] = 0  # psoas_gauche
        assignment[top2[1]['label']] = 1  # psoas_droit
    elif len(top2) == 1:
        assignment[top2[0]['label']] = 0

    if len(bottom4) >= 4:
        # Érecteurs aux extrémités, multifidus au centre
        bottom4_sorted = sorted(bottom4, key=lambda p: p['cx'])
        assignment[bottom4_sorted[0]['label']] = 2  # érecteur gauche
        assignment[bottom4_sorted[1]['label']] = 3  # multifidus gauche
        assignment[bottom4_sorted[2]['label']] = 4  # multifidus droit
        assignment[bottom4_sorted[3]['label']] = 5  # érecteur droit
    elif len(bottom4) == 3:
        bottom4_sorted = sorted(bottom4, key=lambda p: p['cx'])
        assignment[bottom4_sorted[0]['label']] = 2
        assignment[bottom4_sorted[1]['label']] = 3
        assignment[bottom4_sorted[2]['label']] = 5
    elif len(bottom4) >= 1:
        for i, p in enumerate(
                sorted(bottom4, key=lambda p: p['cx'])):
            assignment[p['label']] = min(2 + i, 5)

    # Construire la carte anatomique
    anatomical_map = np.full((H, W), -1, dtype=np.int8)
    for k in range(n_clusters):
        if k in assignment:
            anatomical_map[seg_map == k] = assignment[k]

    # Remplir les pixels non assignés avec le voisin le plus proche
    unknown = anatomical_map == -1
    if unknown.any():
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(
            unknown, return_indices=True)
        anatomical_map[unknown] = \
            anatomical_map[idx[0][unknown], idx[1][unknown]]

    return anatomical_map


def segmentation_to_color(anatomical_map):
    """Convertit la carte anatomique en image RGB colorée."""
    H, W = anatomical_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i, color in enumerate(REGION_COLORS):
        mask = anatomical_map == i
        color_img[mask] = color
    return color_img



#  ANALYSE DE TEXTURE MANNIL PAR RÉGION

def extract_texture_mannil(image, mask):
    """Extrait les features de texture Mannil 2018.

    Features :
    - Histogramme : mean, variance, std, skewness,
                    kurtosis, entropy, p10..p90
    - GLCM : contrast, energy, homogeneity,
             correlation, entropy
    """
    pixels = image[mask].astype(np.float32)
    if len(pixels) < 10:
        return None

    # Histogramme
    counts, _ = np.histogram(pixels, bins=256,
                              range=(0, 1))
    probs     = counts / (counts.sum() + 1e-8)
    probs_nz  = probs[probs > 0]

    features = {
        'n_pixels'          : int(len(pixels)),
        'hist_mean'         : float(np.mean(pixels)),
        'hist_variance'     : float(np.var(pixels)),
        'hist_std'          : float(np.std(pixels)),
        'hist_skewness'     : float(skew(pixels)),
        'hist_kurtosis'     : float(kurtosis(pixels)),
        'hist_entropy'      : float(-np.sum(
                                probs_nz * np.log2(probs_nz))),
        'hist_p10'          : float(np.percentile(pixels, 10)),
        'hist_p25'          : float(np.percentile(pixels, 25)),
        'hist_p50'          : float(np.percentile(pixels, 50)),
        'hist_p75'          : float(np.percentile(pixels, 75)),
        'hist_p90'          : float(np.percentile(pixels, 90)),
        'hist_iqr'          : float(
                                np.percentile(pixels, 75) -
                                np.percentile(pixels, 25)),
    }

    # GLCM
    img_uint8 = (image * 255).astype(np.uint8)
    try:
        glcm = graycomatrix(
            img_uint8,
            distances=[1, 2, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        features['glcm_contrast']    = float(
            graycoprops(glcm, 'contrast').mean())
        features['glcm_energy']      = float(
            graycoprops(glcm, 'energy').mean())
        features['glcm_homogeneity'] = float(
            graycoprops(glcm, 'homogeneity').mean())
        features['glcm_correlation'] = float(
            graycoprops(glcm, 'correlation').mean())

        # Entropie GLCM
        p    = glcm[:, :, 0, 0]
        p_nz = p[p > 0]
        features['glcm_entropy'] = float(
            -np.sum(p_nz * np.log2(p_nz + 1e-10)))

    except Exception as e:
        features['glcm_contrast']    = 0.0
        features['glcm_energy']      = 0.0
        features['glcm_homogeneity'] = 0.0
        features['glcm_correlation'] = 0.0
        features['glcm_entropy']     = 0.0

    return features





def visualize_full(image, anatomical_map,
                   all_features, img_name,
                   save_path=None):
    """Figure complète : IRM + segmentation + features."""

    fig = plt.figure(figsize=(22, 12), facecolor='black')
    fig.suptitle(
        f"Segmentation paraspinale + Analyse texture Mannil 2018\n"
        f"{img_name}",
        color='white', fontsize=13, fontweight='bold')

    # Layout : 2 lignes
    # Ligne 1 : image orig | segmentation | overlay
    # Ligne 2 : histogrammes par région
    gs_top = fig.add_gridspec(
        1, 3, left=0.02, right=0.98,
        top=0.88, bottom=0.52, wspace=0.05)
    gs_bot = fig.add_gridspec(
        2, 6, left=0.02, right=0.98,
        top=0.46, bottom=0.02,
        wspace=0.3, hspace=0.4)

    img01 = np.clip(image, 0, 1)

    # Image originale
    ax = fig.add_subplot(gs_top[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('IRM originale', color='white',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    # Segmentation colorée
    ax = fig.add_subplot(gs_top[1])
    color_seg = segmentation_to_color(anatomical_map)
    ax.imshow(color_seg)
    ax.set_title('Segmentation automatique',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Légende
    patches = []
    labels_pretty = [
        'Psoas gauche', 'Psoas droit',
        'Érecteur gauche', 'Multifidus gauche',
        'Multifidus droit', 'Érecteur droit'
    ]
    for i, (name, color) in enumerate(
            zip(labels_pretty, REGION_COLORS)):
        patches.append(mpatches.Patch(
            color=np.array(color)/255,
            label=name))
    ax.legend(handles=patches,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.18),
              ncol=3,
              fontsize=7,
              facecolor='#222',
              labelcolor='white',
              framealpha=0.8)

    # Overlay
    ax = fig.add_subplot(gs_top[2])
    overlay = np.stack([img01, img01, img01], axis=-1)
    color_f = color_seg.astype(np.float32) / 255
    alpha   = 0.5
    ov      = np.clip(
        (1 - alpha) * overlay + alpha * color_f, 0, 1)
    ax.imshow(ov)
    ax.set_title('Overlay (IRM + segmentation)',
                 color='white', fontsize=10,
                 fontweight='bold')
    ax.axis('off')

    # Histogrammes + tableau features par région
    for i, (region_name, region_info) in enumerate(
            REGIONS.items()):

        feat = all_features.get(region_name)
        color_norm = np.array(
            region_info['color']) / 255

        # Histogramme
        ax_h = fig.add_subplot(gs_bot[0, i])
        ax_h.set_facecolor('#1a1a1a')

        if feat and feat['n_pixels'] > 10:
            mask_r = anatomical_map == i
            pixels = image[mask_r]
            ax_h.hist(pixels, bins=40,
                      color=color_norm,
                      edgecolor='none',
                      alpha=0.85)
            ax_h.axvline(
                feat['hist_mean'],
                color='white',
                linestyle='--',
                linewidth=1.2)
            ax_h.set_title(
                labels_pretty[i],
                color=color_norm,
                fontsize=7,
                fontweight='bold')
        else:
            ax_h.text(0.5, 0.5, 'Région vide',
                      ha='center', va='center',
                      color='gray', fontsize=8,
                      transform=ax_h.transAxes)
            ax_h.set_title(labels_pretty[i],
                           color='gray',
                           fontsize=7)

        ax_h.tick_params(colors='white',
                         labelsize=6)
        for spine in ax_h.spines.values():
            spine.set_edgecolor('#444')

        # Tableau features
        ax_t = fig.add_subplot(gs_bot[1, i])
        ax_t.axis('off')
        ax_t.set_facecolor('#1a1a1a')

        if feat:
            keys_show = [
                ('hist_mean',         'Mean'),
                ('hist_variance',     'Variance'),
                ('hist_entropy',      'Entropy H'),
                ('glcm_entropy',      'GLCM Entropy'),
                ('glcm_contrast',     'Contrast'),
                ('glcm_energy',       'Energy'),
                ('glcm_homogeneity',  'Homogeneity'),
                ('glcm_correlation',  'Correlation'),
            ]
            rows = [[k_label,
                     f"{feat.get(k_key, 0):.4f}"]
                    for k_key, k_label in keys_show
                    if k_key in feat]

            if rows:
                tbl = ax_t.table(
                    cellText   = rows,
                    colLabels  = ['Feature', 'Valeur'],
                    cellLoc    = 'center',
                    loc        = 'center',
                    bbox       = [0, 0, 1, 1]
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(6)
                for (r, c), cell in \
                        tbl.get_celld().items():
                    cell.set_facecolor(
                        '#2a2a2a' if r % 2 == 0
                        else '#1a1a1a')
                    cell.set_text_props(color='white')
                    cell.set_edgecolor('#444')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight',
                    facecolor='black')
        print(f"  [OK] Figure → {save_path}")
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
        print(f"[ERREUR] Aucune image dans : {INPUT_FOLDER}")
        exit(1)

    print(f"\n[INFO] {len(image_paths)} image(s) trouvée(s)")
    print(f"[INFO] Résultats → {OUTPUT_FOLDER}\n")

    all_patients_features = []

    for img_path in image_paths:
        img_name = img_path.stem
        print(f"\n{'='*60}")
        print(f"[IMAGE] {img_name}")
        print(f"{'='*60}")

        # Chargement + prétraitement
        try:
            image, denoised = load_and_preprocess(
                img_path, target_size=(256, 256))
            print(f"  Taille : {image.shape}")
        except Exception as e:
            print(f"  [ERREUR chargement] {e}")
            continue

        # Segmentation K-Means
        try:
            seg_map, _ = segment_6_regions(
                image, denoised, n_clusters=6)
        except Exception as e:
            print(f"  [ERREUR segmentation] {e}")
            continue

        # Attribution labels anatomiques
        try:
            anatomical_map = assign_anatomical_labels(
                seg_map, image)
        except Exception as e:
            print(f"  [ERREUR labels anatomiques] {e}")
            continue

        # Analyse texture Mannil par région
        all_features = {}
        print("  Extraction features de texture :")
        for i, region_name in enumerate(REGION_NAMES):
            mask_r = anatomical_map == i
            n_pix  = mask_r.sum()
            feat   = extract_texture_mannil(image, mask_r)
            all_features[region_name] = feat

            if feat:
                print(f"    {region_name:20s} : "
                      f"{n_pix:5d} px | "
                      f"mean={feat['hist_mean']:.3f} | "
                      f"entropy={feat['hist_entropy']:.3f} | "
                      f"glcm_entropy="
                      f"{feat['glcm_entropy']:.3f}")
            else:
                print(f"    {region_name:20s} : "
                      f"région trop petite")

        # Visualisation
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{img_name}_spine_mannil.png")
        visualize_full(
            image, anatomical_map,
            all_features, img_name,
            save_path=fig_path)

        # Sauvegarde CSV par patient
        patient_row = {'patient': img_name}
        for region_name, feat in all_features.items():
            if feat:
                for k, v in feat.items():
                    patient_row[
                        f"{region_name}_{k}"] = v
        all_patients_features.append(patient_row)

        # Sauvegarde masque segmentation (numpy)
        npy_path = os.path.join(
            OUTPUT_FOLDER,
            f"{img_name}_segmentation.npy")
        np.save(npy_path, anatomical_map)

        # Sauvegarde segmentation colorée (PNG)
        color_png = os.path.join(
            OUTPUT_FOLDER,
            f"{img_name}_segmentation_color.png")
        color_img = segmentation_to_color(anatomical_map)
        Image.fromarray(color_img).save(color_png)

    # CSV global tous patients
    if all_patients_features:
        df = pd.DataFrame(all_patients_features)
        csv_path = os.path.join(
            OUTPUT_FOLDER,
            'all_patients_features.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[CSV] → {csv_path}")
        print(f"  {df.shape[0]} patients × "
              f"{df.shape[1]} features")

        # Résumé statistique par région
        print("\n[RÉSUMÉ] Moyennes par région (hist_mean) :")
        for region_name in REGION_NAMES:
            col = f"{region_name}_hist_mean"
            if col in df.columns:
                print(f"  {region_name:22s} : "
                      f"{df[col].mean():.4f} "
                      f"± {df[col].std():.4f}")

    print(f"\n[TERMINÉ] Tous les résultats dans :")
    print(f"  {OUTPUT_FOLDER}")