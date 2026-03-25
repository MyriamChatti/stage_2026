"""
============================================================
  mannil_texture_analysis.py
  Reproduction de la méthode Mannil et al. 2018
  "Texture analysis of paraspinal musculature in MRI
  of the lumbar spine" — Skeletal Radiology 2018

  Méthode originale : logiciel MaZda + 151 features de
  texture + machine learning + régression logistique.

  Cette implémentation Python utilise :
    - PyRadiomics  → extraction des 151+ features de texture
    - Scikit-learn → machine learning + régression logistique
    - SimpleITK    → lecture NIfTI / DICOM
    - Pillow       → lecture PNG / JPG

  Installation :
      pip install pyradiomics scikit-learn SimpleITK
                  numpy matplotlib pandas pillow scipy

  Usage :
      python mannil_texture_analysis.py
============================================================
"""
#chemin

IMAGE_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
MASK_FOLDER   = None
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/mannil_results"
IMAGE_FORMAT  = "png"
SPINAL_LEVEL  = "L3/L4"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path




# chargement images

def load_image(path):
    """Charge une image IRM en numpy array 2D (float32).

    Supporte : PNG, JPG, NIfTI (.nii/.nii.gz), DICOM (.dcm)
    Retourne : ndarray (H, W) float32 normalisé dans [0, 1]
    """
    path = str(path)
    ext  = path.lower()

    if ext.endswith('.nii') or ext.endswith('.nii.gz'):
        import SimpleITK as sitk
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # NIfTI 3D → prendre la coupe du milieu
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    elif ext.endswith('.dcm'):
        import SimpleITK as sitk
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    else:  # PNG, JPG
        from PIL import Image
        img = Image.open(path).convert('L')  # niveaux de gris
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr


def load_mask(path):
    """Charge un masque binaire en numpy array 2D bool."""
    if path is None:
        return None
    path = str(path)
    ext  = path.lower()

    if ext.endswith('.nii') or ext.endswith('.nii.gz'):
        import SimpleITK as sitk
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        return arr.astype(bool)

    else:
        from PIL import Image
        img = Image.open(path).convert('L')
        arr = np.array(img)
        return arr > 127


# GÉNÉRATION AUTOMATIQUE
# Reproduction de la sélection manuelle de MaZda via seuillage Otsu + sélection de la région centrale

def generate_roi_otsu(image, roi_type='central',
                      margin_frac=0.15):
    """Génère une ROI automatique par seuillage Otsu.

    Args:
        image      : ndarray (H, W) float32 [0, 1]
        roi_type   : 'central' | 'muscle_left' | 'muscle_right'
                     | 'full'
        margin_frac: fraction de marge autour du centre

    Returns:
        mask : ndarray (H, W) bool
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import (binary_closing, binary_opening,
                                    disk, remove_small_objects)
    from skimage.measure import label, regionprops

    H, W  = image.shape
    thresh = threshold_otsu(image)
    binary = image > thresh

    # Nettoyage morphologique
    binary = binary_opening(binary,  disk(3))
    binary = binary_closing(binary,  disk(5))
    binary = remove_small_objects(binary, min_size=200)

    if roi_type == 'full':
        return binary

    elif roi_type == 'central':
        # Zone centrale = disque intervertébral + sac thécal
        mh = int(H * margin_frac)
        mw = int(W * margin_frac)
        mask = np.zeros((H, W), dtype=bool)
        mask[H//4 + mh : 3*H//4 - mh,
             W//4 + mw : 3*W//4 - mw] = True
        return binary & mask

    elif roi_type == 'muscle_left':
        # Muscle paraspinal gauche
        mask = np.zeros((H, W), dtype=bool)
        mask[H//4 : 3*H//4, W//2 : int(W * 0.85)] = True
        return binary & mask

    elif roi_type == 'muscle_right':
        # Muscle paraspinal droit
        mask = np.zeros((H, W), dtype=bool)
        mask[H//4 : 3*H//4, int(W * 0.15) : W//2] = True
        return binary & mask

    return binary


# EXTRACTION DES FEATURES DE TEXTURE (méthode MaZda)
#     151 features extraites par PyRadiomics :
#     - Histogramme (mean, variance, skewness, kurtosis)
#     - GLCM (entropie, énergie, corrélation, contraste)
#     - GLRLM (run length)
#     - GLSZM (zone size)
#     - NGTDM


def extract_texture_features_pyradiomics(image, mask):
    """Extrait les features de texture via PyRadiomics.

    Reproduit les 151 features de MaZda utilisées par Mannil.

    Args:
        image : ndarray (H, W) float32 [0, 1]
        mask  : ndarray (H, W) bool

    Returns:
        features : dict {nom_feature: valeur}
    """
    import SimpleITK as sitk
    from radiomics import featureextractor

    # Conversion en SimpleITK (requis par PyRadiomics)
    # Rescale dans [0, 255] pour compatibilité
    img_uint = (image * 255).astype(np.uint16)
    msk_uint = mask.astype(np.uint8)

    # Ajout dimension z=1 pour SimpleITK (2D → 3D)
    img_sitk = sitk.GetImageFromArray(img_uint[np.newaxis, ...])
    msk_sitk = sitk.GetImageFromArray(msk_uint[np.newaxis, ...])

    # Configuration PyRadiomics (équivalent MaZda)
    params = {
        'imageType'   : {'Original': {}},
        'featureClass': {
            'firstorder' : [],   # histogramme (mean, variance...)
            'glcm'       : [],   # co-occurrence (entropie, énergie...)
            'glrlm'      : [],   # run length matrix
            'glszm'      : [],   # zone size matrix
            'ngtdm'      : [],   # neighborhood gray tone difference
        },
        'setting': {
            'binWidth'          : 25,
            'resampledPixelSpacing': None,
            'interpolator'      : 'sitkBSpline',
            'verbose'           : False,
        }
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('ngtdm')

    result = extractor.execute(img_sitk, msk_sitk)

    # Filtrer uniquement les features numériques
    features = {}
    for k, v in result.items():
        if k.startswith('original_'):
            try:
                features[k] = float(v)
            except (TypeError, ValueError):
                pass

    return features


def extract_texture_features_manual(image, mask):
    """Extraction manuelle des features de texture (sans PyRadiomics).

    Fallback si PyRadiomics n'est pas installé.
    Reproduit les principales features MaZda :
      - Histogramme : mean, variance, skewness, kurtosis, entropy
      - GLCM : energy, contrast, correlation, homogeneity, entropy
    """
    from scipy.stats import skew, kurtosis
    from skimage.feature import graycomatrix, graycoprops

    pixels = image[mask].astype(np.float32)

    if len(pixels) == 0:
        return {}

    #Features histogramme
    hist_mean     = float(np.mean(pixels))
    hist_var      = float(np.var(pixels))
    hist_std      = float(np.std(pixels))
    hist_skew     = float(skew(pixels))
    hist_kurt     = float(kurtosis(pixels))
    hist_p10      = float(np.percentile(pixels, 10))
    hist_p25      = float(np.percentile(pixels, 25))
    hist_p50      = float(np.percentile(pixels, 50))
    hist_p75      = float(np.percentile(pixels, 75))
    hist_p90      = float(np.percentile(pixels, 90))

    # Entropie de Shannon
    counts, _     = np.histogram(pixels, bins=256, range=(0, 1))
    probs         = counts / (counts.sum() + 1e-8)
    probs         = probs[probs > 0]
    hist_entropy  = float(-np.sum(probs * np.log2(probs)))

    # Features GLCM (Gray Level Co-occurrence Matrix)
    img_uint8 = (image * 255).astype(np.uint8)
    # Distances et angles comme dans MaZda
    distances = [1, 2, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm_features = {
        'contrast'   : [],
        'energy'     : [],
        'homogeneity': [],
        'correlation': [],
        'entropy_glcm': [],
    }

    for d in distances:
        glcm = graycomatrix(img_uint8,
                            distances=[d],
                            angles=angles,
                            levels=256,
                            symmetric=True,
                            normed=True)

        glcm_features['contrast'   ].append(
            float(graycoprops(glcm, 'contrast'   ).mean()))
        glcm_features['energy'     ].append(
            float(graycoprops(glcm, 'energy'     ).mean()))
        glcm_features['homogeneity'].append(
            float(graycoprops(glcm, 'homogeneity').mean()))
        glcm_features['correlation'].append(
            float(graycoprops(glcm, 'correlation').mean()))

        # Entropie GLCM manuelle
        p    = glcm[:, :, 0, 0]
        p    = p[p > 0]
        entr = float(-np.sum(p * np.log2(p + 1e-10)))
        glcm_features['entropy_glcm'].append(entr)

    # Moyennes sur les distances
    features = {
        'histogram_mean'        : hist_mean,
        'histogram_variance'    : hist_var,
        'histogram_std'         : hist_std,
        'histogram_skewness'    : hist_skew,
        'histogram_kurtosis'    : hist_kurt,
        'histogram_p10'         : hist_p10,
        'histogram_p25'         : hist_p25,
        'histogram_p50'         : hist_p50,
        'histogram_p75'         : hist_p75,
        'histogram_p90'         : hist_p90,
        'histogram_entropy'     : hist_entropy,
        'glcm_contrast'         : float(np.mean(glcm_features['contrast'])),
        'glcm_energy'           : float(np.mean(glcm_features['energy'])),
        'glcm_homogeneity'      : float(np.mean(glcm_features['homogeneity'])),
        'glcm_correlation'      : float(np.mean(glcm_features['correlation'])),
        'glcm_entropy'          : float(np.mean(glcm_features['entropy_glcm'])),
    }

    return features


def extract_features(image, mask, use_pyradiomics=True):
    """Wrapper : essaie PyRadiomics, sinon fallback manuel."""
    if use_pyradiomics:
        try:
            return extract_texture_features_pyradiomics(image, mask)
        except Exception as e:
            print(f"  [PyRadiomics indisponible : {e}] → fallback manuel")

    return extract_texture_features_manual(image, mask)



# MACHINE LEARNING
#     - Régression logistique
#     - SVM
#     - Random Forest
#     + Validation croisée 5-fold

def run_machine_learning(df_features, labels=None,
                         test_size=0.2, random_state=42):
    """Entraîne plusieurs classifieurs sur les features extraites.

    Si labels=None, fait une analyse non supervisée (clustering).
    Si labels fournis, fait de la classification supervisée.

    Args:
        df_features : DataFrame (n_images, n_features)
        labels      : array (n_images,) ou None
        test_size   : fraction de test
        random_state: graine aléatoire

    Returns:
        results : dict avec scores et importances
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (cross_val_score,
                                         StratifiedKFold)
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline

    X = df_features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {'feature_names': list(df_features.columns)}

    # Analyse non supervisée si pas de labels
    if labels is None:
        print("\n[ML] Pas de labels → analyse non supervisée")

        # PCA pour visualisation
        n_comp = min(2, X_scaled.shape[1],
                     X_scaled.shape[0] - 1)
        pca    = PCA(n_components=n_comp)
        X_pca  = pca.fit_transform(X_scaled)

        # K-Means
        n_clusters = min(3, X_scaled.shape[0])
        kmeans     = KMeans(n_clusters=n_clusters,
                            random_state=random_state,
                            n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        results['pca']           = X_pca
        results['pca_variance']  = pca.explained_variance_ratio_
        results['clusters']      = cluster_labels
        results['n_clusters']    = n_clusters
        results['supervised']    = False

        print(f"  PCA variance expliquée : "
              f"{pca.explained_variance_ratio_}")
        print(f"  Clusters K-Means : {cluster_labels}")
        return results

    # Classification supervisée
    print(f"\n[ML] Classification supervisée "
          f"({len(np.unique(labels))} classes)")

    results['supervised'] = True
    results['labels']     = labels
    cv = StratifiedKFold(n_splits=min(5, len(labels)),
                         shuffle=True,
                         random_state=random_state)

    classifiers = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(max_iter=1000,
                                          random_state=random_state))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    SVC(kernel='rbf', probability=True,
                           random_state=random_state))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    RandomForestClassifier(
                           n_estimators=100,
                           random_state=random_state))
        ]),
    }

    scores = {}
    for name, clf in classifiers.items():
        try:
            cv_scores = cross_val_score(
                clf, X, labels, cv=cv, scoring='accuracy')
            scores[name] = {
                'mean' : float(cv_scores.mean()),
                'std'  : float(cv_scores.std()),
                'scores': cv_scores.tolist()
            }
            print(f"  {name:25s} : "
                  f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as e:
            print(f"  {name} : ERREUR ({e})")

    # Random Forest
    try:
        rf = classifiers['Random Forest']
        rf.fit(X, labels)
        importances = rf.named_steps['clf'].feature_importances_
        feat_imp    = pd.Series(
            importances,
            index=df_features.columns
        ).sort_values(ascending=False)
        results['feature_importances'] = feat_imp
        print(f"\n  Top 5 features importantes :")
        for fname, fimp in feat_imp.head(5).items():
            print(f"    {fname:40s} : {fimp:.4f}")
    except Exception as e:
        print(f"  [Feature importances] ERREUR : {e}")

    results['cv_scores'] = scores

    # PCA pour visualisation
    n_comp  = min(2, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca     = PCA(n_components=n_comp)
    X_pca   = pca.fit_transform(X_scaled)
    results['pca']          = X_pca
    results['pca_variance'] = pca.explained_variance_ratio_

    return results


#visualisation
def visualize_results(image, mask, features, img_name,
                      save_path=None):
    """Figure de synthèse : image + ROI + heatmap + features."""

    fig = plt.figure(figsize=(18, 5), facecolor='black')
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.06)

    img01 = np.clip(image, 0, 1)

    # Image originale 
    ax = fig.add_subplot(gs[0])
    ax.imshow(img01, cmap='gray')
    ax.set_title('Image originale', color='white',
                 fontsize=9, fontweight='bold')
    ax.axis('off')

    # ROI sur image
    ax = fig.add_subplot(gs[1])
    overlay = np.stack([img01, img01, img01], axis=-1)
    overlay[mask, 0] = 0.9
    overlay[mask, 1] = 0.3
    overlay[mask, 2] = 0.1
    ax.imshow(overlay)
    ax.set_title('ROI analysée', color='white',
                 fontsize=9, fontweight='bold')
    ax.axis('off')

    # Heatmap des valeurs de gris dans la ROI
    ax = fig.add_subplot(gs[2])
    roi_img = np.zeros_like(img01)
    roi_img[mask] = img01[mask]
    im = ax.imshow(roi_img, cmap='hot', vmin=0, vmax=1)
    ax.set_title('Intensité ROI', color='white',
                 fontsize=9, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Histogramme des pixels ROI
    ax = fig.add_subplot(gs[3])
    ax.set_facecolor('#1a1a1a')
    pixels = img01[mask]
    ax.hist(pixels, bins=50, color='#3498db',
            edgecolor='none', alpha=0.85)
    if 'histogram_mean' in features:
        ax.axvline(features['histogram_mean'],
                   color='white', linestyle='--',
                   linewidth=1.5,
                   label=f"mean={features['histogram_mean']:.3f}")
    ax.set_title('Histogramme ROI', color='white',
                 fontsize=9, fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#333', labelcolor='white',
              fontsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    # Tableau des top features
    ax = fig.add_subplot(gs[4])
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')

    # Sélection des features clés de Mannil
    mannil_keys = [
        'histogram_mean', 'histogram_variance',
        'histogram_entropy', 'glcm_entropy',
        'glcm_contrast', 'glcm_energy',
        'glcm_homogeneity', 'glcm_correlation',
    ]
    # Chercher aussi les clés PyRadiomics équivalentes
    pyrad_map = {
        'histogram_mean'     : 'original_firstorder_Mean',
        'histogram_variance' : 'original_firstorder_Variance',
        'histogram_entropy'  : 'original_firstorder_Entropy',
        'glcm_entropy'       : 'original_glcm_JointEntropy',
        'glcm_contrast'      : 'original_glcm_Contrast',
        'glcm_energy'        : 'original_glcm_JointEnergy',
        'glcm_homogeneity'   : 'original_glcm_Imc1',
        'glcm_correlation'   : 'original_glcm_Correlation',
    }

    rows = []
    for short_key, pyrad_key in pyrad_map.items():
        val = features.get(short_key,
              features.get(pyrad_key, None))
        if val is not None:
            label = short_key.replace('histogram_', 'hist_')
            rows.append([label, f"{val:.4f}"])

    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=['Feature', 'Valeur'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor('#2a2a2a' if r % 2 == 0
                               else '#1a1a1a')
            cell.set_text_props(color='white')
            cell.set_edgecolor('#444')

    ax.set_title('Features Mannil', color='white',
                 fontsize=9, fontweight='bold')

    fig.suptitle(f"Analyse texture Mannil 2018 — {img_name} "
                 f"({SPINAL_LEVEL})",
                 color='white', fontsize=11,
                 fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor='black')
        print(f"  [OK] Figure → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_ml_results(results, df_features,
                         save_path=None):
    """Visualise les résultats ML : PCA + feature importances."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             facecolor='black')

    # PCA
    ax = axes[0]
    ax.set_facecolor('#1a1a1a')
    X_pca = results.get('pca')

    if X_pca is not None and X_pca.shape[1] >= 2:
        if results.get('supervised') and \
                results.get('labels') is not None:
            labels  = results['labels']
            classes = np.unique(labels)
            colors  = ['#3498db', '#e74c3c', '#2ecc71',
                       '#f39c12', '#9b59b6']
            for i, cls in enumerate(classes):
                mask = labels == cls
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           color=colors[i % len(colors)],
                           label=str(cls), alpha=0.8, s=60)
            ax.legend(facecolor='#333', labelcolor='white')
        else:
            clusters = results.get('clusters',
                                   np.zeros(len(X_pca)))
            scatter  = ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=clusters, cmap='Set1', alpha=0.8, s=60)
            plt.colorbar(scatter, ax=ax)

        var = results.get('pca_variance', [0, 0])
        ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)',
                      color='white')
        ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)',
                      color='white')

    ax.set_title('PCA des features de texture',
                 color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    #  Feature importances (si Random Forest)
    ax = axes[1]
    ax.set_facecolor('#1a1a1a')

    feat_imp = results.get('feature_importances')
    if feat_imp is not None:
        top20   = feat_imp.head(20)
        colors  = ['#2ecc71' if v > top20.mean()
                   else '#3498db'
                   for v in top20.values]
        ax.barh(range(len(top20)), top20.values[::-1],
                color=colors[::-1], edgecolor='none')
        ax.set_yticks(range(len(top20)))
        labels_short = [n.replace('original_', '')
                        .replace('firstorder_', 'hist_')
                        .replace('glcm_', 'glcm_')[:25]
                        for n in top20.index[::-1]]
        ax.set_yticklabels(labels_short,
                           color='white', fontsize=7)
        ax.set_title('Top 20 features importantes (RF)',
                     color='white', fontweight='bold')
    else:
        feat_names = results.get('feature_names', [])
        ax.text(0.5, 0.5,
                f'{len(feat_names)} features extraites\n'
                f'(labels requis pour importances)',
                ha='center', va='center',
                color='white', fontsize=11,
                transform=ax.transAxes)
        ax.set_title('Features extraites',
                     color='white', fontweight='bold')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    if results.get('cv_scores'):
        title_suffix = " | Accuracy CV : " + " / ".join(
            f"{n.split()[0]}={v['mean']:.2f}"
            for n, v in results['cv_scores'].items()
        )
    else:
        title_suffix = ""

    fig.suptitle(f"Résultats ML — Mannil 2018{title_suffix}",
                 color='white', fontsize=11,
                 fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor='black')
        print(f"  [OK] Figure ML → {save_path}")
    else:
        plt.show()
    plt.close(fig)





if __name__ == '__main__':

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg',
                      '.nii', '.dcm', '.tiff', '.tif'}

    #  Collecte des images
    image_paths = sorted([
        Path(IMAGE_FOLDER) / f
        for f in os.listdir(IMAGE_FOLDER)
        if Path(f).suffix.lower() in IMG_EXTENSIONS
        or f.lower().endswith('.nii.gz')
    ])

    if not image_paths:
        print(f"[ERREUR] Aucune image trouvée dans : "
              f"{IMAGE_FOLDER}")
        exit(1)

    print(f"\n[INFO] {len(image_paths)} image(s) trouvée(s)")
    print(f"[INFO] Résultats → {OUTPUT_FOLDER}\n")

    all_features = []
    all_names    = []

    for img_path in image_paths:
        img_name = img_path.stem
        print(f"\n{'='*60}")
        print(f"[IMAGE] {img_name}")
        print(f"{'='*60}")

        # chargement image
        try:
            image = load_image(img_path)
        except Exception as e:
            print(f"  [ERREUR chargement] {e}")
            continue

        # chargement ou génération du masque
        mask = None
        if MASK_FOLDER:
            for ext in ['.png', '.jpg', '.nii',
                        '.nii.gz', '.tiff']:
                mp = Path(MASK_FOLDER) / (img_name + ext)
                if mp.exists():
                    mask = load_mask(mp)
                    print(f"  Masque chargé : {mp.name}")
                    break

        if mask is None:
            print("  Masque non trouvé → génération Otsu...")
            # Générer 3 ROI : centrale + muscles gauche/droite  = zone de l'image qu'on veut analyser
            mask = generate_roi_otsu(image,
                                     roi_type='central')
            n_pixels = mask.sum()
            print(f"  ROI centrale : {n_pixels} pixels")
            if n_pixels < 50:
                print("  ROI trop petite → ROI complète")
                mask = generate_roi_otsu(image,
                                         roi_type='full')

        # extraction des features
        print(f"  Extraction des features de texture...")
        try:
            features = extract_features(image, mask,
                                        use_pyradiomics=True)
            print(f"  {len(features)} features extraites")
        except Exception as e:
            print(f"  [ERREUR features] {e}")
            continue

        all_features.append(features)
        all_names.append(img_name)

        # visualisation par image
        fig_path = os.path.join(
            OUTPUT_FOLDER,
            f"{img_name}_mannil_texture.png")
        visualize_results(image, mask, features,
                          img_name, save_path=fig_path)

    # Synthèse
    if not all_features:
        print("\n[ERREUR] Aucune feature extraite.")
        exit(1)

    print(f"\n{'='*60}")
    print(f"[SYNTHÈSE] {len(all_features)} images traitées")
    print(f"{'='*60}")

    # DataFrame de toutes les features
    df = pd.DataFrame(all_features, index=all_names)
    df = df.fillna(0)

    # Sauvegarde CSV
    csv_path = os.path.join(OUTPUT_FOLDER,
                            'mannil_features_all.csv')
    df.to_csv(csv_path)
    print(f"\n[CSV] Features sauvegardées → {csv_path}")
    print(f"  Dimensions : {df.shape[0]} images × "
          f"{df.shape[1]} features")

    # Machine Learning
    print("\n[ML] Analyse statistique des features...")

    # Sans labels → clustering non supervisé
    # Pour ajouter des labels : labels = np.array([0,1,0,1,...])
    labels = None

    ml_results = run_machine_learning(df, labels=labels)

    ml_fig_path = os.path.join(OUTPUT_FOLDER,
                               'mannil_ml_results.png')
    visualize_ml_results(ml_results, df,
                         save_path=ml_fig_path)

    # Rapport texte
    report_path = os.path.join(OUTPUT_FOLDER,
                               'mannil_rapport.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RAPPORT MANNIL 2018 — Analyse de texture IRM\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Niveau analysé : {SPINAL_LEVEL}\n")
        f.write(f"Images traitées : {len(all_names)}\n")
        f.write(f"Features extraites : {df.shape[1]}\n\n")
        f.write("--- Statistiques globales ---\n")
        f.write(df.describe().to_string())
        f.write("\n\n--- Features clés (Mannil 2018) ---\n")
        mannil_keys = ['histogram_mean',
                       'histogram_variance',
                       'histogram_entropy',
                       'glcm_entropy',
                       'glcm_contrast']
        for k in mannil_keys:
            cols = [c for c in df.columns if k in c.lower()]
            for c in cols[:1]:
                f.write(f"{c:50s} : "
                        f"μ={df[c].mean():.4f} "
                        f"σ={df[c].std():.4f}\n")

    print(f"[RAPPORT] → {report_path}")
    print(f"\n[TERMINÉ] Tous les résultats dans :")
    print(f"  {OUTPUT_FOLDER}")