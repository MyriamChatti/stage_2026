# ============================================================
# LIME OPTIMISÉ IRM — REGROUPEMENT ANATOMIQUE + PROPAGATION
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.ndimage import gaussian_filter, binary_closing, distance_transform_edt
from skimage.segmentation import slic
from skimage import morphology
from skimage.filters import sobel

# ============================================================
# PATHS
# ============================================================

INPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/LIME_results_final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224,224))
    return np.array(img)/255.0

# ============================================================
# SUPERPIXELS
# ============================================================

def segment_image(image):
    return slic(
        image,
        n_segments=120,
        compactness=5,
        sigma=1,
        start_label=0
    )

# ============================================================
#  REGROUPEMENT ANATOMIQUE
# ============================================================

def merge_superpixels_anatomical(image, segments, n_groups=12):

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    H, W = image.shape[:2]
    gray = image[:,:,0]
    grad = sobel(gray)

    features = []

    for k in range(segments.max()+1):

        m = segments == k

        if m.sum() < 30:
            features.append([0]*6)
            continue

        y, x = np.where(m)

        features.append([
            gray[m].mean(),
            gray[m].std(),
            y.mean()/H,
            x.mean()/W,
            np.sqrt((y.mean()/H-0.5)**2+(x.mean()/W-0.5)**2),
            grad[m].mean()
        ])

    features = np.array(features)
    features = StandardScaler().fit_transform(features)

    labels = KMeans(n_clusters=n_groups, random_state=42, n_init=20).fit_predict(features)

    merged = np.zeros_like(segments)

    for k in range(len(labels)):
        merged[segments == k] = labels[k]

    return merged

# ============================================================
#  SÉPARATION GAUCHE / DROITE
# ============================================================

def split_left_right(seg, image):

    H, W = image.shape[:2]
    mid = W // 2

    new_seg = seg.copy()
    max_label = seg.max()

    for k in range(max_label+1):

        mask = seg == k

        if mask.sum() < 50:
            continue

        left = mask.copy()
        left[:, mid:] = False

        right = mask.copy()
        right[:, :mid] = False

        if left.sum() > 0 and right.sum() > 0:
            new_seg[right] = max_label + 1
            max_label += 1

    return new_seg

# ============================================================
#  SUPPRESSION FOND
# ============================================================

def remove_background(seg_map, image):

    H, W = image.shape[:2]
    gray = image[:,:,0]

    props = []

    for k in range(seg_map.max()+1):

        m = seg_map == k
        if m.sum() < 20:
            continue

        y,x = np.where(m)

        props.append({
            'k':k,
            'mean':gray[m].mean(),
            'dist':np.sqrt((y.mean()/H-0.5)**2+(x.mean()/W-0.5)**2)
        })

    fond = min(props, key=lambda p: p['mean'] + p['dist'])

    seg_map[seg_map == fond['k']] = 0

    return seg_map

# ============================================================
#  PROPAGATION + CLEAN
# ============================================================

def refine(seg):

    refined = seg.copy()

    for i in range(1, seg.max()+1):

        m = refined == i

        m = morphology.remove_small_objects(m, 150)
        m = binary_closing(m, morphology.disk(3))

        refined[refined == i] = 0
        refined[m] = i

    unknown = refined == 0

    if unknown.any():
        _, idx = distance_transform_edt(unknown, return_indices=True)
        refined[unknown] = refined[idx[0], idx[1]][unknown]

    return refined

# ============================================================
# MODEL LIME ADAPTÉ IRM
# ============================================================

def model_fn(image):
    gray = image[:,:,0]
    return float(gray.mean() + sobel(gray).mean())

# ============================================================
# LIME CORE
# ============================================================

def compute_lime(image, segments, n_samples=400):

    n_labels = segments.max()+1

    perturbations = np.random.randint(0,2,(n_samples,n_labels))
    scores = np.zeros(n_samples)

    baseline = gaussian_filter(image, sigma=10)

    for i in range(n_samples):

        mask = perturbations[i]
        perturbed = baseline.copy()

        for k in range(n_labels):
            if mask[k] == 1:
                perturbed[segments == k] = image[segments == k]

        scores[i] = model_fn(perturbed)

    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)
    model.fit(perturbations, scores)

    coefs = model.coef_

    saliency = np.zeros(image.shape[:2])

    for k in range(n_labels):
        saliency[segments == k] = coefs[k]

    saliency = np.maximum(saliency,0)
    saliency = saliency / (saliency.max()+1e-8)

    return saliency




def visualize(image, seg, saliency, name):

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("IRM")

    plt.subplot(1,3,2)
    plt.imshow(seg, cmap='tab20')
    plt.title("Segmentation anatomique")

    plt.subplot(1,3,3)
    plt.imshow(image)
    plt.imshow(saliency, cmap='jet', alpha=0.5)
    plt.title("LIME optimisé")

    plt.savefig(f"{OUTPUT_FOLDER}/{name}")
    plt.close()





if __name__ == "__main__":

    images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".png")]

    for f in images:

        print("Processing", f)

        path = os.path.join(INPUT_FOLDER, f)

        img = load_image(path)

        seg = segment_image(img)

        seg = merge_superpixels_anatomical(img, seg)
        seg = split_left_right(seg, img)
        seg = remove_background(seg, img)
        seg = refine(seg)

        sal = compute_lime(img, seg)

        visualize(img, seg, sal, f)

    print("\n DONE — LIME optimisé terminé")