# ============================================================
# UNSAM SPINE SEGMENTATION — VERSION OPTIMISÉE STABLE
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.ndimage import gaussian_filter, binary_closing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from skimage.segmentation import slic
from skimage import morphology
from skimage.filters import sobel

import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
INPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/unsam_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------------------------------------
# PARAMÈTRES
# -------------------------------------------------------------
N_REGIONS = 14

# -------------------------------------------------------------
# COULEURS
# -------------------------------------------------------------
REGIONS = {
    0: ('Fond', [30,30,30]),
    1: ('Disque', [255,200,0]),
    2: ('Sac centre', [0,180,255]),
    3: ('Sac périphérie', [0,100,200]),
    4: ('Éminence', [180,90,0]),
    5: ('Psoas G', [220,60,180]),
    6: ('Psoas D', [80,200,60]),
    7: ('Multifidus G sup', [50,200,100]),
    8: ('Multifidus G prof', [30,120,60]),
    9: ('Multifidus D sup', [180,80,220]),
    10: ('Multifidus D prof', [100,40,140]),
    11: ('Érecteur G', [40,80,200]),
    12: ('Érecteur D', [210,190,40]),
    13: ('Graisse', [255,160,100]),
}

# -------------------------------------------------------------
# PREPROCESS
# -------------------------------------------------------------
def preprocess(path):
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32)
    arr = (arr - arr.min())/(arr.max()-arr.min()+1e-8)
    return arr

# -------------------------------------------------------------
# FEATURES (AMÉLIORÉES)
# -------------------------------------------------------------
def extract_features(img):

    H, W = img.shape

    img_smooth = gaussian_filter(img, sigma=1)

    yy, xx = np.mgrid[0:H, 0:W]

    grad = sobel(img_smooth)

    center_y, center_x = H//2, W//2
    dist_center = np.sqrt((yy-center_y)**2 + (xx-center_x)**2)
    dist_center /= dist_center.max()

    vertical = np.abs(xx - center_x) / W

    return np.stack([
        img_smooth,
        grad,
        yy/H,
        xx/W,
        dist_center,
        vertical
    ], axis=-1)

# -------------------------------------------------------------
# SEGMENTATION (SUPERPIXELS)
# -------------------------------------------------------------
def segment(img):

    segments = slic(img, n_segments=300, compactness=10, start_label=0)

    feat = extract_features(img)

    seg_map = np.zeros_like(segments)

    features = []

    for s in np.unique(segments):
        mask = segments == s
        features.append(feat[mask].mean(axis=0))

    features = np.array(features)
    features = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=N_REGIONS, random_state=42, n_init=10)
    labels = km.fit_predict(features)

    for i, s in enumerate(np.unique(segments)):
        seg_map[segments == s] = labels[i]

    return seg_map

# -------------------------------------------------------------
# ANATOMY ASSIGNMENT (FIX + STABLE)
# -------------------------------------------------------------
def assign_anatomy(seg_map, img):

    H, W = img.shape

    regions = []

    for k in range(int(seg_map.max()) + 1):
        m = seg_map == k
        if m.sum() < 30:
            continue

        ys, xs = np.where(m)

        regions.append({
            'k': k,
            'mean': img[m].mean(),
            'y': ys.mean()/H,
            'x': xs.mean()/W,
            'dist': np.sqrt((ys.mean()/H - 0.5)**2 + (xs.mean()/W - 0.5)**2)
        })

    remaining = regions.copy()

    def pop_best(score):
        best = min(remaining, key=score)
        remaining.remove(best)
        return best['k']

    # structures principales
    sac_center = pop_best(lambda p: -p['mean']*7 + p['dist']*12)
    sac_periph = pop_best(lambda p: -p['mean']*4 + p['dist']*8)

    disc = pop_best(lambda p: abs(p['mean']-0.45)*5 + (p['y']-0.65)**2*6)

    eminence = pop_best(lambda p: p['mean']*2 + p['y']*5)

    graisse = pop_best(lambda p: -p['mean'] + p['dist']*2)

    fond = pop_best(lambda p: p['mean'] + p['dist'])

    # muscles gauche/droite
    left = sorted([p for p in remaining if p['x'] < 0.5], key=lambda p: p['dist'])
    right = sorted([p for p in remaining if p['x'] >= 0.5], key=lambda p: p['dist'])

    def assign_side(regs, side):
        res = {}
        if side == 'L':
            psoas, mf_prof, mf_sup, erect = 5,8,7,11
        else:
            psoas, mf_prof, mf_sup, erect = 6,10,9,12

        if len(regs)>0: res[regs[0]['k']] = psoas
        if len(regs)>1: res[regs[1]['k']] = mf_prof
        if len(regs)>2: res[regs[2]['k']] = mf_sup

        for r in regs[3:]:
            res[r['k']] = erect

        return res

    mapping = {
        fond:0,
        disc:1,
        sac_center:2,
        sac_periph:3,
        eminence:4,
        graisse:13
    }

    mapping.update(assign_side(left,'L'))
    mapping.update(assign_side(right,'R'))

    anat = np.zeros_like(seg_map)

    for k in range(int(seg_map.max())+1):
        anat[seg_map==k] = mapping.get(k,0)

    # nettoyage morphologique
    for i in range(1,14):
        mask = anat==i
        mask = morphology.remove_small_objects(mask, min_size=50)
        mask = binary_closing(mask, morphology.disk(2))
        anat[anat==i] = 0
        anat[mask] = i

    return anat

# -------------------------------------------------------------
# VISUALISATION
# -------------------------------------------------------------
def visualize(img, anat, name):

    color = np.zeros((*img.shape,3),dtype=np.uint8)

    for k,(n,c) in REGIONS.items():
        color[anat==k]=c

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title("IRM")

    plt.subplot(1,2,2)
    plt.imshow(color)
    plt.title("Segmentation")

    plt.savefig(f"{OUTPUT_FOLDER}/{name}")
    plt.close()

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__=="__main__":

    images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".png")]

    for f in images:
        print("Processing",f)

        path = os.path.join(INPUT_FOLDER,f)

        img = preprocess(path)
        seg = segment(img)
        anat = assign_anatomy(seg,img)

        visualize(img,anat,f)

    print("\nDONE")