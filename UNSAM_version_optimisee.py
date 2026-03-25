
# UNSAM SPINE SEGMENTATION

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage import morphology
from skimage.filters import sobel
from scipy.ndimage import binary_closing
import warnings
warnings.filterwarnings('ignore')



INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/unsam_results2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------------------------------------
# PARAMS
N_REGIONS = 14

# 
# COULEURS (par cphérence anatomique

REGIONS = {
    0: ('Fond', [20,20,20]),
    1: ('Disque', [255,180,0]),
    2: ('Sac centre', [0,200,255]),
    3: ('Sac périphérie', [0,120,200]),
    4: ('Éminence', [150,80,0]),

    5: ('Psoas G', [255,80,150]),
    6: ('Psoas D', [255,80,150]),

    7: ('Multifidus G sup', [60,200,100]),
    8: ('Multifidus G prof', [30,120,60]),
    9: ('Multifidus D sup', [60,200,100]),
    10: ('Multifidus D prof', [30,120,60]),

    11: ('Erecteur G', [80,120,255]),
    12: ('Erecteur D', [80,120,255]),

    13: ('Graisse', [255,200,120]),
}



# PREPROCESS

def preprocess(path):
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32)
    arr = (arr - arr.min())/(arr.max()-arr.min()+1e-8)
    return arr



# FEATURES 
def extract_features(img):

    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]

    grad = sobel(img)

    center_y, center_x = H//2, W//2
    dist_center = np.sqrt((yy-center_y)**2 + (xx-center_x)**2)
    dist_center = dist_center / dist_center.max()

    return np.stack([
        img,
        grad,
        yy/H,
        xx/W,
        dist_center
    ], axis=-1)




# SEGMENTATION
def segment(img):

    feat = extract_features(img).reshape(-1,5)
    feat = StandardScaler().fit_transform(feat)

    km = KMeans(n_clusters=N_REGIONS, random_state=42, n_init=10)
    return km.fit_predict(feat).reshape(img.shape)





# CLUSTER SIMILAR REGIONS

def group_similar_regions(regions):

    if len(regions) < 4:
        for r in regions:
            r['cluster'] = 0
        return regions

    X = np.array([[r['mean'], r['std'], r['dist']] for r in regions])
    km = KMeans(n_clusters=min(4, len(regions)), random_state=0).fit(X)

    for i, r in enumerate(regions):
        r['cluster'] = km.labels_[i]

    return regions



# ASSIGN ANATOMY
def assign_anatomy(seg_map, img):

    H, W = img.shape

    regions = []

    for k in range(int(seg_map.max()) + 1):

        m = seg_map == k
        if m.sum() < 40:
            continue

        ys, xs = np.where(m)

        mean_y = ys.mean()
        mean_x = xs.mean()

        norm_y = mean_y / H
        norm_x = mean_x / W

        dist_center = np.sqrt((norm_y - 0.5)**2 + (norm_x - 0.5)**2)

        regions.append({
            'k': k,
            'mean': img[m].mean(),
            'std': img[m].std(),
            'y': norm_y,
            'x': norm_x,
            'dist': dist_center
        })

    # IMPORTANT
    regions = group_similar_regions(regions)

    remaining = regions.copy()

    def pop_best(score):
        best = min(remaining, key=score)
        remaining.remove(best)
        return best['k']

    # STRUCTURES
    sac_center = pop_best(lambda p: -p['mean']*6 + p['dist']*10)
    sac_periph = pop_best(lambda p: -p['mean']*4 + p['dist']*6)

    disc = pop_best(lambda p: abs(p['mean']-0.45)*4 + (p['y']-0.6)**2*5)
    eminence = pop_best(lambda p: p['mean']*3 + (1-p['y'])*6)

    graisse = pop_best(lambda p: -p['mean']*2 + p['dist']*3)
    fond = pop_best(lambda p: p['mean'] + p['dist'])

    # MUSCLES
    left = [p for p in remaining if p['x'] < 0.5]
    right = [p for p in remaining if p['x'] >= 0.5]

    def sort_by_cluster(regs):
        clusters = {}
        for r in regs:
            clusters.setdefault(r['cluster'], []).append(r)

        ordered = sorted(clusters.values(),
                         key=lambda c: np.mean([r['dist'] for r in c]))

        return [item for sub in ordered for item in sub]

    left = sort_by_cluster(left)
    right = sort_by_cluster(right)

    def assign_side(regs, side):

        res = {}

        if side == 'L':
            psoas, mf_prof, mf_sup, erect = 5, 8, 7, 11
        else:
            psoas, mf_prof, mf_sup, erect = 6, 10, 9, 12

        if len(regs) > 0:
            res[regs[0]['k']] = psoas
        if len(regs) > 1:
            res[regs[1]['k']] = mf_prof
        if len(regs) > 2:
            res[regs[2]['k']] = mf_sup

        for r in regs[3:]:
            res[r['k']] = erect

        return res

    mapping = {
        fond: 0,
        disc: 1,
        sac_center: 2,
        sac_periph: 3,
        eminence: 4,
        graisse: 13
    }

    mapping.update(assign_side(left, 'L'))
    mapping.update(assign_side(right, 'R'))

    anat = np.zeros_like(seg_map)

    for k in range(int(seg_map.max()) + 1):
        anat[seg_map == k] = mapping.get(k, 0)

    # CLEANING
    for i in range(1, 14):
        mask = anat == i
        mask = morphology.remove_small_objects(mask, min_size=80)
        mask = binary_closing(mask, morphology.disk(3))
        anat[anat == i] = 0
        anat[mask] = i

    return anat




# VISUALIZATION

def visualize(img, anat, name):

    color = np.zeros((*img.shape,3),dtype=np.uint8)

    for k,(n,c) in REGIONS.items():
        color[anat==k] = c

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title("IRM")

    plt.subplot(1,2,2)
    plt.imshow(color)
    plt.title("Segmentation")

    plt.savefig(f"{OUTPUT_FOLDER}/{name}.png")
    plt.close()


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