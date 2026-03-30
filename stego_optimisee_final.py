# ============================================================
# STEGO 
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    zoom
)

from skimage import exposure, morphology
from skimage.filters import sobel

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================

INPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/stego_FINAL"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMG_EXTENSIONS = {'.png','.jpg','.jpeg','.tif','.tiff'}

# ============================================================
# PARAMS
# ============================================================

IMG_SIZE = 224
PATCH_SIZE = 14
N_REGIONS = 10

# ============================================================
# COLORS
# ============================================================

REGIONS = {
    0: ('Fond', [30,30,30]),
    1: ('Disque', [255,200,0]),
    2: ('Sac', [0,180,255]),
    3: ('Éminence', [180,90,0]),
    4: ('Psoas G', [220,60,180]),
    5: ('Psoas D', [80,200,60]),
    6: ('Multifidus G', [50,160,80]),
    7: ('Multifidus D', [160,60,200]),
    8: ('Érecteur G', [40,80,200]),
    9: ('Érecteur D', [210,190,40]),
}

# ============================================================
# LOAD DINOv2
# ============================================================

def load_dino(device):
    import torch
    model = torch.hub.load(
        'facebookresearch/dinov2',
        'dinov2_vits14',
        pretrained=True
    )
    model.eval().to(device)
    return model

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(path):
    import torchvision.transforms as T

    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32)

    arr = (arr - arr.min())/(arr.max()-arr.min()+1e-8)

    size = (IMG_SIZE // PATCH_SIZE) * PATCH_SIZE

    img_r = Image.fromarray((arr*255).astype(np.uint8)).resize((size,size))
    arr = np.array(img_r).astype(np.float32)/255

    arr = gaussian_filter(arr,1)
    arr = exposure.equalize_adapthist(arr)

    rgb = Image.merge('RGB',[Image.fromarray((arr*255).astype(np.uint8))]*3)

    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485]*3,[0.229]*3)
    ])

    return arr, t(rgb).unsqueeze(0)

# ============================================================
# FEATURES DINO
# ============================================================

def extract_features(model, tensor, device):
    import torch

    with torch.no_grad():
        feats = model.forward_features(tensor.to(device))
        f = feats["x_norm_patchtokens"]

    f = f.squeeze(0).cpu().numpy()

    h = w = int(np.sqrt(f.shape[0]))

    return f.reshape(h,w,-1)

# ============================================================
# STEGO SIMPLE
# ============================================================

def stego_segment(feat_map, img):

    h,w,d = feat_map.shape
    H,W = img.shape

    X = feat_map.reshape(-1,d)
    X = normalize(X)

    X = PCA(min(64,d)).fit_transform(X)

    yy,xx = np.mgrid[0:h,0:w]

    pos = np.stack([
        yy.flatten()/h,
        xx.flatten()/w
    ],axis=1)

    grad = sobel(img)
    grad_small = zoom(grad,(h/H,w/W)).flatten()[:,None]

    center = np.exp(-((pos[:,0]-0.5)**2+(pos[:,1]-0.5)**2)*5)

    X = np.hstack([
        X,
        pos*0.3,
        grad_small*0.5,
        center[:,None]
    ])

    X = StandardScaler().fit_transform(X)

    labels = KMeans(n_clusters=N_REGIONS, n_init=20).fit_predict(X)

    seg = labels.reshape(h,w)
    seg = zoom(seg,(H/h,W/w),order=0)

    return seg

# ============================================================
# ASSIGN ANATOMY
# ============================================================

def assign(seg,img):

    H,W = img.shape

    props = []

    for k in range(int(seg.max())+1):
        m = seg==k
        if m.sum()<20: continue

        y,x = np.where(m)

        props.append({
            'k':k,
            'mean':img[m].mean(),
            'cy':y.mean()/H,
            'cx':x.mean()/W,
            'dist':np.sqrt((y.mean()/H-0.5)**2+(x.mean()/W-0.5)**2)
        })

    rem = props.copy()

    def pick(f):
        b=min(rem,key=f)
        rem.remove(b)
        return b['k']

    sac = pick(lambda p: -p['mean']*5 + p['dist']*8)
    disc = pick(lambda p: abs(p['mean']-0.4)+p['cy']*2)
    fond = pick(lambda p: p['mean'])
    emin = pick(lambda p: p['mean']+(1-p['cy']))

    left  = sorted([p for p in rem if p['cx']<0.5], key=lambda p:p['dist'])
    right = sorted([p for p in rem if p['cx']>=0.5], key=lambda p:p['dist'])

    mapping = {
        fond:0,
        disc:1,
        sac:2,
        emin:3
    }

    def assign_side(lst, base):
        if len(lst)>0: mapping[lst[0]['k']] = base
        if len(lst)>1: mapping[lst[1]['k']] = base+2
        for p in lst[2:]:
            mapping[p['k']] = base+4

    assign_side(left,4)
    assign_side(right,5)

    anat = np.zeros_like(seg)

    for k in range(int(seg.max())+1):
        anat[seg==k] = mapping.get(k,0)

    return anat

# ============================================================
# PROPAGATIOn
# ============================================================

def refine_segmentation(anat):

    refined = anat.copy()

    for i in range(1,N_REGIONS):
        m = refined==i

        m = morphology.remove_small_objects(m,50)
        m = binary_closing(m,morphology.disk(3))
        m = binary_fill_holes(m)

        refined[refined==i]=0
        refined[m]=i

    unknown = refined==0

    if unknown.any():
        _,idx = distance_transform_edt(unknown, return_indices=True)
        nearest = refined[idx[0],idx[1]]
        refined[unknown] = nearest[unknown]

    return refined





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
    plt.title("Segmentation finale")

    plt.savefig(f"{OUTPUT_FOLDER}/{name}")
    plt.close()





if __name__=="__main__":

    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:",device)

    model = load_dino(device)

    images = [f for f in os.listdir(INPUT_FOLDER)
              if Path(f).suffix.lower() in IMG_EXTENSIONS]

    for f in images:

        print("Processing",f)

        img, tensor = preprocess(os.path.join(INPUT_FOLDER,f))

        feat = extract_features(model,tensor,device)

        seg  = stego_segment(feat,img)

        anat = assign(seg,img)

       
        anat = refine_segmentation(anat)

        visualize(img,anat,f)

    print("\n DONE — segmentation terminée")