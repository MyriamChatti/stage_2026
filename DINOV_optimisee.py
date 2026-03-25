# ============================================================
# DINOV2 ONLY + ATTENTION MAP (VERSION CORRIGÉE)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from scipy.ndimage import binary_closing, distance_transform_edt, zoom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from skimage import exposure, morphology
from skimage.feature import graycomatrix, graycoprops

import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
INPUT_FOLDER  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
OUTPUT_FOLDER = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/dinov2_attention_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMG_EXTENSIONS = {'.png','.jpg','.jpeg','.tif','.tiff'}

# -------------------------------------------------------------
# PARAMS
# -------------------------------------------------------------
IMG_SIZE = 224
PATCH_SIZE = 14
N_REGIONS = 10

# -------------------------------------------------------------
# REGIONS
# -------------------------------------------------------------
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
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
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

    sz = (IMG_SIZE//PATCH_SIZE)*PATCH_SIZE
    img = Image.fromarray((arr*255).astype(np.uint8)).resize((sz,sz))

    arr = np.array(img)/255
    arr = exposure.equalize_adapthist(arr)

    rgb = Image.merge('RGB',[Image.fromarray((arr*255).astype(np.uint8))]*3)

    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485]*3,[0.229]*3)
    ])

    return arr, t(rgb).unsqueeze(0)

# ============================================================
# FEATURES
# ============================================================
def extract_features(model, tensor, device):
    import torch

    with torch.no_grad():
        out = model.get_intermediate_layers(tensor.to(device), n=1, reshape=False)[0]

    f = out.squeeze(0).cpu().numpy()
    h = w = int(np.sqrt(f.shape[0]))

    return f.reshape(h,w,-1)

# ============================================================
# 🔥 ATTENTION MAP (CORRIGÉE)
# ============================================================
def extract_attention(model, tensor, device):
    import torch

    with torch.no_grad():
        try:
            attn = model.get_last_selfattention(tensor.to(device))
            attn = attn[0].mean(0)[0,1:]
        except:
            # fallback DINOv2
            attn = model.get_intermediate_layers(tensor.to(device), n=1)[0]
            attn = attn.mean(-1).flatten()

    h = w = int(np.sqrt(len(attn)))
    attn = attn.reshape(h,w).cpu().numpy()

    attn = (attn - attn.min())/(attn.max()-attn.min()+1e-8)

    return attn

# ============================================================
# SEGMENTATION
# ============================================================
def segment(feat_map, img):

    h,w,d = feat_map.shape
    H,W = img.shape

    X = feat_map.reshape(-1,d)
    X = normalize(X)

    X = PCA(32).fit_transform(X)

    yy,xx = np.mgrid[0:h,0:w]
    pos = np.stack([yy.flatten()/h, xx.flatten()/w],axis=1)

    X = np.hstack([X,pos])
    X = StandardScaler().fit_transform(X)

    labels = KMeans(n_clusters=N_REGIONS, n_init=10).fit_predict(X)

    seg = labels.reshape(h,w)
    seg = zoom(seg,(H/h,W/w),order=0)

    return seg

# ============================================================
# ANATOMY
# ============================================================
def assign(seg,img):

    H,W = img.shape
    cy,cx = H/2,W/2

    props = []

    for k in range(int(seg.max())+1):
        m = seg==k
        if m.sum()==0: continue
        y,x = np.where(m)
        props.append({
            'k':k,
            'mean':img[m].mean(),
            'cy':y.mean(),
            'cx':x.mean()
        })

    rem = props.copy()

    def pick(f):
        b=min(rem,key=f)
        rem.remove(b)
        return b['k']

    fond = pick(lambda p:p['mean'])
    sac  = pick(lambda p:-p['mean'])
    disc = pick(lambda p:abs(p['cy']-cy))
    emin = pick(lambda p:p['cy'])

    left  = [p for p in rem if p['cx']<cx]
    right = [p for p in rem if p['cx']>=cx]

    mapping = {
        fond:0, disc:1, sac:2, emin:3
    }

    for i,p in enumerate(left):
        mapping[p['k']] = 4+i

    for i,p in enumerate(right):
        mapping[p['k']] = 6+i

    anat = np.zeros_like(seg)

    for k in range(int(seg.max())+1):
        anat[seg==k] = mapping.get(k,0)

    return anat

# ============================================================
# VISUALISATION
# ============================================================
def visualize(img, seg, anat, attn, name):

    color = np.zeros((*img.shape,3),dtype=np.uint8)

    for k,(n,c) in REGIONS.items():
        color[anat==k]=c

    attn_full = zoom(attn,(img.shape[0]/attn.shape[0],img.shape[1]/attn.shape[1]))

    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.imshow(img,cmap='gray')
    plt.title("IRM")

    plt.subplot(1,3,2)
    plt.imshow(color)
    plt.title("Segmentation")

    plt.subplot(1,3,3)
    plt.imshow(img,cmap='gray')
    plt.imshow(attn_full,cmap='jet',alpha=0.5)
    plt.title("Attention DINOv2")

    plt.savefig(f"{OUTPUT_FOLDER}/{name}.png")
    plt.close()

# ============================================================
# MAIN
# ============================================================
if __name__=="__main__":

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_dino(device)

    images = [f for f in os.listdir(INPUT_FOLDER) if Path(f).suffix.lower() in IMG_EXTENSIONS]

    for f in images:

        print("Processing",f)

        img, tensor = preprocess(os.path.join(INPUT_FOLDER,f))

        feat = extract_features(model,tensor,device)
        attn = extract_attention(model,tensor,device)

        seg  = segment(feat,img)
        anat = assign(seg,img)

        visualize(img,seg,anat,attn,f)

    print("\nDONE")