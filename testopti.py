# CARTES DE SAILLANCE - TOUTES MÉTHODES
# Gradient | Integrated Gradients | GradCAM | Guided IG | SmoothGrad
# Occlusion | Blur IG | XRAI
# 
# pip install torch torchvision numpy matplotlib opencv-python Pillow scipy scikit-image

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# chemin
input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/predicfGRAMCAM"
os.makedirs(output_folder, exist_ok=True)

# configuration
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]
MODEL_NAME   = "resnet50" 
COLORMAP     = "inferno"       
ALPHA        = 1           # transparence overlay (0=image pure, 1=heatmap pure)
EXTENSIONS   = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

print(f"Device : {DEVICE}")
print(f"Images : {input_folder}")
print(f"Sorties: {output_folder}")

# pre et post processing
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def load_image(path):
    return Image.open(path).convert("RGB")

def to_tensor(img):
    return preprocess(img).unsqueeze(0).to(DEVICE)

def to_numpy(t):
    t = t.squeeze(0).cpu().detach()
    mean = torch.tensor(MEAN).view(3,1,1)
    std  = torch.tensor(STD).view(3,1,1)
    t = (t * std + mean).permute(1,2,0).numpy()
    return t.clip(0, 1)

def norm(sal):
    sal = np.abs(sal)
    mn, mx = sal.min(), sal.max()
    if mx - mn < 1e-8:
        return np.zeros_like(sal)
    return (sal - mn) / (mx - mn)

def resize2d(sal, h, w):
    return cv2.resize(sal.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

 
# modèle
def load_model():
    model = getattr(models, MODEL_NAME)(pretrained=True)
    model.eval().to(DEVICE)
    return model

def predict(model, t):
    with torch.no_grad():
        logits = model(t)
    prob = F.softmax(logits, dim=1)
    idx  = prob.argmax(dim=1).item()
    return idx, prob[0, idx].item()

# gradient
def method_gradient(model, t, cls):
    x = t.clone().requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits[0, cls].backward()
    sal = np.max(np.abs(x.grad.data.cpu().numpy()[0]), axis=0)
    return norm(sal)

# integrated gradient
def method_ig(model, t, cls, steps=50):
    baseline = torch.zeros_like(t).to(DEVICE)
    grads = []
    for alpha in torch.linspace(0, 1, steps).to(DEVICE):
        interp = (baseline + alpha * (t - baseline)).requires_grad_(True)
        model.zero_grad()
        model(interp)[0, cls].backward()
        grads.append(interp.grad.data.cpu().numpy()[0])
    avg = np.mean(grads, axis=0)
    delta = (t - baseline).cpu().numpy()[0]
    return norm(np.sum(np.abs(delta * avg), axis=0))


# smoothgrad
def method_smoothgrad(model, t, cls, n=40, noise=0.15):
    stdev = noise * (t.max() - t.min()).item()
    grads = []
    for _ in range(n):
        noisy = (t + torch.randn_like(t) * stdev).requires_grad_(True)
        model.zero_grad()
        model(noisy)[0, cls].backward()
        grads.append(noisy.grad.data.cpu().numpy()[0])
    sal = np.max(np.abs(np.mean(grads, axis=0)), axis=0)
    return norm(sal)



# gradcam
def method_gradcam(model, t, cls):
    activations, gradients = {}, {}

    if hasattr(model, "layer4"):
        target = model.layer4[-1]
    elif hasattr(model, "features"):
        target = model.features[-1]
    else:
        raise ValueError("Ajoute target_layer manuellement pour ce modèle.")

    fwd = target.register_forward_hook(lambda m,i,o: activations.update({"v": o.detach()}))
    bwd = target.register_full_backward_hook(lambda m,i,o: gradients.update({"v": o[0].detach()}))

    x = t.clone().requires_grad_(True)
    model.zero_grad()
    model(x)[0, cls].backward()

    fwd.remove(); bwd.remove()

    acts = activations["v"][0].cpu().numpy()
    gds  = gradients["v"][0].cpu().numpy()
    weights = gds.mean(axis=(1,2))
    cam = np.maximum(np.sum(weights[:,None,None] * acts, axis=0), 0)
    H, W = t.shape[2], t.shape[3]
    return norm(resize2d(cam, H, W))



# guided integraded gradient

def method_guided_ig(model, t, cls, steps=50):
    import copy
    gm = copy.deepcopy(model).to(DEVICE)

    handles = []
    def guided_hook(m, gi, go):
        return tuple(g.clamp(min=0) if g is not None else g for g in gi)
    for mod in gm.modules():
        if isinstance(mod, nn.ReLU):
            handles.append(mod.register_backward_hook(guided_hook))

    baseline = torch.zeros_like(t).to(DEVICE)
    grads = []
    for alpha in torch.linspace(0, 1, steps).to(DEVICE):
        interp = (baseline + alpha * (t - baseline)).requires_grad_(True)
        gm.zero_grad()
        gm(interp)[0, cls].backward()
        if interp.grad is not None:
            grads.append(interp.grad.data.cpu().numpy()[0])

    for h in handles: h.remove()

    if not grads:
        return np.zeros((t.shape[2], t.shape[3]))
    avg = np.mean(grads, axis=0)
    delta = (t - baseline).cpu().numpy()[0]
    return norm(np.sum(np.abs(delta * avg), axis=0))

 
# occlusion

def method_occlusion(model, t, cls, patch=28, stride=14):
    H, W = t.shape[2], t.shape[3]
    sal   = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    with torch.no_grad():
        base = F.softmax(model(t), dim=1)[0, cls].item()
    for r in range(0, H - patch + 1, stride):
        for c in range(0, W - patch + 1, stride):
            occ = t.clone()
            occ[0, :, r:r+patch, c:c+patch] = 0.5
            with torch.no_grad():
                sc = F.softmax(model(occ), dim=1)[0, cls].item()
            sal[r:r+patch, c:c+patch]   += base - sc
            count[r:r+patch, c:c+patch] += 1
    count = np.where(count == 0, 1, count)
    return norm(sal / count)



# blur ig
def method_blur_ig(model, t, cls, steps=40, max_sigma=10.0):
    img = t.cpu().numpy()[0]
    grads = []
    for sigma in np.linspace(max_sigma, 0, steps):
        blurred = np.stack([gaussian_filter(img[c], sigma=sigma) for c in range(3)], axis=0)
        bt = torch.tensor(blurred, dtype=torch.float32).unsqueeze(0).to(DEVICE).requires_grad_(True)
        model.zero_grad()
        model(bt)[0, cls].backward()
        grads.append(bt.grad.data.cpu().numpy()[0])
    avg = np.mean(grads, axis=0)
    baseline = np.stack([gaussian_filter(img[c], sigma=max_sigma) for c in range(3)], axis=0)
    delta = img - baseline
    return norm(np.sum(np.abs(delta * avg), axis=0))



# XRAI 
def method_xrai(model, t, cls, steps=50, n_segments=150):
    try:
        from skimage.segmentation import slic
    except ImportError:
        print("  ⚠ skimage absent → XRAI utilise IG. pip install scikit-image")
        return method_ig(model, t, cls, steps)

    img_np = to_numpy(t)
    H, W   = img_np.shape[:2]

    # IG baseline noire
    ig_black = method_ig(model, t, cls, steps)

    # IG baseline blanche
    baseline_w = torch.ones_like(t).to(DEVICE)
    grads = []
    for alpha in torch.linspace(0, 1, steps).to(DEVICE):
        interp = (baseline_w + alpha * (t - baseline_w)).requires_grad_(True)
        model.zero_grad()
        model(interp)[0, cls].backward()
        grads.append(interp.grad.data.cpu().numpy()[0])
    avg_w   = np.mean(grads, axis=0)
    delta_w = (t - baseline_w).cpu().numpy()[0]
    ig_white = norm(np.sum(np.abs(delta_w * avg_w), axis=0))

    ig_fused = (ig_black + ig_white) / 2.0

    # Superpixels =  score par région
    segments = slic(img_np, n_segments=n_segments, compactness=10, sigma=1,
                    start_label=0, channel_axis=-1)
    xrai_map = np.zeros((H, W), dtype=np.float32)
    for sid in np.unique(segments):
        mask = segments == sid
        xrai_map[mask] = ig_fused[mask].mean()

    return norm(xrai_map)




# segmentation

def segment(sal, morph=5):
    sal_u8 = (sal * 255).astype(np.uint8)
    _, mask = cv2.threshold(sal_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    return mask




#visualisation
def heatmap_rgb(sal):
    return cm.get_cmap(COLORMAP)(sal)[:, :, :3].astype(np.float32)

def overlay(img, sal):
    return ((1 - ALPHA) * img + ALPHA * heatmap_rgb(sal)).clip(0, 1)

def with_contour(img, mask):
    out = (img * 255).astype(np.uint8).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (50, 230, 80), 2)
    return out / 255.0

def save_grid(img_np, results, fname):
    """
    Grille : 1 ligne par méthode × 4 colonnes (heatmap / overlay / masque / contour)
    """
    methods = list(results.keys())
    n_rows  = len(methods) + 1   # +1 ligne de titres
    fig, axes = plt.subplots(n_rows, 4,
                              figsize=(16, 3.2 * n_rows),
                              facecolor="#0d0d0d")

    # Titres colonnes
    for j, title in enumerate(["Heatmap", "Overlay", "Masque", "Contour"]):
        axes[0, j].set_facecolor("#0d0d0d")
        axes[0, j].text(0.5, 0.5, title, transform=axes[0, j].transAxes,
                        ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white", fontfamily="monospace")
        axes[0, j].axis("off")

    for i, method in enumerate(methods):
        sal  = results[method]
        mask = segment(sal)
        row  = i + 1
        imgs = [heatmap_rgb(sal), overlay(img_np, sal),
                np.stack([mask/255]*3, axis=-1), with_contour(img_np, mask)]
        for j, im in enumerate(imgs):
            ax = axes[row, j]
            ax.imshow(im, aspect="auto")
            ax.set_facecolor("#0d0d0d")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#2a2a2a")
            if j == 0:
                ax.set_ylabel(method, color="white", fontsize=8,
                               fontfamily="monospace", rotation=0,
                               ha="right", va="center", labelpad=72)

    plt.suptitle(os.path.basename(fname), color="white",
                 fontsize=11, fontfamily="monospace")
    plt.tight_layout(pad=0.3)
    plt.savefig(fname, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)





# pipeline par image
 
METHODS = {
    "Gradient":           method_gradient,
    "Integrated_Grads":   method_ig,
    "SmoothGrad":         method_smoothgrad,
    "GradCAM":            method_gradcam,
    "Guided_IG":          method_guided_ig,
    "Occlusion":          method_occlusion,
    "Blur_IG":            method_blur_ig,
    "XRAI":               method_xrai,
}

def process_image(model, img_path):
    img   = load_image(img_path)
    t     = to_tensor(img)
    img_np = to_numpy(t)
    cls, score = predict(model, t)
    print(f"  Classe prédite : {cls}  |  score : {score:.3f}")

    results = {}
    for name, fn in METHODS.items():
        print(f"    ⏳ {name} ...", end="\r")
        try:
            results[name] = fn(model, t, cls)
            print(f"    succès {name:<25}")
        except Exception as e:
            print(f"   non {name} : {e}")

    # Sauvegarde grille complète
    base  = os.path.splitext(os.path.basename(img_path))[0]
    out   = os.path.join(output_folder, f"{base}_saliency.png")
    save_grid(img_np, results, out)
    print(f"  Sauvegardé : {out}")

    # Sauvegarde individuelle de chaque overlay
    for name, sal in results.items():
        ind_out = os.path.join(output_folder, f"{base}_{name}.png")
        plt.imsave(ind_out, overlay(img_np, sal))


# lancement sur tout le dossier
if __name__ == "__main__":
    model = load_model()
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(EXTENSIONS)]

    if not images:
        print(f" Aucune image trouvée dans : {input_folder}")
    else:
        print(f"\n{len(images)} image trouvée\n")
        for i, fname in enumerate(sorted(images), 1):
            path = os.path.join(input_folder, fname)
            print(f"[{i}/{len(images)}] {fname}")
            process_image(model, path)

    print(f"\nTerminé. Résultats dans : {output_folder}")
