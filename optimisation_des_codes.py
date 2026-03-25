"""
============================================================
  combined_saliency.py
  Combinaison et optimisation de toutes les méthodes de
  saliency : Gradient, GradCAM, IntegratedGradients,
  GuidedIG, Occlusion, XRAI.
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


def normalize_map(smap, percentile=99):
    smap = np.abs(smap)
    if smap.ndim == 3:
        smap = smap.max(axis=-1)
    vmax = np.percentile(smap, percentile)
    if vmax == 0:
        return smap
    return np.clip(smap / vmax, 0, 1)


def smooth_map(smap, sigma=1.5):
    return gaussian_filter(smap, sigma=sigma)


def overlay_heatmap(image, heatmap, alpha=0.55, colormap='jet'):
    cmap  = plt.get_cmap(colormap)
    hmap3 = cmap(heatmap)[..., :3]
    img01 = np.clip(image, 0, 1)
    return np.clip((1 - alpha) * img01 + alpha * hmap3, 0, 1)


def resize_map(smap, target_shape):
    from scipy.ndimage import zoom
    zh = target_shape[0] / smap.shape[0]
    zw = target_shape[1] / smap.shape[1]
    return zoom(smap, (zh, zw), order=1)






class _MethodWrapper:
    def __init__(self, saliency_instance):
        self.method = saliency_instance

    def compute(self, x_value, call_model_function,
                call_model_args=None, **kwargs):
        return self.method.GetMask(
            x_value, call_model_function,
            call_model_args=call_model_args, **kwargs)


class GradientWrapper(_MethodWrapper):
    pass


class SmoothGradWrapper:
    def __init__(self, gradient_saliency_instance):
        self.method = gradient_saliency_instance

    def compute(self, x_value, call_model_function,
                call_model_args=None,
                noise_level=0.15, num_samples=30, **kwargs):
        signal_range = x_value.max() - x_value.min()
        noise_std    = noise_level * signal_range
        accumulated  = np.zeros_like(x_value, dtype=np.float64)
        for _ in range(num_samples):
            noise   = np.random.normal(0, noise_std, x_value.shape)
            x_noisy = x_value + noise
            grad    = self.method.GetMask(
                x_noisy, call_model_function,
                call_model_args=call_model_args, **kwargs)
            accumulated += grad
        return accumulated / num_samples


class GradCamWrapper(_MethodWrapper):
    def compute(self, x_value, call_model_function,
                call_model_args=None, **kwargs):
        return self.method.GetMask(
            x_value, call_model_function,
            call_model_args=call_model_args,
            should_resize=True, three_dims=False, **kwargs)


class IntegratedGradientsWrapper(_MethodWrapper):
    def compute(self, x_value, call_model_function,
                call_model_args=None,
                x_baseline=None, x_steps=50, batch_size=4, **kwargs):
        return self.method.GetMask(
            x_value, call_model_function,
            call_model_args=call_model_args,
            x_baseline=x_baseline,
            x_steps=x_steps,
            batch_size=batch_size, **kwargs)


class GuidedIGWrapper(_MethodWrapper):
    def compute(self, x_value, call_model_function,
                call_model_args=None,
                x_baseline=None, x_steps=200,
                fraction=0.25, max_dist=0.02, **kwargs):
        return self.method.GetMask(
            x_value, call_model_function,
            call_model_args=call_model_args,
            x_baseline=x_baseline,
            x_steps=x_steps,
            fraction=fraction,
            max_dist=max_dist, **kwargs)


class OcclusionWrapper(_MethodWrapper):
    def compute(self, x_value, call_model_function,
                call_model_args=None,
                size=15, value=0, **kwargs):
        return self.method.GetMask(
            x_value, call_model_function,
            call_model_args=call_model_args,
            size=size, value=value, **kwargs)


class XRAIWrapper(_MethodWrapper):
    def compute(self, x_value, call_model_function,
                call_model_args=None, **kwargs):
        return self.method.GetMask(
            x_value, call_model_function,
            call_model_args=call_model_args, **kwargs)


# stratégie de fusion 

def fusion_mean(maps):
    return np.mean(maps, axis=0)


def fusion_max(maps):
    return np.max(maps, axis=0)


def fusion_weighted(maps, weights):
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()
    result = np.zeros_like(maps[0], dtype=np.float64)
    for w, m in zip(weights, maps):
        result += w * m
    return result


def fusion_rank(maps):
    h, w     = maps[0].shape
    rank_sum = np.zeros((h, w), dtype=np.float64)
    k        = 60
    for m in maps:
        flat  = m.flatten()
        order = np.argsort(-flat)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(flat))
        rank_sum += 1.0 / (k + ranks.reshape(h, w))
    rank_sum -= rank_sum.min()
    if rank_sum.max() > 0:
        rank_sum /= rank_sum.max()
    return rank_sum


#classe principale

DEFAULT_WEIGHTS = {
    'gradient'   : 0.05,
    'smoothgrad' : 0.15,
    'gradcam'    : 0.15,
    'ig'         : 0.25,
    'guided_ig'  : 0.25,
    'occlusion'  : 0.10,
    'xrai'       : 0.05,
}


class SaliencyEnsemble:

    def __init__(self,
                 gradient_instance=None,
                 gradcam_instance=None,
                 ig_instance=None,
                 guided_ig_instance=None,
                 occlusion_instance=None,
                 xrai_instance=None,
                 weights=None):

        self.wrappers = {}

        if gradient_instance is not None:
            self.wrappers['gradient']   = GradientWrapper(gradient_instance)
            self.wrappers['smoothgrad'] = SmoothGradWrapper(gradient_instance)
        if gradcam_instance is not None:
            self.wrappers['gradcam']    = GradCamWrapper(gradcam_instance)
        if ig_instance is not None:
            self.wrappers['ig']         = IntegratedGradientsWrapper(ig_instance)
        if guided_ig_instance is not None:
            self.wrappers['guided_ig']  = GuidedIGWrapper(guided_ig_instance)
        if occlusion_instance is not None:
            self.wrappers['occlusion']  = OcclusionWrapper(occlusion_instance)
        if xrai_instance is not None:
            self.wrappers['xrai']       = XRAIWrapper(xrai_instance)

        self.weights = weights or DEFAULT_WEIGHTS

    def compute(self, x_value, call_model_function,
                call_model_args=None, methods='all',
                fusion='weighted', smooth_sigma=1.0,
                method_kwargs=None):

        if methods == 'all':
            selected = list(self.wrappers.keys())
        else:
            selected = [m for m in methods if m in self.wrappers]

        if not selected:
            raise ValueError(f"Aucune méthode valide. Disponibles : {list(self.wrappers)}")

        method_kwargs = method_kwargs or {}
        individual    = {}
        h, w          = x_value.shape[:2]

        print(f"[SaliencyEnsemble] Calcul de {len(selected)} méthodes : {selected}")

        for name in selected:
            print(f"  → {name} ...", end=' ', flush=True)
            try:
                kwargs = method_kwargs.get(name, {})
                raw    = self.wrappers[name].compute(
                    x_value, call_model_function,
                    call_model_args=call_model_args, **kwargs)
                smap = normalize_map(raw)
                if smap.shape != (h, w):
                    smap = resize_map(smap, (h, w))
                if smooth_sigma > 0:
                    smap = smooth_map(smap, sigma=smooth_sigma)
                individual[name] = smap
                print("OK")
            except Exception as e:
                print(f"ERREUR ({e})")

        maps_list = list(individual.values())

        if len(maps_list) == 1:
            fused = maps_list[0]
        elif fusion == 'mean':
            fused = fusion_mean(maps_list)
        elif fusion == 'max':
            fused = fusion_max(maps_list)
        elif fusion == 'rank':
            fused = fusion_rank(maps_list)
        else:
            w_list = [self.weights.get(n, 1.0) for n in individual]
            fused  = fusion_weighted(maps_list, w_list)

        if fused.max() > 0:
            fused = fused / fused.max()

        img01   = np.clip(x_value, 0, 1) if x_value.max() <= 1 \
                  else x_value / 255.0
        overlay = overlay_heatmap(img01, fused)

        print(f"[SaliencyEnsemble] Fusion '{fusion}' terminée.")

        return {'individual': individual, 'fused': fused, 'overlay': overlay}

    def visualize(self, x_value, results,
                  save_path=None, figsize=None,
                  cmap_individual='hot', cmap_fused='RdYlGn'):

        individual = results['individual']
        fused      = results['fused']
        overlay    = results['overlay']
        n_methods  = len(individual)
        n_cols     = n_methods + 3
        figsize    = figsize or (4 * n_cols, 4)

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.04)

        img01 = np.clip(x_value, 0, 1) if x_value.max() <= 1 \
                else x_value / 255.0
        col = 0

        ax = fig.add_subplot(gs[col]); col += 1
        ax.imshow(img01, cmap='gray' if img01.ndim == 2 else None)
        ax.set_title('Image\noriginale', fontsize=9, fontweight='bold')
        ax.axis('off')

        for name, smap in individual.items():
            ax = fig.add_subplot(gs[col]); col += 1
            im = ax.imshow(smap, cmap=cmap_individual, vmin=0, vmax=1)
            ax.set_title(self._pretty_name(name), fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[col]); col += 1
        im = ax.imshow(fused, cmap=cmap_fused, vmin=0, vmax=1)
        ax.set_title('Fusion\n(ensemble)', fontsize=9, fontweight='bold', color='#1a5276')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[col])
        ax.imshow(overlay)
        ax.set_title('Overlay\nsur image', fontsize=9, fontweight='bold', color='#7b241c')
        ax.axis('off')

        fig.suptitle('Ensemble de méthodes de saillance — comparaison',
                     fontsize=12, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SaliencyEnsemble] Figure sauvegardée → {save_path}")
        else:
            plt.tight_layout()
            plt.show()

        return fig

    def coherence_report(self, results):
        from scipy.stats import spearmanr
        individual = results['individual']
        names      = list(individual.keys())
        maps_flat  = [individual[n].flatten() for n in names]
        n          = len(names)
        matrix     = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                r, _         = spearmanr(maps_flat[i], maps_flat[j])
                matrix[i, j] = r
                matrix[j, i] = r
        print("\n[Cohérence inter-méthodes — corrélation de Spearman]")
        print(f"{'':12s}" + "".join(f"{n:12s}" for n in names))
        for i, name in enumerate(names):
            print(f"{name:12s}" + "".join(f"{matrix[i,j]:12.3f}" for j in range(n)))
        return {'matrix': matrix, 'names': names}

    @staticmethod
    def _pretty_name(name):
        labels = {
            'gradient'  : 'Gradient\nVanilla',
            'smoothgrad': 'SmoothGrad',
            'gradcam'   : 'GradCAM',
            'ig'        : 'Integrated\nGradients',
            'guided_ig' : 'Guided IG',
            'occlusion' : 'Occlusion',
            'xrai'      : 'XRAI',
        }
        return labels.get(name, name)


#boucles sur images

if __name__ == '__main__':

    import tensorflow as tf
    from PIL import Image
    from saliency.core import (GradientSaliency, GradCam,
                                IntegratedGradients, GuidedIG,
                                Occlusion, XRAI)
    from saliency.core.base import (INPUT_OUTPUT_GRADIENTS,
                                    CONVOLUTION_LAYER_VALUES,
                                    CONVOLUTION_OUTPUT_GRADIENTS,
                                    OUTPUT_LAYER_VALUES)


    #chemin

    input_folder  = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/prediction"
    output_folder = "/home/myriam/Documents/stage_M2/Stage_2026/code/MASKCONTRAST/xai_consensus_results"
    os.makedirs(output_folder, exist_ok=True)

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    # modele
    IMG_SIZE   = (224, 224)
    CONV_LAYER = 'conv5_block3_out'   # dernière couche conv ResNet50
    CLASS_IDX  = 0

    print("[INFO] Chargement du modèle...")
    model = tf.keras.applications.ResNet50(weights='imagenet')

    model_with_conv = tf.keras.Model(
        inputs  = model.input,
        outputs = [model.get_layer(CONV_LAYER).output, model.output]
    )

    #  FONCTION MODÈLE (API Google Saliency)
    @tf.function
    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class  = (call_model_args or {}).get('class_idx', CLASS_IDX)
        images_tensor = tf.cast(images, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(images_tensor)
            conv_out, logits = model_with_conv(images_tensor)
            target_logit     = logits[:, target_class]

        grads      = tape.gradient(target_logit, images_tensor)
        conv_grads = tape.gradient(target_logit, conv_out)
        del tape

        result = {}
        if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
            result[INPUT_OUTPUT_GRADIENTS] = grads.numpy()
        if expected_keys and CONVOLUTION_LAYER_VALUES in expected_keys:
            result[CONVOLUTION_LAYER_VALUES]     = conv_out.numpy()
            result[CONVOLUTION_OUTPUT_GRADIENTS] = conv_grads.numpy()
        if expected_keys and OUTPUT_LAYER_VALUES in expected_keys:
            result[OUTPUT_LAYER_VALUES] = logits.numpy()
        return result

    #  ENSEMBLE
    ensemble = SaliencyEnsemble(
        gradient_instance  = GradientSaliency(),
        gradcam_instance   = GradCam(),
        ig_instance        = IntegratedGradients(),
        guided_ig_instance = GuidedIG(),
        occlusion_instance = Occlusion(),
        xrai_instance      = XRAI(),
    )

    call_model_args = {'class_idx': CLASS_IDX}

    #
    #boucle sur image
    image_paths = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])

    if not image_paths:
        print(f"[ERREUR] Aucune image trouvée dans : {input_folder}")
    else:
        print(f"\n[INFO] {len(image_paths)} image(s) trouvée(s) dans :")
        print(f"       {input_folder}")
        print(f"[INFO] Résultats sauvegardés dans :")
        print(f"       {output_folder}\n")

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'='*60}")
        print(f"[IMAGE] {img_name}")
        print(f"{'='*60}")

        try:
            pil_img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            x_value = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"  [ERREUR chargement] {e} — image ignorée.")
            continue

        try:
            results = ensemble.compute(
                x_value,
                call_model_function,
                call_model_args = call_model_args,
                methods         = 'all',
                fusion          = 'weighted',
                smooth_sigma    = 1.0,
                method_kwargs   = {
                    'ig'        : {'x_steps': 50,  'batch_size': 4},
                    'guided_ig' : {'x_steps': 200, 'fraction': 0.25, 'max_dist': 0.02},
                    'occlusion' : {'size': 15, 'value': 0},
                    'smoothgrad': {'noise_level': 0.15, 'num_samples': 30},
                }
            )
        except Exception as e:
            print(f"  [ERREUR compute] {e} — image ignorée.")
            continue

        save_path = os.path.join(output_folder, f"{img_name}_saliency_ensemble.png")
        ensemble.visualize(x_value, results, save_path=save_path)

        npy_dir = os.path.join(output_folder, img_name)
        os.makedirs(npy_dir, exist_ok=True)

        for method_name, smap in results['individual'].items():
            np.save(os.path.join(npy_dir, f"{method_name}.npy"), smap)

        np.save(os.path.join(npy_dir, "fused.npy"),   results['fused'])
        np.save(os.path.join(npy_dir, "overlay.npy"), results['overlay'])

        ensemble.coherence_report(results)

        print(f"  [OK] Résultats sauvegardés → {save_path}")

    print(f"\n[TERMINÉ] Toutes les images ont été traitées.")