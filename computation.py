from umap import UMAP
from pystackreg import StackReg
from pystackreg.util import simple_slice
import hdbscan
from pathlib import Path
from typing import Optional, Union, Literal
import numpy as np

from typing import Dict, Literal
from scipy.optimize import linear_sum_assignment

from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr
from scipy.spatial import procrustes

_SQRT2 = np.sqrt(2)


def register_imagestack(image_stack: np.ndarray,
                        save_dir: Union[str, Path],
                        transformation: int = StackReg.RIGID_BODY,
                        reg_mode: Literal['first', 'mean', 'reference'] = 'first',
                        reference: Optional[np.ndarray] = None):
    reg_fn = save_dir.joinpath(f"registered_{reg_mode}.npy")
    tmats_fn = save_dir.joinpath(f"tmats_{reg_mode}.npy")

    if reg_fn.exists():
        x_reg = np.load(reg_fn)
        tmats = np.load(tmats_fn)
    else:
        # register predictions
        sr = StackReg(transformation)
        if reg_mode == 'reference':
            tmats = np.repeat(np.identity(3).reshape((1, 3, 3)), image_stack.shape[0], axis=0).astype(np.double)
            iterable = range(0, image_stack.shape[0])
            for i in iterable:
                slc = [slice(None)] * len(image_stack.shape)
                slc[0] = i
                tmats[i, :, :] = sr.register(reference, simple_slice(image_stack, i, 0))
        else:
            tmats = sr.register_stack(image_stack, reference=reg_mode)
        x_reg = sr.transform_stack(image_stack, tmats=tmats)

        np.save(reg_fn, x_reg.astype(np.float32))
        np.save(tmats_fn, tmats)

    return x_reg


def clusterable_embedding_umap(imgs: np.ndarray,
                               embedding_config
                               ) -> np.ndarray:
    img_flat = imgs.reshape(imgs.shape[0], -1)
    return UMAP(**embedding_config).fit_transform(img_flat)


def clustering_hdbscan(embedding: np.ndarray,
                       clustering_config
                       ) -> np.ndarray:
    return hdbscan.HDBSCAN(**clustering_config).fit_predict(embedding)


def uniquify_labels(labels) -> np.ndarray:
    lbl_unique, lbl_index, _ = np.unique(labels, return_counts=True, return_index=True)
    return lbl_unique[np.argsort(lbl_index)]


def correlation_similarity(map_a, map_b):
    return pearsonr(map_a.flatten(), map_b.flatten())[0]


def procrustes_similarity(skeleton_a, skeleton_b):
    _, _, disparity = procrustes(skeleton_a, skeleton_b)
    return -disparity


def hausdorff_similarity(contour_a, contour_b):
    dist_ab = directed_hausdorff(contour_a, contour_b)[0]
    dist_ba = directed_hausdorff(contour_b, contour_a)[0]
    return -(dist_ab + dist_ba) / 2


def js_divergence_similarity(map_a, map_b):
    js_divergence = jensenshannon(map_a.flatten(), map_b.flatten())
    return 1 - js_divergence


def composite_similarity(map_a, map_b, contour_a, contour_b):
    correlation_score = correlation_similarity(map_a, map_b)
    hausdorff_score = hausdorff_similarity(contour_a, contour_b)
    js_score = js_divergence_similarity(map_a, map_b)

    composite_score = (0.5 * correlation_score) + (0.3 * hausdorff_score) + (0.2 * js_score)
    return composite_score


def build_cluster_dict(reg_stack: np.ndarray,
                       cluster_labels: np.ndarray,
                       cluster_labels_unique: np.ndarray,
                       intra_cluster_registration: bool = True,
                       transformation: int = StackReg.RIGID_BODY,
                       reg_mode: Literal['first', 'mean'] = 'mean'):
    """
    """
    cluster_dict = {}

    for clabel in cluster_labels_unique:
        subset: np.ndarray = reg_stack[cluster_labels == clabel]

        if intra_cluster_registration:
            sr = StackReg(transformation)
            tmats = sr.register_stack(subset, reference=reg_mode)
            subset_reg = sr.transform_stack(subset, tmats=tmats)
            cluster_dict[clabel] = subset_reg
        else:
            cluster_dict[clabel] = subset

    return cluster_dict


def build_centroid_dict(cluster_dict: Dict[int, np.ndarray],
                        op_centroid: Literal['mean'] = 'mean'):
    centroid_dict = {}

    clabel: int
    subset: np.ndarray
    for clabel, subset in cluster_dict.items():
        if op_centroid == 'mean':
            centroid = np.mean(subset, axis=0)
        elif op_centroid == 'median':
            centroid = np.median(subset, axis=0)
        elif op_centroid == 'first':
            centroid = subset[0]
        else:
            return NotImplementedError("Centroid Operation not supported")
        centroid_dict[clabel] = centroid

    return centroid_dict


def hungarian_matching(cluster_centroids_A: Dict[str, np.ndarray],
                       cluster_centroids_B: Dict[str, np.ndarray],
                       similarity_measure: Literal['js', 'procrustes', 'hausdorff', 'pearsonr', 'composite'] = 'js',
                       cross_registration: bool = True):
    def compute_similarity(ca, cb):
        if cross_registration:
            sr = StackReg(StackReg.SCALED_ROTATION)
            tmat = sr.register(ca, cb)
            cb_mapped = sr.transform(cb, tmat)
            return procrustes_similarity(ca, cb_mapped)
        else:
            return procrustes_similarity(ca, cb)

    num_clusters_a = len(cluster_centroids_A)
    num_clusters_b = len(cluster_centroids_B)
    cost_matrix = np.zeros((num_clusters_a, num_clusters_b))

    for i, centroid_a in enumerate(cluster_centroids_A.values()):
        for j, centroid_b in enumerate(cluster_centroids_B.values()):
            cost_matrix[i, j] = compute_similarity(centroid_a, centroid_b)

    cost_matrix = -cost_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = [(list(cluster_centroids_A.keys())[i], list(cluster_centroids_B.keys())[j]) for i, j in
                     zip(row_ind, col_ind)]
    unmatched_a = set(cluster_centroids_A.keys()) - set([x[0] for x in matched_pairs])
    unmatched_b = set(cluster_centroids_B.keys()) - set([x[1] for x in matched_pairs])
    matched_cost = []
    for pair in matched_pairs:
        matched_cost.append(cost_matrix[pair[0], pair[1]])

    return matched_pairs, matched_cost, unmatched_a, unmatched_b
