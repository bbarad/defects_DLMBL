from affogato.segmentation import compute_mws_segmentation
import waterz
from scipy.ndimage import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes
import numpy as np


def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        return_seeds=False,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    # print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint16), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        fragments_in_xy=True,
        return_seeds=False,
        min_seed_distance=10,
        threshold=0.95,
        labels_mask=None):

    mean_affs = (affs[1] + affs[2])*0.5
    depth = mean_affs.shape[0]

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
    if return_seeds:
        seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

    id_offset = 0

    for z in range(depth):

        boundary_mask = mean_affs[z]>threshold*max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)
        if labels_mask is not None:

            boundary_mask *= labels_mask.astype(bool)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask,
            return_seeds=return_seeds,
            id_offset=id_offset,
            min_seed_distance=min_seed_distance)

        fragments[z] = ret[0]
        if return_seeds:
            seeds[z] = ret[2]

        id_offset = ret[1]

    ret = (fragments)
    if return_seeds:
        ret += (seeds,)

    return ret

def mutex_watershed(affinities,offsets,seperating_channel=2,strides=None):
		attractive_repuslive_weights = affinities.copy()
		attractive_repuslive_weights[:seperating_channel] *= -1
		attractive_repuslive_weights[:seperating_channel] += +1
		seg = compute_mws_segmentation(attractive_repuslive_weights, offsets, seperating_channel, strides=strides)
		return seg


""" # utility function to agglomerate fragments using underlying affinities as edge weights
# returns a segmentation from a final threshold

def get_segmentation(affinities, threshold=.5, labels_mask=None):
    threshold=threshold
    
    fragments = watershed_from_affinities(
            affinities,
            labels_mask=labels_mask)[0]

    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(generator)

    
    # remove small holes and relabel connected components
    final_seg, _ = label(
            remove_small_holes(
            segmentation.astype(bool),
            area_threshold=256))

    final_seg = final_seg.astype(np.uint64)

    return final_seg
 """
