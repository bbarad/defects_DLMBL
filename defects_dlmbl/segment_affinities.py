import waterz
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed


def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        fragments_in_xy=True,
        return_seeds=False,
        min_seed_distance=10,
        labels_mask=None):

    mean_affs = 0.5*(affs[1] + affs[2])
    depth = mean_affs.shape[0]

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
    if return_seeds:
        seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

    id_offset = 0

    for z in range(depth):

        boundary_mask = mean_affs[z]>0.5*max_affinity_value
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

    ret = (fragments, id_offset)
    if return_seeds:
        ret += (seeds,)

    return ret



# utility function to agglomerate fragments using underlying affinities as edge weights
# returns a segmentation from a final threshold

def get_segmentation(affinities, threshold, labels_mask=None):

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

    return segmentation