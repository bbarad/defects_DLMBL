import numpy as np
from .rand_metrics import adapted_rand
from .vi_metrics import voi
from .border_mask import create_border_mask


def cremi_scores(seg, gt, border_threshold=None, return_all=True):
    """
    Compute the cremi scores (Average of adapted rand error, vi-split, vi-merge)

    Parameters
    ----------
    seg: np.ndarray - the candidate segmentation
    gt: np.ndarray - the groundtruth
    border_threshold: value by which the border is eroded (default: None = no erosion)

    Returns
    -------
    cremi-score: average of rand error, vi-split, vi-merge
    vi-split: variation of information, split score
    vi-merge: variation of information, merge score
    adapted rand: adapted rand error
    """
    assert seg.shape == gt.shape, "%s, %s" % (str(seg.shape, gt.shape))
    # compute border threshold if specified
    if border_threshold is not None:
        xy_resolution = 4.
        gt_ = create_border_mask(gt, border_threshold / xy_resolution, np.uint64(-1))
        # add 1 to map back to 0 as lowest label
        gt_ += 1
    else:
        gt_ = gt
    ## Try except because sometimes both have nothing in them.
    try:
        vi_s, vi_m = voi(seg, gt_)
        are = adapted_rand(seg, gt_)
        cs = (vi_s + vi_m + are) / 3
    except: 
        cs = np.nan
        vi_s = np.nan
        vi_m = np.nan
        are = np.nan
    if return_all:
        return {'cremi-score': cs, 'vi-split': vi_s, 'vi-merge': vi_m, 'adapted_rand': are}
    else:
        return cs


def DiceMetric(seg, gt):
    """
    Compute the Dice metric

    Parameters
    ----------
    seg: np.ndarray - the candidate segmentation
    gt: np.ndarray - the groundtruth

    Returns
    -------
    Dice: Dice metric
    """
    assert seg.shape == gt.shape, "%s, %s" % (str(seg.shape, gt.shape))
    seg = seg.astype(np.uint64)
    gt = gt.astype(np.uint64)
    error = 0.00001
    return 2 * np.sum(seg * gt) / (np.sum(seg) + np.sum(gt)+error)