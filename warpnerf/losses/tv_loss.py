from typing import Tuple, TypeVar
import torch

from fvdb import GridBatch

T = TypeVar("T", bound=Tuple[torch.Tensor, ...])

def tv_loss(
    grid: GridBatch,
    ijk: torch.Tensor,
    features: T,
    res: int,
) -> T:
    nhood = grid.neighbor_indexes(ijk, 1).jdata.view(-1, 3, 3, 3)
    n_up = nhood[:, 1, 0, 0]
    n_right = nhood[:, 0, 1, 0]
    n_front = nhood[:, 0, 0, 1]
    n_center = nhood[:, 0, 0, 0]

    mask = torch.logical_and(torch.logical_and(n_center != -1, n_up != -1), n_front != -1)
    fmask = mask.float()
    n_up_mask, n_right_mask, n_center_mask, n_front_mask = n_up[mask], n_right[mask], n_center[mask], n_front[mask]

    tv_reg_feats = []

    for feature in features:
        diff_up = (feature[n_up_mask] - feature[n_center_mask]) / (256.0 / res)
        diff_right = (feature[n_right_mask] - feature[n_center_mask]) / (256.0 / res)
        diff_front = (feature[n_front_mask] - feature[n_center_mask]) / (256.0 / res)

        tv_reg = (diff_up ** 2.0 + diff_right ** 2.0 + diff_front ** 2.0).sum(-1)
        tv_reg_feats.append(tv_reg.mean())

    return tuple(tv_reg_feats)


