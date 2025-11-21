from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from .image import transform_preds, pt_transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = np.concatenate(
                [
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32),
                ],
                axis=1,
            ).tolist()
        ret.append(top_preds)
    return ret


def ctdet_post_process_post_recale(dets: torch.Tensor, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    c_t = torch.from_numpy(c[0]).to(dets.device)
    s_t = torch.from_numpy(np.array(s)).to(dets.device)
    w_h = torch.tensor((w, h), dtype=torch.float, device=dets.device)
    trans = None
    # Across batch items
    for i in range(dets.shape[0]):
        top_preds = {}
        # pt_transform_preds is across detections
        dets[i, :, :2], trans = pt_transform_preds(
            dets[i, :, 0:2], c_t, s_t, w_h, trans=trans
        )
        dets[i, :, 2:4], _ = pt_transform_preds(
            dets[i, :, 2:4], c_t, s_t, w_h, trans=trans
        )
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = (
                torch.cat(
                    [
                        dets[i, inds, :4].to(torch.float),
                        dets[i, inds, 4:5].to(torch.float),
                    ],
                    axis=1,
                ).cpu()
                # .detach()
                # .tolist()
            )
        ret.append(top_preds)
    return ret
