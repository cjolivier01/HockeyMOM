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


def ctdet_post_process_post_recale(
    dets: torch.Tensor, c, s, h, w, num_classes, dataloader
):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    c_t = torch.from_numpy(c[0]).to(dets.device)
    s_t = torch.from_numpy(np.array(s)).to(dets.device)
    # dets[:, :, 0:2] = dets[:, :, 0:2] + c_t
    # dets[:, :, 2:4] = dets[:, :, 2:4] + c_t
    # dets[:, :, 0:2] = dets[:, :, 0:2] + c_t
    #dets = dataloader.scale_letterbox_to_original_image_coordinates(dets)
    #print(dets[0, :, 0:4])
    w_h = torch.tensor((w, h), dtype=torch.float32, device=dets.device)
    # d_scaled = dets[:,:,0:2] * s_t
    for i in range(dets.shape[0]):
        top_preds = {}
        #dets[i, :, :2] = pt_transform_preds(dets[i, :, 0:2], c_t, s_t, w_h)
        #dets[i, :, 2:4] = pt_transform_preds(dets[i, :, 2:4], c_t, s_t, w_h)
        dets[i, :, :2] = pt_transform_preds(dets[i, :, 0:2], c_t, s_t, w_h)
        dets[i, :, 2:4] = pt_transform_preds(dets[i, :, 2:4], c_t, s_t, w_h)
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = (
                torch.cat(
                    [
                        dets[i, inds, :4].to(torch.float32),
                        dets[i, inds, 4:5].to(torch.float32),
                    ],
                    axis=1,
                )
                .cpu()
                .detach()
                .tolist()
            )
        ret.append(top_preds)
    return ret
