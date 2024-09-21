import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd


def get_control_points(
    image0: Union[str, Path, torch.Tensor],
    image1: Union[str, Path, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    if not isinstance(image0, torch.Tensor):
        image0 = load_image(image0)
    if not isinstance(image1, torch.Tensor):
        image1 = load_image(image1)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot("matches.png")

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    viz2d.save_plot("keypoints.png")
    return dict(kpts0=kpts0, m_kpts0=m_kpts0, kpts1=kpts1, m_kpts1=m_kpts1)


if __name__ == "__main__":
    results = get_control_points(
        image0=f"{os.environ['HOME']}/Videos/ev-sabercats-1/left.png",
        image1=f"{os.environ['HOME']}/Videos/ev-sabercats-1/right.png",
    )
    print("Done.")
