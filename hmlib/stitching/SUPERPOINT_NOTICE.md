# Third-party notice — `superpoint.py`

`hmlib/stitching/superpoint.py` is derived from
[cvg/LightGlue](https://github.com/cvg/LightGlue), file
`lightglue/superpoint.py`, which was previously consumed here as the
`xmodels/LightGlue` submodule.

## LightGlue

Copyright 2023 ETH Zurich. Licensed under the Apache License, Version 2.0; see
`LIGHTGLUE_LICENSE.txt` in this directory for the full text. You may not use
these files except in compliance with that licence.

## SuperPoint

The SuperPoint network definition and the `superpoint_v1.pth` weights it loads
originate with Magic Leap, Inc. and carry the proprietary banner reproduced at
the top of `superpoint.py`, not the Apache-2.0 terms above. The published
SuperPoint weights are licensed for **non-commercial use only**.

This is the reason kornia — which supplies the LightGlue matcher used alongside
this file — does not ship a SuperPoint extractor: the terms are incompatible
with kornia's Apache-2.0 licence.

Anything that redistributes `hmlib` (the wheel, the container image) therefore
redistributes this file and triggers a download of those weights at first use.
Review whether that is acceptable for the distribution channel in question.
`kornia.feature.DISK` and `kornia.feature.ALIKED`, both Apache-2.0 and both
with pretrained LightGlue weights, are drop-in alternatives that carry no such
restriction — at the cost of different detections than the current
calibrations were computed with.
