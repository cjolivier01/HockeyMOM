Jersey Number Plugins (Aspen)

Overview
- Two jersey recognition plugins are available:
  - Simple ROI + MMOCR: `hmlib/aspen/plugins/jersey_pose.py`
  - Koshkina-style framework: `hmlib/aspen/plugins/jersey_koshkina.py`

Configs (aspen-namespaced)
- Simple: `hmlib/config/aspen/tracking_pose_pose_det_jersey.yaml`
- Koshkina: `hmlib/config/aspen/tracking_pose_pose_det_jersey_koshkina.yaml`

All Aspen YAMLs are nested under a top-level `aspen:` key. Use them via hmtrack `--config` (repeatable), e.g.:
- `./hm_run.sh --config hmlib/config/aspen/tracking_pose_pose_det_jersey.yaml`

Koshkina Plugin Parameters (jersey_numbers.params)
- ROI selection
  - `roi_mode`: `bbox` (default), `pose`, or `sam`
    - `pose` uses shoulders+hips to form a torso ROI (matched to tracks by IoU).
    - `sam` refines bbox/pose ROI using Segment Anything (box prompt).
  - SAM options (optional):
    - `sam_enabled`: true|false (default false)
    - `sam_checkpoint`: path to SAM weights
    - `sam_model_type`: e.g., `vit_b`
    - `sam_device`: cuda|cpu

- Scene Text Recognition (STR)
  - `str_backend`: `mmocr` (default) or `parseq`
    - `mmocr`: uses MMOCR detection+recognition over packed ROI tiles.
    - `parseq`: still uses MMOCR detection for polygons, then crops and runs PARSeq on each crop.
      - `parseq_weights`: (optional) path to PARSeq weights; otherwise uses pretrained builder.
      - `parseq_device`: (optional) device string, defaults to image device.
    - Fallback: if PARSeq is unavailable, trunk logs and uses MMOCR recognition.

- Legibility Classifier (optional)
  - `legibility_enabled`: true|false (default false)
  - `legibility_weights`: path to resnet34 binary head weights (trained on jersey crops)
  - `legibility_threshold`: default 0.5
  - Fallback: if weights not provided, behaves as pass-through.

- Robust Track Aggregation (optional)
  - `vote_window`: max age (frames) of votes used for per-track aggregation
  - `vote_decay`: per-frame decay applied to older votes
  - `vote_filter_thresh`: ignore individual votes below this weight
  - `vote_sum_thresh`: require aggregated evidence above this to emit a number

- Side-View Sleeve ROIs (optional; requires `roi_mode: pose`)
  - `side_view_enabled`: true|false (default false)
  - `side_view_shoulder_ratio_thresh`: smaller means "more strictly side-on"
  - `side_view_vote_scale`: extra weight for sleeve votes when side-on

- ReID Occlusion / Outlier Removal (optional)
  - `reid_enabled`: true|false
  - `reid_backend`: `resnet` (default) or `centroid`
    - `resnet`: torchvision resnet18/34 embeddings
      - `reid_backbone`: `resnet18`|`resnet34`
    - `centroid`: tries to import/load a centroid-reid backend
      - `centroid_reid_path`: optional path to repo or model
      - or set env var `HM_CENTROID_REID_PATH`
      - `centroid_reid_device`: optional device string
  - `reid_threshold`: Mahalanobis distance threshold (default 3.0)
  - Fallback: if centroid-reid is unavailable, logs and uses resnet embeddings.

Rendering
- Add `--plot-jersey-numbers` to overlay jersey numbers on output video.
- Results are written into `data['jersey_results']` and saved with `SaveTrackingPlugin`.

Examples
- Enable pose RoIs and legibility filter:
  - `roi_mode: pose`
  - `legibility_enabled: true`
  - `legibility_weights: /abs/path/to/hockey_legibility.pth`

- Enable PARSeq STR (if installed) and centroid-reid (if available):
  - `str_backend: parseq`
  - `parseq_weights: /abs/path/to/parseq.ckpt`
  - `reid_enabled: true`
  - `reid_backend: centroid`
  - `centroid_reid_path: /abs/path/to/centroid-reid`

Fallbacks
- If any optional backend (SAM, PARSeq, centroid-reid) is missing or fails to initialize, the trunk logs a single INFO line and falls back to the default behavior without interrupting the pipeline.
