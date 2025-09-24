## HockeyMOM command-line tools (hm*)

This folder contains user-facing CLI commands that are installed with hm-prefixed names when you build/install the wheel. Each command targets a common workflow like stitching two camera angles, tracking players, clipping highlights, etc.

### Installed commands
- **hmtrack** — Run tracking/inference on a game video directory
- **hmstitch** — Stitch left/right camera videos into one panorama
- **hmcreate_control_points** — Compute Hugin control points with SuperPoint/LightGlue and update a .pto
- **hmplayers** — Analyze tracked players and generate per-player timestamp files
- **hmfind_ice_rink** — Detect the ice rink mask and save configuration for a game
- **hmvideo_clipper** — Create a highlight reel from timestamps or a file list
- **hmorientation** — Inspect a game directory and label left/right camera sets
- **hmconcatenate_videos** — Normalize and concatenate multiple videos with ffmpeg

### Conventions and prerequisites
- **Game directory layout**: Most tools expect a game-id directory under `$HOME/Videos/<game-id>` with `config.yaml` and a stitched frame `s.png` (some tools will generate these). Example: `$HOME/Videos/ev-stockton-1`.
- **GPU acceleration**: Many commands can use CUDA for decoding/encoding or ML inference. Set `--gpus` or device flags as needed. CPU-only can work but is slower.
- **Common options**: A number of flags are shared across commands via `hm_opts` (e.g., `--game-id`, `--gpus`, `--output`, `--lfo/--rfo`, `--start-frame-time`). When in doubt, run `<cmd> --help`.

---

## hmtrack — Tracking and post-processing

Run end-to-end tracking on a game: optional stitching, detections, tracking, jersey numbers, pose, boundaries, and render the output video. Can operate from raw left/right videos or a pre-stitched file.

### Quick start
```bash
# From a configured game directory
hmtrack --game-id ev-stockton-1 --output tracking_output-with-audio.mp4

# From an explicit input file
hmtrack --input-video path/to/stitched_output-with-audio.mp4 --output tracking.mp4
```

### Notable options
- `--game-id <id>` — Use `$HOME/Videos/<id>` and its config
- `--input-video <file|dir|left,right>` — Override inputs (two files or a dir implies stitching)
- `--stitch` / `--force-stitching` — Force stitching phase
- `--detect-jersey-numbers`, `--plot-jersey-numbers` — Jersey number detection and overlay
- `--plot-pose` — Pose skeletons overlay
- `--plot-tracking`, `--plot-trajectories` — Visualize boxes and paths
- `--cvat-output` — Export data for CVAT
- `--save-tracking-data`, `--save-detection-data`, `--save-camera-data` — Write CSV artifacts
- `--output-video <file>` and `--no-save-video` — Control rendered output
- `--checkpoint` / `--detector` / `--reid` — Override model weights

---

## hmstitch — Stitch left/right camera videos

Stitch the two perspectives into a single panorama, optionally writing only the configuration (pto, mapping files) or generating the stitched video.

### Quick start
```bash
# Auto-discover from game-id and generate a stitched video
hmstitch --game-id ev-stockton-1 -o stitched_output-with-audio.mp4

# Configure only (no render)
hmstitch --game-id ev-stockton-1 --configure-only
```

### Notable options
- `--game-id <id>` — Scan `$HOME/Videos/<id>` for left/right files and chapters
- `--lfo` / `--rfo <frames>` — Frame offsets (if sync known)
- `--stitch-frame-time HH:MM:SS` — Choose reference frame for alignment; `--start-frame-time` to start processing at a time
- `--max-control-points N` — Control points for homography
- `--blend-mode laplacian|multiblend|gpu-hard-seam` — Blending mode
- `--batch-size`, `--stitch-cache-size`, `--multi-gpu` — Performance tuning
- `--show` / `--show-scaled` — Preview frames

---

## hmcreate_control_points — Compute and write Hugin control points

Synchronize by audio (optional), extract frames, compute feature matches with SuperPoint + LightGlue, and update the Hugin `.pto` project file with control points. Helpful before stitching.

### Quick start
```bash
# From game-id
hmcreate_control_points --game-id ev-stockton-1 --max-control-points 300

# From explicit videos
hmcreate_control_points --left left.mp4 --right right.mp4 --synchronize-only
```

### Notable options
- `--left` / `--right` — Input videos (or supply `--game-id`)
- `--synchronize-only` — Print left/right frame offsets and exit
- `--max-control-points N` — Limit control point matches
- `--scale <float>` — Downscale when optimizing/visualizing

---

## hmplayers — Analyze tracked players and emit highlight ranges

Uses `tracking.csv` and `camera.csv` produced by `hmtrack` to infer per-shift intervals for jersey numbers and generate per-player timestamp files, plus a helper shell script to assemble highlights.

### Quick start
```bash
hmplayers --game-id ev-stockton-1
```

### Outputs
- Writes files like `player_29.txt` with HH:MM:SS ranges and a `make_player_highlights.sh` script in the game directory.

---

## hmfind_ice_rink — Detect ice rink mask and save

Run the rink segmentation model on the stitched frame `s.png` in a game directory, save the combined mask and related metadata into the private config, and print centroid/edges info.

### Quick start
```bash
hmfind_ice_rink --game-id ev-stockton-1 --show
```

### Notable options
- `--scale <float>` — Optional image scaling
- `--device cpu|cuda[:N]` — Choose inference device
- `--force` — Recompute and overwrite existing mask configuration

---

## hmvideo_clipper — Make a highlight reel with bumpers

Produce a single video by concatenating: a 3s title card for each clip, then the clip with an overlaid “Label <N>” in the top-right. Supports two input styles: a timestamp file for a single source video, or a list of pre-cut clip files.

### Quick start
```bash
# From timestamps
hmvideo_clipper -i input.mp4 -t timestamps.txt "Team vs Opponent"

# From a file list
hmvideo_clipper --video-file-list clips.txt "Team vs Opponent"
```

### Timestamps format
- Each non-empty line: `START [END]` with times in `HH:MM:SS` (END optional to continue to EOF)
- Example:
	- `00:02:15 00:03:05`
	- `00:10:00 00:11:42`

### Options
- `--threads` / `-j N` — Parallelize per-clip processing
- `--temp-dir DIR` — Work directory for intermediates
- Environment: set `VIDEO_CLIPPER_HQ=1` for lossless intermediates; default uses high-quality NVENC settings

---

## hmorientation — Label left/right inputs and save to private config

Discover available video files in a game directory, infer which perspective is left vs right using rink masks, and write an ordered chapter list per side to the private config.

### Quick start
```bash
hmorientation --game-id ev-stockton-1
```

---

## hmconcatenate_videos — Normalize and concatenate multiple videos

Normalize resolution, fps, and audio across inputs, optionally trim each segment, and concatenate to one output using ffmpeg. Supports GPU NVENC and CPU x265 encoders.

### Quick start
```bash
# NVENC VBR
hmconcatenate_videos --use-gpu --inputs a.mp4 b.mp4 c.mp4 -o out.mkv \
	--video-quality vbr --cq 19 --b-v 30M --maxrate 60M

# Lossless (NVENC)
hmconcatenate_videos --use-gpu --video-quality lossless --inputs a.mp4 b.mp4 -o out.mkv

# x265 CRF
hmconcatenate_videos --inputs a.mp4 b.mp4 -o out.mp4 --video-quality crf --crf 20 --preset slow
```

### Notable options
- `--clip START-END` — Per-input trims; repeat per input or leave empty for full
- `--aspect-mode pad|crop|stretch` — Aspect handling when normalizing
- `--force-res WxH`, `--force-fps NUM/DEN` — Force target profile
- `--audio-rate`, `--audio-channels`, `--audio-bitrate` — Audio normalization

---

## Shared/common flags (hm_opts)

Some commands accept these shared flags:
- `--game-id <id>` — Target game
- `--gpus "0,1"` — GPU selection for ML/inference
- `--output <file>` — Output artifact path
- `--lfo` / `--rfo` — Frame offsets for left/right
- `--start-frame-time HH:MM:SS` and `--stitch-frame-time HH:MM:SS` — Time selection
- `--fp16` — Mixed precision in some stages
- `--show` / `--show-scaled` — Preview frames
- `--cache-size`, `--stitch-cache-size` — Pipeline buffering

---

## Tips
- Use `hmorientation` first on new game folders to auto-label left/right and save chapter lists.
- If alignment drifts, run `hmcreate_control_points` to refresh control points; then `hmstitch`.
- After stitching, run `hmfind_ice_rink` to cache the rink mask; `hmtrack` will use it to improve boundaries.
- Use `hmplayers` to generate per-player time ranges, then `hmvideo_clipper` or `hmconcatenate_videos` to produce highlight reels.

---

## Help
Each command supports `--help` to print the full set of options and defaults.
