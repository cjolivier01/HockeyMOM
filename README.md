HockeyMOM

## Docs
- Repository guidelines: see [AGENTS.md](AGENTS.md).
- How to contribute: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Web App (Flask UI)
- Install (sudo): `sudo python3 tools/webapp/install_webapp.py --watch-root /data/incoming --server-name _ --port 8008`. This creates `/opt/hm-webapp/app`, a venv at `/opt/hm-webapp/venv`, `hm-webapp.service`, and an nginx proxy on port 80 → 8008.
- Config: `/opt/hm-webapp/app/config.json`; uploads/watch root: `/data/incoming` (overridable via flag). Uninstall with `sudo python3 tools/webapp/uninstall_webapp.py`.
- Seed demo data: `/opt/hm-webapp/venv/bin/python tools/webapp/seed_demo.py --config /opt/hm-webapp/app/config.json --email demo@example.com --name "Demo User"`.
- Access: `http://localhost/`. Logs: `journalctl -u hm-webapp` and nginx error log in `/var/log/nginx/`.

## System dependencies and native tools

For a fresh Ubuntu/Debian machine you should run, from the repo root:

```bash
./env/install_deps.sh
./build_deps.sh
```

- `env/install_deps.sh` installs the system libraries needed by the native C++ tools used by HockeyMOM:
  - Hugin CLI tools (`pto_gen`, `autooptimiser`, `nona`) for stitching.
  - Enblend/Enfuse and image/codec libraries (tiff, jpeg, png, OpenEXR, FFTW, wxWidgets, etc.).
  - These are required so the Bazel/CMake builds in `external/hugin` and `external/enblend-enfuse` succeed and the CLI binaries can run.
- `build_deps.sh` runs the Hugin Bazel build:
  - Changes into `external/hugin` and runs `bazelisk build //:install_tree`.
  - This drives CMake to build (and stage an install tree for) the Hugin command-line tools.
  - Those tools are then available to `hmcreate_control_points` and `hmstitch` for pano generation.

If you are on a different distro, use `env/install_deps.sh` as a reference list of required packages and install the equivalents manually, then run `./build_deps.sh`.

If you do **not** build these native tools, you will not be able to run Hugin-based stitching from raw left/right videos, but you can still run tracking on an already-stitched panorama (see below).

## Building and running hmtrack and stitching

You can run the CLIs either from an installed wheel (preferred for end users) or directly from this repo for development.

### From an installed wheel (recommended)

After building wheels with `./bdist_wheel` and installing them into your environment (e.g. `pip install dist/hmlib-*.whl dist/hockeymom-*.whl`), the CLI entry points are available on your `PATH`:

```bash
# Full pipeline from a configured game directory (stitch + track)
hmtrack --game-id ev-stockton-1 --output tracking_output-with-audio.mp4

# Explicit stitching step
hmstitch --game-id ev-stockton-1 -o stitched_output-with-audio.mp4
```

### From the repo checkout (no wheel install)

From the repo root you can invoke the same commands via Python modules:

```bash
# Tracking (equivalent to hmtrack)
python -m hmlib.cli.hmtrack --game-id ev-stockton-1 --output tracking_output-with-audio.mp4

# Stitching (equivalent to hmstitch)
python -m hmlib.cli.stitch --game-id ev-stockton-1 -o stitched_output-with-audio.mp4
```

Ensure your `PYTHONPATH` includes the repo root when running directly (for example, see `hm_run.sh` for how we set `PYTHONPATH` in development).

## Running hmtrack with a pre-stitched or panoramic video

If you already have a stitched panorama (for example, produced by `hmstitch` on another machine or by an external tool) and you do not have Hugin/enblend built locally, you can run tracking directly on that video and skip the stitching phase:

```bash
# Using an explicit stitched video (no game-id layout required)
hmtrack --input-video path/to/stitched_panorama.mp4 --output tracking.mp4

# From the repo without installed wheel
python -m hmlib.cli.hmtrack --input-video path/to/stitched_panorama.mp4 --output tracking.mp4
```

Optionally, if you still use a game directory layout under `$HOME/Videos/<game-id>`, you can combine `--game-id` with `--input-video` so the tracker can pick up game-specific config while reading frames from your pre-stitched file.

## Unified Configuration
- HM configs (`hmlib/config/` camera/rink/game/model) and Aspen graph configs (`hmlib/config/aspen/`) can be combined.
- Aspen YAMLs are nested under a top-level `aspen:` key (e.g., `aspen.trunks`, `aspen.inference_pipeline`, `aspen.video_out_pipeline`).
- The CLI merges multiple YAML files in order using `--config` (repeatable). Later files override earlier values and add new fields.
- Backward-compat `--aspen-config` has been removed. Use `--config` exclusively.

Examples
- Single file (everything in one YAML):
  - `./hm_run.sh --config my_unified.yaml`
- Compose baseline + Aspen graph:
  - `./hm_run.sh --config hmlib/config/baseline.yaml --config hmlib/config/aspen/tracking.yaml`
- Add rink- or game-specific overrides after base files:
  - `./hm_run.sh --config base.yaml --config rink/vallco.yaml --config hmlib/config/aspen/tracking_pose.yaml`

YAML structure (snippet)
```yaml
camera:
  name: GoPro
game:
  game_id: 2024-10-01
aspen:
  inference_pipeline: [...]
  video_out_pipeline: [...]
  trunks:
    detector_factory: {...}
    tracker: {...}
```

**AspenNet Execution Modes**
- Normal (sequential): single thread runs trunks in topological order.
  
  ![AspenNet Normal](docs/images/aspennet-normal.svg)

- Threaded + CUDA streams: each trunk executes in its own thread and CUDA stream; queues connect stages.
  
  ![AspenNet Threaded (CUDA streams)](docs/images/aspennet-threaded-streams.svg)

- Threaded without streams: threaded execution on default stream/CPU; same queueing.
  
  ![AspenNet Threaded (no streams)](docs/images/aspennet-threaded-no-streams.svg)

- Configure via YAML `aspen.pipeline`: `threaded: bool`, `queue_size: int`, `cuda_streams: bool`.
- CLI toggles: `--aspen-threaded`, `--aspen-thread-queue-size`, `--aspen-thread-cuda-streams` or `--no-aspen-thread-cuda-streams`.

**Video Clipper**
- Binary: `hmlib/cli/video_clipper.py`
- New options:
  - `--blink-circle`: overlays a blinking orange circle near the top-left around the clip midpoint.
  - `--blink-pre` / `--blink-post`: seconds before/after midpoint to blink (default 2.0/2.0).
- Behavior notes:
  - Circle is a 250px radius image mask overlaid at `x=10,y=10` and blinks 0.2s on/0.2s off.
  - When trimming with timestamps, decode-side `-t` is used and overlay uses `shortest=1` to ensure exact clip duration.

**Shift Spreadsheet Parser**
- Script: `scripts/parse_shift_spreadsheet_ex.py`
- Goal clips: fixed 20s window with 15s before and 5s after the goal.
  - Constants: `GOAL_CLIP_PRE_S`, `GOAL_CLIP_POST_S` inside the script.
- Event clip scripts (non-goal): enable `--blink-circle` by default.
  - Generated scripts accept `--no-blink` to disable the blinking overlay while preserving other arguments.
  - Player shift clips are unaffected (no blinking overlay added).
- Times files location: all generated time window text files are written under a `times/` subdirectory inside the chosen output directory. Generated shell scripts reference `times/<file>.txt` accordingly.
- Shell scripts location: generated shell scripts are written under a `scripts/` subdirectory. Event scripts are further organized by team (`scripts/<TeamTag>/...`). Player shift scripts live in the `scripts/` subdir of each team folder (or legacy output folder).
 - On-ice boundary rule: when a shift ends at the exact same scoreboard time as an event, the event counts for the player leaving the ice and not for the player entering at that time. Internally, membership uses an interval that excludes the shift-start boundary and includes the shift-end boundary.
