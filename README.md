HockeyMOM

## Docs
- Repository guidelines: see [AGENTS.md](AGENTS.md).
- How to contribute: see [CONTRIBUTING.md](CONTRIBUTING.md).

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
