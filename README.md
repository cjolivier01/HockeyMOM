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
