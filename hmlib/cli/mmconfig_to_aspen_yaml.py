import argparse
import os
from typing import Any, Dict

import yaml
from mmengine.config import Config, ConfigDict


def _normalize(x):
    if isinstance(x, ConfigDict):
        x = dict(x)
    if isinstance(x, dict):
        return {k: _normalize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_normalize(v) for v in x]
    return x


def extract_model(cfg: Config) -> Dict[str, Any]:
    m = cfg.get("model", {})
    res: Dict[str, Any] = {}
    res["data_preprocessor"] = _normalize(m.get("data_preprocessor"))
    # If top-level detector exists, prefer it
    detector = cfg.get("detector", None)
    if detector is None:
        detector = m.get("detector")
    res["detector"] = _normalize(detector)
    res["tracker"] = _normalize(m.get("tracker"))
    return res


def extract_inference_pipeline(cfg: Config):
    if "inference_pipeline" in cfg:
        return _normalize(cfg["inference_pipeline"])
    # Some configs embed under cfg.model or elsewhere; fall back to empty
    return []


def convert(mm_config: str) -> Dict[str, Any]:
    cfg = Config.fromfile(mm_config)
    out: Dict[str, Any] = {}
    model = extract_model(cfg)
    out["model"] = {
        "class": "hmlib.models.end_to_end_plugin.HmEndToEnd",
        "params": model,
    }
    out["inference_pipeline"] = extract_inference_pipeline(cfg)
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert mmengine Config to Aspen YAML")
    ap.add_argument("mm_config", type=str, help="Path to mmengine python config")
    ap.add_argument("--out", type=str, default=None, help="Output YAML path")
    ap.add_argument(
        "--emit",
        type=str,
        choices=["full", "detector", "model", "inference_pipeline"],
        default="full",
        help="Which section to emit. 'full' writes model + inference_pipeline.",
    )
    args = ap.parse_args()

    cfg = Config.fromfile(args.mm_config)
    model = extract_model(cfg)
    pipeline = extract_inference_pipeline(cfg)

    if args.emit == "detector":
        det = model.get("detector")
        if det is None:
            # For plain mmdet detector configs, the whole model is the detector
            det = _normalize(cfg.get("model"))
        out_data = det
        suffix = ".detector.yaml"
    elif args.emit == "model":
        # Emit under Aspen namespace for consistency, though current plugins use detector_factory
        out_data = {
            "aspen": {
                "model": {"class": "hmlib.models.end_to_end_plugin.HmEndToEnd", "params": model}
            }
        }
        suffix = ".model.yaml"
    elif args.emit == "inference_pipeline":
        out_data = {"aspen": {"inference_pipeline": pipeline}}
        suffix = ".pipeline.yaml"
    else:
        out_data = {
            "aspen": {
                "model": {"class": "hmlib.models.end_to_end_plugin.HmEndToEnd", "params": model},
                "inference_pipeline": pipeline,
            }
        }
        suffix = ".aspen.yaml"

    out_path = args.out or os.path.splitext(args.mm_config)[0] + suffix
    with open(out_path, "w") as f:
        yaml.safe_dump(out_data, f, sort_keys=False)
    print(f"Wrote Aspen YAML to {out_path}")


if __name__ == "__main__":
    main()
