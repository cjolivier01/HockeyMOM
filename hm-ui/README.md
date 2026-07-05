# hm-ui

Rust operator UI for HockeyMOM runtime camera controls.

`hm-ui` is a sidecar process. The Python tracker owns the video pipeline and writes a JSON control spec/state file; `hm-ui` renders the controls and writes value changes back.

## Build

```bash
bazelisk build //hm-ui:hm-ui
```

or:

```bash
cargo build --locked --manifest-path hm-ui/Cargo.toml
```

## Use With hmtrack

```bash
hmtrack --game-id <game> --camera-ui=1 --camera-ui-backend=rust
```

For local source-tree runs, this also builds the sidecar first:

```bash
make hmtrack-rust-ui ARGS="--game-id <game>"
```

The Python bridge searches for `hm-ui` in this order:

1. `HM_UI_BIN`
2. `PATH`
3. Bazel runfiles
4. installed wheel path, `hmlib/bin/hm-ui`
5. `bazel-bin/hmlib/bin/hm-ui` or `bazel-bin/hm-ui/hm-ui-bin`
6. `hm-ui/target/release/hm-ui` or `hm-ui/target/debug/hm-ui`

The hmlib wheel bundles `hm-ui` at `hmlib/bin/hm-ui`.
