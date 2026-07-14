use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(name = "hm-ui")]
#[command(about = "HockeyMOM runtime operator UI")]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    state: PathBuf,
    #[arg(long, default_value = "HM UI")]
    title: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct UiSpec {
    #[serde(default)]
    title: String,
    #[serde(default)]
    subtitle: String,
    #[serde(default)]
    preview_path: Option<PathBuf>,
    #[serde(default)]
    windows: Vec<WindowSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct WindowSpec {
    name: String,
    #[serde(default)]
    controls: Vec<ControlSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ControlSpec {
    name: String,
    max_value: i32,
    value: i32,
    #[serde(default)]
    default_value: Option<i32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct UiState {
    version: u32,
    updated_ms: u128,
    #[serde(default)]
    windows: BTreeMap<String, BTreeMap<String, i32>>,
    #[serde(default)]
    last_action: Option<UiAction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct UiAction {
    seq: u64,
    kind: String,
}

struct HmUiApp {
    spec_path: PathBuf,
    state_path: PathBuf,
    spec: UiSpec,
    values: BTreeMap<String, BTreeMap<String, i32>>,
    selected_window: usize,
    last_spec_modified: Option<SystemTime>,
    last_spec_poll: SystemTime,
    last_preview_modified: Option<SystemTime>,
    last_preview_poll: SystemTime,
    preview_texture: Option<egui::TextureHandle>,
    preview_status: String,
    action_seq: u64,
    last_action: Option<UiAction>,
    status: String,
}

impl HmUiApp {
    fn new(spec_path: PathBuf, state_path: PathBuf, title: String) -> Self {
        let mut app = Self {
            spec_path,
            state_path,
            spec: UiSpec {
                title,
                subtitle: "Runtime camera controls".to_string(),
                preview_path: None,
                windows: Vec::new(),
            },
            values: BTreeMap::new(),
            selected_window: 0,
            last_spec_modified: None,
            last_spec_poll: UNIX_EPOCH,
            last_preview_modified: None,
            last_preview_poll: UNIX_EPOCH,
            preview_texture: None,
            preview_status: "Waiting for preview frame".to_string(),
            action_seq: 0,
            last_action: None,
            status: "Starting".to_string(),
        };
        if let Err(err) = app.reload_spec(true) {
            app.status = format!("Waiting for spec: {err}");
        }
        app
    }

    fn reload_spec(&mut self, force: bool) -> Result<()> {
        let meta = fs::metadata(&self.spec_path)
            .with_context(|| format!("metadata {}", self.spec_path.display()))?;
        let modified = meta.modified().ok();
        if !force && modified.is_some() && modified == self.last_spec_modified {
            return Ok(());
        }

        let data = fs::read_to_string(&self.spec_path)
            .with_context(|| format!("read {}", self.spec_path.display()))?;
        let mut spec: UiSpec = serde_json::from_str(&data).context("parse UI spec")?;
        if spec.title.is_empty() {
            spec.title = "HM UI".to_string();
        }
        let state_values = self.read_state_values().unwrap_or_default();

        for window in &spec.windows {
            let entry = self.values.entry(window.name.clone()).or_default();
            for control in &window.controls {
                let state_value = state_values
                    .get(&window.name)
                    .and_then(|controls| controls.get(&control.name))
                    .copied();
                let initial_value = state_value.unwrap_or(control.value);
                entry
                    .entry(control.name.clone())
                    .and_modify(|value| *value = initial_value)
                    .or_insert(initial_value);
            }
            let valid_names: Vec<String> = window.controls.iter().map(|c| c.name.clone()).collect();
            entry.retain(|name, _| valid_names.contains(name));
        }
        let valid_windows: Vec<String> = spec.windows.iter().map(|w| w.name.clone()).collect();
        self.values.retain(|name, _| valid_windows.contains(name));
        if self.selected_window >= spec.windows.len() {
            self.selected_window = 0;
        }
        self.last_spec_modified = modified;
        self.spec = spec;
        self.status = "Connected".to_string();
        self.write_state()?;
        Ok(())
    }

    fn read_state_values(&self) -> Result<BTreeMap<String, BTreeMap<String, i32>>> {
        if !self.state_path.exists() {
            return Ok(BTreeMap::new());
        }
        let data = fs::read_to_string(&self.state_path)
            .with_context(|| format!("read {}", self.state_path.display()))?;
        let state: UiState = serde_json::from_str(&data).context("parse UI state")?;
        Ok(state.windows)
    }

    fn poll_spec(&mut self) {
        let now = SystemTime::now();
        if now
            .duration_since(self.last_spec_poll)
            .unwrap_or(Duration::from_secs(1))
            < Duration::from_millis(300)
        {
            return;
        }
        self.last_spec_poll = now;
        if let Err(err) = self.reload_spec(false) {
            self.status = format!("Spec error: {err}");
        }
    }

    fn poll_preview(&mut self, ctx: &egui::Context) {
        let now = SystemTime::now();
        if now
            .duration_since(self.last_preview_poll)
            .unwrap_or(Duration::from_secs(1))
            < Duration::from_millis(120)
        {
            return;
        }
        self.last_preview_poll = now;
        let Some(path) = self.spec.preview_path.clone() else {
            self.preview_status = "No preview path in spec".to_string();
            return;
        };
        let Ok(meta) = fs::metadata(&path) else {
            self.preview_status = "Waiting for preview frame".to_string();
            return;
        };
        let modified = meta.modified().ok();
        if modified.is_some() && modified == self.last_preview_modified {
            return;
        }
        match load_color_image(&path) {
            Ok(image) => {
                let options = egui::TextureOptions::LINEAR;
                if let Some(texture) = self.preview_texture.as_mut() {
                    texture.set(image, options);
                } else {
                    self.preview_texture = Some(ctx.load_texture("hm-ui-preview", image, options));
                }
                self.last_preview_modified = modified;
                self.preview_status = "Live preview".to_string();
            }
            Err(err) => {
                self.preview_status = format!("Preview load failed: {err}");
            }
        }
    }

    fn set_action(&mut self, kind: &str) {
        self.action_seq += 1;
        self.last_action = Some(UiAction {
            seq: self.action_seq,
            kind: kind.to_string(),
        });
        if let Err(err) = self.write_state() {
            self.status = format!("State write failed: {err}");
        }
    }

    fn reset_values(&mut self) {
        for window in &self.spec.windows {
            let entry = self.values.entry(window.name.clone()).or_default();
            for control in &window.controls {
                entry.insert(
                    control.name.clone(),
                    control.default_value.unwrap_or(control.value),
                );
            }
        }
        self.set_action("reset");
    }

    fn write_state(&mut self) -> Result<()> {
        let state = UiState {
            version: 1,
            updated_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            windows: self.values.clone(),
            last_action: self.last_action.clone(),
        };
        write_json_atomic(&self.state_path, &state)
            .with_context(|| format!("write {}", self.state_path.display()))?;
        self.status = "Connected".to_string();
        Ok(())
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(if self.spec.title.is_empty() {
                "HM UI"
            } else {
                &self.spec.title
            });
            ui.separator();
            ui.label(&self.spec.subtitle);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Save").clicked() {
                    self.set_action("save");
                }
                if ui.button("Reset").clicked() {
                    self.reset_values();
                }
            });
        });
    }

    fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.label("Controls");
            for (idx, window) in self.spec.windows.iter().enumerate() {
                let selected = idx == self.selected_window;
                if ui
                    .selectable_label(selected, compact_window_name(&window.name))
                    .clicked()
                {
                    self.selected_window = idx;
                }
            }
            ui.separator();
            if ui.selectable_label(false, "Commands").clicked() {
                self.selected_window = self.spec.windows.len();
            }
            if ui
                .selectable_label(false, "Status")
                .on_hover_text("Runtime connection and state file paths")
                .clicked()
            {
                self.selected_window = self.spec.windows.len() + 1;
            }
        });
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui, window_idx: usize) {
        if window_idx >= self.spec.windows.len() {
            return;
        }
        let window = self.spec.windows[window_idx].clone();
        ui.heading(compact_window_name(&window.name));
        ui.add_space(8.0);
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new(format!("controls-{}", window.name))
                .num_columns(3)
                .spacing([16.0, 10.0])
                .striped(true)
                .show(ui, |ui| {
                    for control in &window.controls {
                        let max_value = control.max_value.max(1);
                        ui.label(display_name(&control.name));
                        let value = self
                            .values
                            .entry(window.name.clone())
                            .or_default()
                            .entry(control.name.clone())
                            .or_insert(control.value);

                        let changed = if max_value == 1 {
                            let mut checked = *value > 0;
                            let changed = ui.checkbox(&mut checked, "").changed();
                            if changed {
                                *value = if checked { 1 } else { 0 };
                            }
                            changed
                        } else {
                            ui.add(
                                egui::Slider::new(value, 0..=max_value)
                                    .clamping(egui::SliderClamping::Always)
                                    .show_value(false),
                            )
                            .changed()
                        };
                        ui.monospace(format_value(&control.name, *value, max_value));
                        ui.end_row();

                        if changed {
                            if let Err(err) = self.write_state() {
                                self.status = format!("State write failed: {err}");
                            }
                        }
                    }
                });
        });
    }

    fn draw_preview(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size_before_wrap();
        let height = (available.y * 0.52).clamp(220.0, 520.0);
        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            ui.set_min_height(height);
            ui.set_width(available.x);
            if let Some(texture) = &self.preview_texture {
                let texture_size = texture.size_vec2();
                if texture_size.x > 0.0 && texture_size.y > 0.0 {
                    let max_size = egui::vec2(ui.available_width(), height);
                    let scale = (max_size.x / texture_size.x)
                        .min(max_size.y / texture_size.y)
                        .max(0.01);
                    let desired_size = texture_size * scale;
                    ui.vertical_centered(|ui| {
                        ui.add(
                            egui::Image::new(texture)
                                .fit_to_exact_size(desired_size)
                                .maintain_aspect_ratio(true),
                        );
                    });
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label(&self.preview_status);
                });
            }
        });
        ui.add_space(10.0);
    }

    fn draw_commands(&mut self, ui: &mut egui::Ui) {
        ui.heading("Commands");
        ui.add_space(8.0);
        ui.label("Common local commands");
        ui.monospace("hmtrack --game-id <game> --camera-ui=1 --camera-ui-backend=rust");
        ui.monospace("hmstitch --game-id <game>");
        ui.monospace("bazelisk build //hm-ui:hm-ui");
        ui.add_space(14.0);
        ui.label("This panel is intentionally a launcher guide for now. The tracking process remains the owner of video, detector, and stitch runtime state.");
    }

    fn draw_status(&mut self, ui: &mut egui::Ui) {
        ui.heading("Status");
        ui.add_space(8.0);
        ui.label(format!("Connection: {}", self.status));
        ui.label(format!("Spec: {}", self.spec_path.display()));
        ui.label(format!("State: {}", self.state_path.display()));
        ui.label(format!("Windows: {}", self.spec.windows.len()));
        let controls: usize = self.spec.windows.iter().map(|w| w.controls.len()).sum();
        ui.label(format!("Controls: {controls}"));
    }
}

impl eframe::App for HmUiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_spec();
        self.poll_preview(ctx);
        ctx.set_pixels_per_point(1.1);
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            self.draw_top_bar(ui);
        });
        egui::SidePanel::left("sidebar")
            .min_width(160.0)
            .resizable(false)
            .show(ctx, |ui| {
                self.draw_sidebar(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_preview(ui);
            if self.selected_window < self.spec.windows.len() {
                self.draw_controls(ui, self.selected_window);
            } else if self.selected_window == self.spec.windows.len() {
                self.draw_commands(ui);
            } else {
                self.draw_status(ui);
            }
        });
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

fn display_name(name: &str) -> String {
    name.replace('_', " ")
}

fn compact_window_name(name: &str) -> String {
    name.replace("Tracker Controls", "Tracker")
        .replace("(", "")
        .replace(")", "")
}

fn format_value(name: &str, value: i32, max_value: i32) -> String {
    if max_value == 1 {
        if value > 0 {
            "on".to_string()
        } else {
            "off".to_string()
        }
    } else if name.ends_with("_x100") {
        format!("{:.2}", value as f32 / 100.0)
    } else if name.ends_with("_x10") {
        format!("{:.1}", value as f32 / 10.0)
    } else if name.contains("Kelvin") || name.contains("Temperature") {
        format!("{value} K")
    } else if name.ends_with("_Degrees") {
        format!("{} deg", 90 - value)
    } else {
        value.to_string()
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let data = serde_json::to_vec_pretty(value)?;
    fs::write(&tmp, data)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn load_color_image(path: &Path) -> Result<egui::ColorImage> {
    let image = image::open(path)
        .with_context(|| format!("decode {}", path.display()))?
        .to_rgba8();
    let size = [image.width() as usize, image.height() as usize];
    Ok(egui::ColorImage::from_rgba_unmultiplied(
        size,
        image.as_raw(),
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([760.0, 520.0]),
        ..Default::default()
    };
    let title = args.title.clone();
    eframe::run_native(
        &args.title,
        native_options,
        Box::new(move |_cc| {
            Ok(Box::new(HmUiApp::new(
                args.spec.clone(),
                args.state.clone(),
                title.clone(),
            )))
        }),
    )
    .map_err(|err| anyhow::anyhow!("{err}"))
}
