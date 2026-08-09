// TODO(jb-doc): module docs — that every button here has a ctl verb behind it, and that
// the rule runs that way round rather than the other.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use watershed::SaveOptions;

use crate::document::Document;
use crate::material;
use crate::preset::Preset;
use crate::view::{EditorCamera, ViewRange, fit_camera};

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NewDialog>()
            .init_resource::<FilePath>()
            .add_systems(EguiPrimaryContextPass, (toolbar, new_dialog, legend));
    }
}

/// TODO(jb-doc): why the dialog's fields are held here rather than read out of the
/// document — that a dialog is what you are *about* to make, and the document is what you
/// have.
#[derive(Resource)]
struct NewDialog {
    open: bool,
    width: u32,
    height: u32,
    seed: u32,
    preset: Preset,
}

impl Default for NewDialog {
    fn default() -> Self {
        Self {
            open: false,
            width: 1024,
            height: 1024,
            seed: 1,
            preset: Preset::default(),
        }
    }
}

#[derive(Resource)]
struct FilePath(String);

impl Default for FilePath {
    fn default() -> Self {
        // A text field rather than a native file dialog: the editor is driven far more
        // often than it is clicked, and a path is what the ctl takes anyway.
        Self("terrain.watershed".to_owned())
    }
}

fn toolbar(
    mut contexts: EguiContexts,
    mut document: ResMut<Document>,
    mut dialog: ResMut<NewDialog>,
    mut path: ResMut<FilePath>,
    camera: Single<(&mut Transform, &mut Projection), With<EditorCamera>>,
) -> Result {
    let context = contexts.ctx_mut()?;
    let busy = document.is_busy();
    // Taken apart before the panel: the closure is `FnMut`, so it cannot consume a
    // `Single` captured from outside it.
    let (mut transform, mut projection) = camera.into_inner();

    // A panel is shown *inside* a `Ui` since egui 0.35, so the root one over the viewport
    // has to be built by hand. Windows and areas still take the context, which is why only
    // this system does it.
    let mut viewport = egui::Ui::new(
        context.clone(),
        "viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(context.viewport_rect()),
    );

    egui::Panel::top("toolbar").show(&mut viewport, |ui| {
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(!busy, egui::Button::new("New…")).clicked() {
                dialog.open = true;
            }

            ui.separator();

            ui.add(
                egui::TextEdit::singleline(&mut path.0)
                    .desired_width(220.0)
                    .hint_text("path"),
            );
            if ui.add_enabled(!busy, egui::Button::new("Open")).clicked() {
                report(document.start_load(path.0.clone().into()));
            }
            if ui
                .add_enabled(
                    !busy && document.terrain().is_some(),
                    egui::Button::new("Save"),
                )
                .clicked()
            {
                report(document.start_save(path.0.clone().into(), SaveOptions::document()));
            }

            ui.separator();

            let active = document.active().to_owned();
            let names = document.field_names();
            egui::ComboBox::from_id_salt("field")
                .selected_text(&active)
                .show_ui(ui, |ui| {
                    for name in &names {
                        if ui.selectable_label(*name == active, name).clicked() {
                            report(document.set_active(name));
                        }
                    }
                });

            ui.separator();

            let has_water = document
                .terrain()
                .is_some_and(|terrain| terrain.water().is_some());
            if ui
                .add_enabled(
                    !busy && document.terrain().is_some(),
                    egui::Button::new("Solve water"),
                )
                .clicked()
            {
                report(document.start_solve());
            }
            if ui
                .add_enabled(!busy && has_water, egui::Button::new("Reset water"))
                .clicked()
            {
                report(document.reset_water());
            }

            ui.separator();

            if ui.button("Fit").clicked()
                && let Some(terrain) = document.terrain()
            {
                fit_camera(&mut transform, &mut projection, terrain.size);
            }

            ui.separator();

            // Whatever the run is doing takes the right-hand end, because a job in flight
            // is the answer to every "why has nothing changed".
            match (document.job(), document.error()) {
                (Some(kind), _) => {
                    ui.spinner();
                    ui.label(format!("{}…", kind.name()));
                }
                (None, Some(error)) => {
                    ui.colored_label(egui::Color32::from_rgb(200, 80, 70), error);
                }
                (None, None) => {
                    ui.label(match document.terrain() {
                        Some(terrain) => format!(
                            "{}x{}  {} field(s){}",
                            terrain.size.x,
                            terrain.size.y,
                            terrain.fields.len(),
                            if has_water { "  water" } else { "" },
                        ),
                        None => "no document".to_owned(),
                    });
                }
            }
        });
    });

    Ok(())
}

fn new_dialog(
    mut contexts: EguiContexts,
    mut document: ResMut<Document>,
    mut dialog: ResMut<NewDialog>,
) -> Result {
    if !dialog.open {
        return Ok(());
    }
    let context = contexts.ctx_mut()?;
    let mut open = dialog.open;

    egui::Window::new("New terrain")
        .open(&mut open)
        .resizable(false)
        .show(context, |ui| {
            egui::Grid::new("new-terrain")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("width");
                    ui.add(egui::DragValue::new(&mut dialog.width).range(16..=8192));
                    ui.end_row();

                    ui.label("height");
                    ui.add(egui::DragValue::new(&mut dialog.height).range(16..=8192));
                    ui.end_row();

                    ui.label("seed");
                    ui.add(egui::DragValue::new(&mut dialog.seed));
                    ui.end_row();

                    ui.label("preset");
                    egui::ComboBox::from_id_salt("preset")
                        .selected_text(dialog.preset.name())
                        .show_ui(ui, |ui| {
                            for preset in Preset::ALL {
                                ui.selectable_value(&mut dialog.preset, preset, preset.name());
                            }
                        });
                    ui.end_row();
                });

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!document.is_busy(), egui::Button::new("Create"))
                    .clicked()
                {
                    report(document.start_new(
                        UVec2::new(dialog.width, dialog.height),
                        dialog.seed,
                        dialog.preset,
                    ));
                    dialog.open = false;
                }
                if ui.button("Cancel").clicked() {
                    dialog.open = false;
                }
            });
        });

    // Only close on the frame's own decision: the window's own X writes `open`, and
    // overwriting it unconditionally would reopen a dialog Create has just dismissed.
    if !open {
        dialog.open = false;
    }
    Ok(())
}

/// TODO(jb-doc): why the legend prints the live ends rather than the field's declared
/// range, and what a sub-3:1 ramp needs from it besides the colour.
fn legend(mut contexts: EguiContexts, document: Res<Document>, range: Res<ViewRange>) -> Result {
    if document.terrain().is_none() {
        return Ok(());
    }
    let context = contexts.ctx_mut()?;

    egui::Area::new("legend".into())
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(12.0, -12.0))
        .show(context, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(200.0);
                ui.label(egui::RichText::new(document.active()).strong());

                let (response, painter) =
                    ui.allocate_painter(egui::vec2(200.0, 14.0), egui::Sense::hover());
                let rect = response.rect;
                let steps = 64;
                for step in 0..steps {
                    let t = step as f32 / (steps - 1) as f32;
                    let colour = if range.diverging {
                        material::diverging(t * 2.0 - 1.0)
                    } else {
                        material::sequential(t)
                    };
                    let slice = egui::Rect::from_min_size(
                        egui::pos2(
                            rect.min.x + rect.width() * step as f32 / steps as f32,
                            rect.min.y,
                        ),
                        egui::vec2(rect.width() / steps as f32 + 1.0, rect.height()),
                    );
                    painter.rect_filled(slice, 0.0, to_colour(colour));
                }
                // A hairline ring, because the ramp's dark end is under 2:1 against the
                // panel and without it the strip has no visible edge.
                painter.rect_stroke(
                    rect,
                    0.0,
                    egui::Stroke::new(1.0, egui::Color32::from_gray(90)),
                    egui::StrokeKind::Inside,
                );

                ui.horizontal(|ui| {
                    ui.label(format!("{:.4}", range.low));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("{:.4}", range.high));
                    });
                });

                ui.label(
                    egui::RichText::new(if range.diverging {
                        "diverging about 0"
                    } else {
                        "fitted to the view"
                    })
                    .small()
                    .weak(),
                );
            });
        });

    Ok(())
}

fn to_colour(colour: Vec3) -> egui::Color32 {
    egui::Color32::from_rgb(
        (colour.x * 255.0).round().clamp(0.0, 255.0) as u8,
        (colour.y * 255.0).round().clamp(0.0, 255.0) as u8,
        (colour.z * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

/// A refusal here is synchronous and never touches [`Document::error`], which only a job
/// writes — so the log is the whole of where it goes, and `observe log` is what reads it.
fn report(result: Result<(), String>) {
    if let Err(error) = result {
        warn!("{error}");
    }
}
