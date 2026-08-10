// TODO(jb-doc): module docs — that every button here has a ctl verb behind it, and that
// the rule runs that way round rather than the other.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use watershed::brush::Brush;
use watershed::layer::{Layer, LayerOp, Mask, Remap, SlopeMode};
use watershed::noise::{NoiseKind, NoiseSpec};
use watershed::raster::Raster;
use watershed::{FieldId, SaveOptions};

use crate::brush::{BrushSettings, target_of};
use crate::document::{Baked, Document};
use crate::edit::{
    BLENDS, BRUSH_MODES, Edit, NOISE_KINDS, SLOPE_MODES, blend_name, brush_mode_name,
    noise_kind_name, op_name, parse_region_output, region_output_name, slope_mode_name,
};
use crate::material;
use crate::preset::Preset;
use crate::view::{EditorCamera, FreeView, ViewRange, fit_camera};

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NewDialog>()
            .init_resource::<FilePath>()
            .init_resource::<AddLayer>()
            .add_systems(EguiPrimaryContextPass, (panels, new_dialog, legend));
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

/// The op the add button will make. Held across frames because a combo box is a choice
/// standing until it is acted on, exactly as the new-terrain dialog's fields are.
#[derive(Resource)]
struct AddLayer(String);

impl Default for AddLayer {
    fn default() -> Self {
        Self("noise".to_owned())
    }
}

/// Points the layer stack asks for. Wide enough for a region table's columns, and the
/// panel is resizable from there.
const PANEL_WIDTH: f32 = 300.0;

/// The ops a button can make, against the [`crate::edit::parse_op`] grammar the ctl takes.
/// A regions op is deliberately absent from both — a region table is not something either
/// a command line or a single button can write.
const ADDABLE: [&str; 5] = ["noise", "constant", "fieldref", "slope", "paint"];

/// Both panels in one system, because egui lays a panel out against what the panels before
/// it have already taken: shown from two systems, each would build its own root `Ui` over
/// the whole window and the second would sit on top of the first.
fn panels(
    mut contexts: EguiContexts,
    mut document: ResMut<Document>,
    mut dialog: ResMut<NewDialog>,
    mut path: ResMut<FilePath>,
    mut add: ResMut<AddLayer>,
    mut brush: ResMut<BrushSettings>,
    mut free_view: ResMut<FreeView>,
    camera: Single<(&mut Transform, &mut Projection), With<EditorCamera>>,
) -> Result {
    let context = contexts.ctx_mut()?;
    let window = context.viewport_rect();
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

    // The fit reads the fractions the *previous* frame measured, which is right: a panel
    // has to have been laid out before there is anything for a fit to avoid, and one frame
    // of lag on a panel being dragged is invisible against a fit that is a jump.
    let free = *free_view;
    egui::Panel::top("toolbar").show(&mut viewport, |ui| {
        toolbar(
            ui,
            &mut document,
            &mut dialog,
            &mut path,
            &mut transform,
            &mut projection,
            free,
        );
    });

    egui::Panel::right("layers")
        .default_size(PANEL_WIDTH)
        .show(&mut viewport, |ui| {
            layer_stack(ui, &mut document, &mut add, &mut brush);
        });

    // What is left after both panels is the part of the window the world can be seen in,
    // kept as a fraction of the window so that reading it cannot change it — see
    // [`FreeView`] for the loop that spelling it any other way opens.
    let free = free_rect(&mut viewport);
    *free_view = FreeView::new(
        Rect::from_corners(
            Vec2::new(free.min.x, free.min.y),
            Vec2::new(free.max.x, free.max.y),
        ),
        Vec2::new(window.width(), window.height()),
    );
    Ok(())
}

/// What the panels have left the world.
fn free_rect(viewport: &mut egui::Ui) -> egui::Rect {
    viewport.available_rect_before_wrap()
}

fn toolbar(
    ui: &mut egui::Ui,
    document: &mut Document,
    dialog: &mut NewDialog,
    path: &mut FilePath,
    transform: &mut Transform,
    projection: &mut Projection,
    free: FreeView,
) {
    let busy = document.is_busy();
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
            let result = document.start_load(path.0.clone().into());
            report(document, result);
        }
        if ui
            .add_enabled(
                !busy && document.terrain().is_some(),
                egui::Button::new("Save"),
            )
            .clicked()
        {
            let result = document.start_save(path.0.clone().into(), SaveOptions::document());
            report(document, result);
        }

        ui.separator();

        let active = document.active().to_owned();
        let names = document.field_names();
        egui::ComboBox::from_id_salt("field")
            .selected_text(&active)
            .show_ui(ui, |ui| {
                for name in &names {
                    if ui.selectable_label(*name == active, name).clicked() {
                        let result = document.set_active(name);
                        report(document, result);
                    }
                }
            });

        ui.separator();

        let whole = document.baked() == Baked::Whole && !document.is_dirty();
        let has_water = document
            .terrain()
            .is_some_and(|terrain| terrain.water().is_some());

        // The whole-document bake. An edit only re-bakes what is on screen, so this is how
        // the rest of the document catches up without solving anything.
        if ui
            .add_enabled(
                !busy && !whole && document.terrain().is_some(),
                egui::Button::new("Bake all"),
            )
            .on_hover_text("bake the whole document, not just the view")
            .clicked()
        {
            let result = document.start_bake(None);
            report(document, result);
        }

        // Enabled whenever there is a document, and it bakes first if it has to. Gating it
        // on the document being wholly baked is what it did first, and a disabled button is
        // indistinguishable from a broken one: after any edit it went quietly dead with
        // nothing on screen saying why.
        if ui
            .add_enabled(
                !busy && document.terrain().is_some(),
                egui::Button::new("Solve water"),
            )
            .on_hover_text(if whole {
                "solve the water"
            } else {
                "bake the whole document, then solve"
            })
            .clicked()
        {
            let result = document.solve_with_bake();
            report(document, result);
        }
        if ui
            .add_enabled(!busy && has_water, egui::Button::new("Reset water"))
            .clicked()
        {
            let result = document.reset_water();
            report(document, result);
        }

        ui.separator();

        if ui.button("Fit").clicked()
            && let Some(terrain) = document.terrain()
        {
            let size = terrain.size;
            fit_camera(transform, projection, size, free);
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
                        "{}x{}  {} field(s){}{}",
                        terrain.size.x,
                        terrain.size.y,
                        terrain.fields.len(),
                        if has_water { "  water" } else { "" },
                        if whole { "" } else { "  preview" },
                    ),
                    None => "no document".to_owned(),
                });
            }
        }
    });
}

/// The stack of the field the toolbar has selected, which is what makes moisture and
/// temperature editable by this panel with nothing about them written here.
fn layer_stack(
    ui: &mut egui::Ui,
    document: &mut Document,
    add: &mut AddLayer,
    brush: &mut BrushSettings,
) {
    let active = document.active().to_owned();
    let names = document.field_names();
    if document.terrain().is_none() {
        ui.label("no document");
        return;
    }

    ui.horizontal(|ui| {
        ui.heading(&active);
        if document.baked() != Baked::Whole || document.is_dirty() {
            ui.label(egui::RichText::new("preview").small().weak());
        }
    });

    let mut changed = false;
    let from_brush = brush_controls(ui, document, &mut brush.0);
    // Structural changes are collected rather than made: the list is being iterated, and
    // an add or a remove partway through it is the one thing that cannot be done in place.
    let mut structural: Option<Edit> = from_brush;

    egui::ScrollArea::vertical().show(ui, |ui| {
        let Some(terrain) = document.terrain_mut() else {
            return;
        };
        // Read before the field is borrowed. The water solve reads its height one texel
        // per cell, so a coarse one is a document that can never solve.
        let pinned = crate::edit::is_solve_height(terrain, &active);
        let Some(field) = terrain.field_mut(&active) else {
            return;
        };

        egui::Grid::new("field-properties")
            .num_columns(2)
            .show(ui, |ui| {
                // A shift is the one edit that discards the bake rather than patching it,
                // since it changes the raster the field is written onto.
                //
                // Routed through `Edit::Set` rather than written here, so the rule about
                // the water spec's height field lives in one place. Greying it out is a
                // hint and not the guard — and only while the field is *already* right, so
                // a document that arrived at a coarse height some other way can be put
                // back rather than being locked out of the panel that would repair it.
                ui.label("shift");
                let mut shift = field.shift;
                let settled = pinned && shift == 0;
                if ui
                    .add_enabled(!settled, egui::DragValue::new(&mut shift).range(0..=8))
                    .on_disabled_hover_text("the water solve reads this field one texel per cell")
                    .changed()
                {
                    structural = Some(Edit::Set {
                        path: format!("{active}.shift"),
                        words: vec![shift.to_string()],
                    });
                }
                ui.end_row();

                ui.label("range");
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(egui::DragValue::new(&mut field.range.0).speed(0.01))
                        .changed();
                    changed |= ui
                        .add(egui::DragValue::new(&mut field.range.1).speed(0.01))
                        .changed();
                });
                ui.end_row();
            });

        ui.separator();

        let count = field.layers.len();
        for index in 0..count {
            let layer = &mut field.layers[index];
            let title = format!("{index}  {}", op_name(&layer.op));

            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut layer.enabled, "").changed();
                // Plain ASCII: egui's default font has no arrows or multiplication sign,
                // and a glyph it cannot draw comes out as an empty box rather than as
                // nothing, which reads as a broken button rather than a missing one.
                if ui
                    .add_enabled(index > 0, egui::Button::new("^").small())
                    .on_hover_text("move up")
                    .clicked()
                {
                    structural = Some(Edit::Move {
                        field: active.clone(),
                        index,
                        to: index - 1,
                    });
                }
                if ui
                    .add_enabled(index + 1 < count, egui::Button::new("v").small())
                    .on_hover_text("move down")
                    .clicked()
                {
                    structural = Some(Edit::Move {
                        field: active.clone(),
                        index,
                        to: index + 1,
                    });
                }
                if ui.button("x").on_hover_text("remove").clicked() {
                    structural = Some(Edit::Remove {
                        field: active.clone(),
                        index,
                    });
                }
                ui.label(egui::RichText::new(&title).strong());
            });

            egui::CollapsingHeader::new(crate::edit::op_summary(&layer.op))
                .id_salt(("layer", index))
                .show(ui, |ui| {
                    changed |= layer_body(ui, layer, &names, index);
                });
        }

        ui.separator();
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("add-op")
                .selected_text(&add.0)
                .width(110.0)
                .show_ui(ui, |ui| {
                    for kind in ADDABLE {
                        ui.selectable_value(&mut add.0, kind.to_owned(), kind);
                    }
                });
            if ui.button("Add layer").clicked() {
                structural = Some(Edit::Add {
                    field: active.clone(),
                    op: default_op(&add.0, &names),
                });
            }
        });
    });

    if let Some(edit) = structural {
        let result = document.apply(&edit).map(|_| ());
        report(document, result);
    } else if changed {
        document.note_edit();
    }
}

/// The brush, and where a drag over the world would land. Answers with an edit rather than
/// making one, because the only button here adds a layer to the stack the panel below is
/// about to iterate.
///
/// TODO(jb-doc): why the target is shown even though nothing here can choose it, and what a
/// panel that only offered the numbers would leave a person guessing about.
fn brush_controls(ui: &mut egui::Ui, document: &Document, brush: &mut Brush) -> Option<Edit> {
    let mut edit = None;
    let active = document.active().to_owned();
    let target = target_of(document);

    egui::CollapsingHeader::new("brush")
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("brush-properties")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("mode");
                    egui::ComboBox::from_id_salt("brush-mode")
                        .selected_text(brush_mode_name(brush.mode))
                        .show_ui(ui, |ui| {
                            for mode in BRUSH_MODES {
                                ui.selectable_value(&mut brush.mode, mode, brush_mode_name(mode));
                            }
                        });
                    ui.end_row();

                    ui.label("radius");
                    ui.add(
                        egui::DragValue::new(&mut brush.radius_cells)
                            .speed(0.5)
                            .range(0.0..=512.0),
                    )
                    .on_hover_text("document cells");
                    ui.end_row();

                    ui.label("falloff");
                    ui.add(
                        egui::DragValue::new(&mut brush.falloff)
                            .speed(0.01)
                            .range(0.0..=1.0),
                    );
                    ui.end_row();

                    ui.label("strength");
                    ui.add(egui::DragValue::new(&mut brush.strength).speed(0.01));
                    ui.end_row();

                    ui.label("value");
                    ui.add(egui::DragValue::new(&mut brush.value).speed(0.01))
                        .on_hover_text("what `set` moves toward");
                    ui.end_row();
                });

            match &target {
                Some((field, index)) => {
                    ui.label(
                        egui::RichText::new(format!("drag paints {field} layer {index}"))
                            .small()
                            .weak(),
                    );
                }
                None => {
                    ui.label(
                        egui::RichText::new(format!("{active} has no paint layer"))
                            .small()
                            .weak(),
                    );
                    if ui.button("Add paint layer").clicked() {
                        edit = Some(Edit::Add {
                            field: active.clone(),
                            op: LayerOp::Paint(Raster::default()),
                        });
                    }
                }
            }
        });

    edit
}

fn layer_body(ui: &mut egui::Ui, layer: &mut Layer, names: &[String], salt: usize) -> bool {
    let mut changed = false;

    egui::Grid::new(("layer-properties", salt))
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("blend");
            egui::ComboBox::from_id_salt(("blend", salt))
                .selected_text(blend_name(layer.blend))
                .show_ui(ui, |ui| {
                    for blend in BLENDS {
                        changed |= ui
                            .selectable_value(&mut layer.blend, blend, blend_name(blend))
                            .changed();
                    }
                });
            ui.end_row();

            ui.label("amplitude");
            changed |= ui
                .add(egui::DragValue::new(&mut layer.amplitude).speed(0.01))
                .changed();
            ui.end_row();
        });

    ui.separator();
    changed |= mask_editor(ui, &mut layer.mask, names, salt);
    ui.separator();
    changed |= op_editor(ui, &mut layer.op, names, salt);
    changed
}

fn mask_editor(ui: &mut egui::Ui, mask: &mut Mask, names: &[String], salt: usize) -> bool {
    let mut changed = false;
    let current = match mask {
        Mask::Constant(_) => "constant",
        Mask::Painted(_) => "painted",
        Mask::Field(..) => "field",
    };

    ui.horizontal(|ui| {
        ui.label("mask");
        egui::ComboBox::from_id_salt(("mask", salt))
            .selected_text(current)
            .width(110.0)
            .show_ui(ui, |ui| {
                // A painted mask is not offered: there is nothing here that paints one, and
                // a shape that cannot be filled in is worse than one that cannot be chosen.
                if ui
                    .selectable_label(current == "constant", "constant")
                    .clicked()
                    && current != "constant"
                {
                    *mask = Mask::Constant(1.0);
                    changed = true;
                }
                if ui.selectable_label(current == "field", "field").clicked() && current != "field"
                {
                    *mask = Mask::Field(first_field(names), Remap::IDENTITY);
                    changed = true;
                }
            });
    });

    match mask {
        Mask::Constant(value) => {
            changed |= ui
                .add(egui::DragValue::new(value).speed(0.01).range(0.0..=1.0))
                .changed();
        }
        Mask::Painted(raster) => {
            ui.label(format!("{}x{} painted", raster.width(), raster.height()));
        }
        Mask::Field(id, remap) => {
            changed |= field_combo(ui, ("mask-field", salt), id, names);
            ui.horizontal(|ui| {
                ui.label("from");
                changed |= drag(ui, &mut remap.from.0);
                changed |= drag(ui, &mut remap.from.1);
            });
            ui.horizontal(|ui| {
                ui.label("to");
                changed |= drag(ui, &mut remap.to.0);
                changed |= drag(ui, &mut remap.to.1);
            });
        }
    }
    changed
}

fn op_editor(ui: &mut egui::Ui, op: &mut LayerOp, names: &[String], salt: usize) -> bool {
    let mut changed = false;
    ui.label(egui::RichText::new(op_name(op)).strong());

    match op {
        LayerOp::Constant(value) => {
            changed |= drag(ui, value);
        }

        LayerOp::Noise(spec) => {
            egui::Grid::new(("noise", salt))
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("kind");
                    egui::ComboBox::from_id_salt(("noise-kind", salt))
                        .selected_text(noise_kind_name(spec.kind))
                        .show_ui(ui, |ui| {
                            for kind in NOISE_KINDS {
                                changed |= ui
                                    .selectable_value(&mut spec.kind, kind, noise_kind_name(kind))
                                    .changed();
                            }
                        });
                    ui.end_row();

                    ui.label("seed");
                    changed |= ui.add(egui::DragValue::new(&mut spec.seed)).changed();
                    ui.end_row();

                    // A scale is a reciprocal wavelength, so the useful range spans three
                    // decades and a linear speed is unusable at the small end.
                    ui.label("scale");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut spec.scale)
                                .speed(0.0002)
                                .range(0.0..=1.0),
                        )
                        .changed();
                    ui.end_row();

                    ui.label("octaves");
                    changed |= ui
                        .add(egui::DragValue::new(&mut spec.octaves).range(1..=10))
                        .changed();
                    ui.end_row();

                    ui.label("strike");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut spec.transform.strike_degrees)
                                .speed(1.0)
                                .range(-180.0..=180.0),
                        )
                        .changed();
                    ui.end_row();

                    ui.label("aspect");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut spec.transform.aspect)
                                .speed(0.05)
                                .range(1.0..=32.0),
                        )
                        .changed();
                    ui.end_row();
                });

            if let Some(warp) = &mut spec.warp {
                ui.horizontal(|ui| {
                    ui.label("warp");
                    changed |= ui.add(egui::DragValue::new(&mut warp.amplitude)).changed();
                    changed |= ui
                        .add(egui::DragValue::new(&mut warp.scale).speed(0.0002))
                        .changed();
                    changed |= ui
                        .add(egui::DragValue::new(&mut warp.octaves).range(1..=6))
                        .changed();
                });
            }
        }

        LayerOp::Slope {
            of,
            sample_tiles,
            mode,
        } => {
            changed |= field_combo(ui, ("slope-of", salt), of, names);
            ui.horizontal(|ui| {
                ui.label("sample tiles");
                changed |= ui
                    .add(
                        egui::DragValue::new(sample_tiles)
                            .speed(0.25)
                            .range(0.5..=64.0),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("mode");
                egui::ComboBox::from_id_salt(("slope-mode", salt))
                    .selected_text(slope_mode_name(*mode))
                    .show_ui(ui, |ui| {
                        for candidate in SLOPE_MODES {
                            changed |= ui
                                .selectable_value(mode, candidate, slope_mode_name(candidate))
                                .changed();
                        }
                    });
            });
        }

        LayerOp::FieldRef(id) => {
            changed |= field_combo(ui, ("fieldref", salt), id, names);
        }

        LayerOp::Regions { spec, output } => {
            let current = region_output_name(output);
            ui.horizontal(|ui| {
                ui.label("output");
                egui::ComboBox::from_id_salt(("region-output", salt))
                    .selected_text(&current)
                    .show_ui(ui, |ui| {
                        // The two categorical outputs sit under the columns because they
                        // are read differently rather than because they are a different
                        // kind of thing to choose.
                        for name in spec
                            .columns
                            .iter()
                            .cloned()
                            .chain(["region_id".to_owned(), "cover_class".to_owned()])
                        {
                            if ui.selectable_label(name == current, &name).clicked()
                                && name != current
                            {
                                *output = parse_region_output(&name);
                                changed = true;
                            }
                        }
                    });
            });

            egui::Grid::new(("region-lattice", salt))
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("seed");
                    changed |= ui.add(egui::DragValue::new(&mut spec.seed)).changed();
                    ui.end_row();
                    ui.label("cell tiles");
                    changed |= ui
                        .add(egui::DragValue::new(&mut spec.cell_tiles).range(8..=4096))
                        .changed();
                    ui.end_row();
                    ui.label("blend tiles");
                    changed |= ui
                        .add(egui::DragValue::new(&mut spec.blend_tiles).range(0..=1024))
                        .changed();
                    ui.end_row();
                });

            let columns = spec.columns.clone();
            egui::Grid::new(("region-table", salt))
                .num_columns(columns.len() + 1)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("weight").small().weak());
                    for column in &columns {
                        ui.label(egui::RichText::new(column).small().weak());
                    }
                    ui.end_row();

                    for region in &mut spec.regions {
                        changed |= ui.add(egui::DragValue::new(&mut region.weight)).changed();
                        for value in &mut region.values {
                            changed |= drag(ui, value);
                        }
                        ui.end_row();
                    }
                });
        }

        LayerOp::Paint(raster) => {
            ui.label(if raster.is_empty() {
                "unpainted".to_owned()
            } else {
                format!("{}x{} painted", raster.width(), raster.height())
            });
        }

        LayerOp::External(raster) => {
            ui.label(format!("{}x{} raster", raster.width(), raster.height()));
        }
    }
    changed
}

fn field_combo(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash + std::fmt::Debug,
    id: &mut FieldId,
    names: &[String],
) -> bool {
    let mut changed = false;
    let current = id.to_string();
    egui::ComboBox::from_id_salt(salt)
        .selected_text(&current)
        .show_ui(ui, |ui| {
            for name in names {
                if ui.selectable_label(*name == current, name).clicked() && *name != current {
                    *id = name.as_str().into();
                    changed = true;
                }
            }
        });
    changed
}

fn drag(ui: &mut egui::Ui, value: &mut f32) -> bool {
    ui.add(egui::DragValue::new(value).speed(0.01)).changed()
}

/// What the add button makes, which has to be something that bakes on its own — a field
/// reference to nothing would put the document in an error state on the frame it was
/// added, and the panel would look like it had refused.
fn default_op(kind: &str, names: &[String]) -> LayerOp {
    match kind {
        "constant" => LayerOp::Constant(0.5),
        "fieldref" => LayerOp::FieldRef(first_field(names)),
        "slope" => LayerOp::Slope {
            of: first_field(names),
            sample_tiles: 4.0,
            mode: SlopeMode::default(),
        },
        // Empty, and sized by the first stroke — see `edit::parse_op`, which is the same
        // decision reached from the other end.
        "paint" => LayerOp::Paint(Raster::default()),
        _ => LayerOp::Noise(NoiseSpec::new(1, NoiseKind::Fbm, 0.02)),
    }
}

fn first_field(names: &[String]) -> FieldId {
    names
        .first()
        .map(|name| FieldId::from(name.as_str()))
        .unwrap_or_else(|| FieldId::from("height"))
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
                    let result = document.start_new(
                        UVec2::new(dialog.width, dialog.height),
                        dialog.seed,
                        dialog.preset,
                    );
                    report(&mut document, result);
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

/// A refusal is synchronous, where an error is something a job hands back — but both are
/// the answer to "why did that button do nothing", so both go where the toolbar can show
/// them. Logging alone was not enough: `observe log` reads it and a person does not.
fn report(document: &mut Document, result: Result<(), String>) {
    if let Err(error) = result {
        warn!("{error}");
        document.refuse(error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect this guards drew the whole world into four pixels: the rectangle left
    /// over after the panels is what the camera is given, and a degenerate one is not a
    /// visible mistake in the panel — it is a missing picture everywhere else.
    #[test]
    fn the_panels_leave_the_world_a_rectangle_it_can_be_drawn_in() {
        let context = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(592.0, 720.0));
        let mut free = egui::Rect::NOTHING;

        let input = egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        };
        // The pass builds its own root `Ui` over the viewport and ignores the one egui
        // hands it, which is exactly what `panels` does inside bevy_egui's pass.
        let _ = context.run_ui(input, |_| {
            let mut viewport = egui::Ui::new(
                context.clone(),
                "viewport".into(),
                egui::UiBuilder::new()
                    .layer_id(egui::LayerId::background())
                    .max_rect(context.viewport_rect()),
            );
            egui::Panel::top("toolbar").show(&mut viewport, |ui| {
                ui.label("toolbar");
            });
            egui::Panel::right("layers")
                .default_size(PANEL_WIDTH)
                .show(&mut viewport, |ui| {
                    ui.label("layers");
                });
            free = free_rect(&mut viewport);
        });

        assert!(
            free.width() > screen.width() * 0.25,
            "the world was left {} points of {}",
            free.width(),
            screen.width()
        );
        assert!(
            free.height() > screen.height() * 0.5,
            "the world was left {} points of {}",
            free.height(),
            screen.height()
        );
        // And it has to be the space the panels are not in, rather than merely a big
        // rectangle: the toolbar is above it and the layer stack to the right of it.
        assert!(free.min.y > 0.0, "the world overlaps the toolbar");
        assert!(free.max.x < screen.max.x, "the world overlaps the panel");
    }
}
