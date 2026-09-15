//! What the document can go back to: what one change leaves behind, and the two stacks
//! that turn those into undo and redo.
//!
//! An entry does not cost the size of the document. It is kept as the authored state of
//! every layer and nothing derived from it, so it costs the settings and parameter
//! values.

use crate::terrain::{Layer, TerrainSpec, WaterSpec};

/// How many changes can be undone. Recording past this drops the oldest.
pub const HISTORY_DEPTH: usize = 100;
/// The authored state of every layer on the far side of one change, the water spec
/// there, the layer that was on screen, and whether crossing that change reaches the
/// bake.
///
/// Which layer was on screen is part of what one change leaves behind because a
/// change may move the view, so going back across that change has to put the earlier
/// one back, in the same step.
pub struct Snapshot {
    layers: Vec<Layer>,
    water_spec: Option<WaterSpec>,
    active: String,
    reaches_bake: bool,
}

impl Snapshot {
    /// Copies every layer's authored state as it stands: no bake and no shader values.
    ///
    /// `active` is the layer on screen at the moment of the copy, which crossing the
    /// change puts back.
    pub fn take(terrain: &TerrainSpec, reaches_bake: bool, active: &str) -> Self {
        Self {
            layers: terrain.layers.iter().map(Layer::authored).collect(),
            water_spec: terrain.water_spec.clone(),
            active: active.to_owned(),
            reaches_bake,
        }
    }
    /// Puts each held layer's settings and parameter values back onto the live layer
    /// of the same name, and the water spec back into `terrain`.
    ///
    /// Never adds or removes a layer: a live layer the snapshot does not hold is left
    /// as it is, and a held layer no live layer matches is dropped. Everything the
    /// history does not own — bakes, shader values, the layers a file reads — stays
    /// with the live layer.
    pub fn restore(self, terrain: &mut TerrainSpec) {
        for held in self.layers {
            let Some(live) = terrain.layer_mut(held.id.as_str()) else {
                continue;
            };
            live.role = held.role;
            live.shift = held.shift;
            live.range = held.range;
            live.export = held.export;
            live.hillshade = held.hillshade;
            live.light_azimuth = held.light_azimuth;
            live.contours = held.contours;
            live.contour_interval = held.contour_interval;
            live.shader.params = held.shader.params;
        }
        terrain.water_spec = self.water_spec;
    }
}

/// What crossing one change put back, and what the document has to do about it.
pub struct Restored {
    /// Whether crossing this change reaches the bake.
    pub reaches_bake: bool,
    /// The layer that was on screen on the far side of the change. Equal to the
    /// current one for every change that did not move the view.
    pub active: String,
}

/// How far the history reaches in each direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryDepth {
    /// Changes that can be undone.
    pub undo: usize,
    /// Changes that can be redone.
    pub redo: usize,
}

/// The changes behind the document and the ones ahead of it.
///
/// Every entry sits on the far side of exactly one change, and undoing or redoing swaps
/// it with the document — so the entry that comes back describes the same change from
/// the other side, and the two stacks together always hold one entry per change made.
#[derive(Default)]
pub struct History {
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
}

impl History {
    /// Records a change that has just been made, `before` being the snapshot taken
    /// ahead of it. Forgets everything that could have been redone, and the oldest
    /// entry past [`HISTORY_DEPTH`].
    pub fn record(&mut self, before: Snapshot) {
        self.undo.push(before);
        self.redo.clear();
        if self.undo.len() > HISTORY_DEPTH {
            self.undo.remove(0);
        }
    }

    /// Puts the document back to before the last change and says what came back, or
    /// `None` with nothing to undo.
    ///
    /// `active` is the layer on screen now, recorded so that redoing the change puts
    /// it back.
    pub fn undo(&mut self, terrain: &mut TerrainSpec, active: &str) -> Option<Restored> {
        let entry = self.undo.pop()?;
        let (restored, back) = swap(entry, terrain, active);
        self.redo.push(back);
        Some(restored)
    }

    /// Replays the last change undone and says what came back, or `None` with nothing to
    /// redo.
    ///
    /// `active` is the layer on screen now, recorded so that undoing the change again
    /// puts it back.
    pub fn redo(&mut self, terrain: &mut TerrainSpec, active: &str) -> Option<Restored> {
        let entry = self.redo.pop()?;
        let (restored, back) = swap(entry, terrain, active);
        self.undo.push(back);
        Some(restored)
    }

    /// Forgets everything, for a document that has just been replaced.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    /// How far the history reaches in each direction.
    pub fn depth(&self) -> HistoryDepth {
        HistoryDepth {
            undo: self.undo.len(),
            redo: self.redo.len(),
        }
    }
}

fn swap(entry: Snapshot, terrain: &mut TerrainSpec, active: &str) -> (Restored, Snapshot) {
    let restored = Restored {
        reaches_bake: entry.reaches_bake,
        active: entry.active.clone(),
    };
    let now = Snapshot::take(terrain, entry.reaches_bake, active);
    entry.restore(terrain);
    (restored, now)
}
#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::UVec2;

    fn held(size: u32) -> TerrainSpec {
        let mut terrain =
            TerrainSpec::new(UVec2::splat(size)).with_layer(Layer::new("height").held(0.5));
        terrain.bake_in_place().unwrap();
        terrain
    }

    // The whole reason a snapshot is affordable: neither a bake nor a shader's values is
    // kept, since both are re-obtained by baking.
    #[test]
    fn a_recorded_snapshot_keeps_no_bake_and_no_shader_values() {
        let terrain = held(8);
        let mut history = History::default();
        history.record(Snapshot::take(&terrain, true, "height"));

        let kept = &history.undo[0].layers[0];
        assert!(kept.baked().is_empty(), "a bake was kept");
        assert!(
            kept.shader.values().is_empty(),
            "the shader values were kept"
        );
    }

    // Ctrl+Z covers parameter values, and the picture on screen stays the last one
    // until the re-bake lands rather than going blank.
    #[test]
    fn undoing_a_parameter_change_restores_it_and_keeps_the_bake() {
        let mut terrain = held(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true, "height");
        terrain
            .layer_mut("height")
            .unwrap()
            .shader
            .params
            .insert("value".to_owned(), vec![0.75]);
        history.record(before);

        history.undo(&mut terrain, "height").unwrap();
        let layer = terrain.layer("height").unwrap();
        assert_eq!(layer.shader.params.get("value"), Some(&vec![0.5]));
        assert!(!layer.baked().is_empty());
        assert!(!layer.shader.values().is_empty());
    }

    // Adding a layer is a file operation outside the history, so an undo across an
    // earlier change must not take a layer out whose file is still on disk.
    #[test]
    fn a_layer_added_after_a_snapshot_survives_an_undo() {
        let mut terrain = held(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true, "height");
        terrain.layer_mut("height").unwrap().range = (0.0, 2.0);
        history.record(before);
        terrain.layers.push(Layer::new("temperature"));

        history.undo(&mut terrain, "height").unwrap();
        assert!(terrain.layer("temperature").is_some());
        assert_eq!(terrain.layer("height").unwrap().range, (0.0, 1.0));
    }

    // Removing a layer deleted its file, so an undo must not bring back a layer whose
    // file is gone.
    #[test]
    fn a_layer_removed_after_a_snapshot_is_not_brought_back() {
        let mut terrain = held(8).with_layer(Layer::new("temperature"));
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true, "height");
        terrain.layer_mut("height").unwrap().range = (0.0, 2.0);
        history.record(before);
        terrain
            .layers
            .retain(|layer| layer.id.as_str() != "temperature");

        history.undo(&mut terrain, "height").unwrap();
        assert!(terrain.layer("temperature").is_none());
    }

    // The two stacks are one sequence of changes: a new change forgets what could have
    // been redone, and the depth is bounded.
    #[test]
    fn a_new_change_forgets_the_redo_stack_and_the_depth_is_bounded() {
        let mut terrain = held(8);
        let mut history = History::default();
        for step in 0..(HISTORY_DEPTH + 5) {
            let before = Snapshot::take(&terrain, true, "height");
            terrain.layer_mut("height").unwrap().range = (0.0, step as f32);
            history.record(before);
        }
        assert_eq!(history.depth().undo, HISTORY_DEPTH);

        history.undo(&mut terrain, "height").unwrap();
        history.undo(&mut terrain, "height").unwrap();
        assert_eq!(history.depth().redo, 2);
        let before = Snapshot::take(&terrain, true, "height");
        terrain.layer_mut("height").unwrap().range = (0.0, 1.0);
        history.record(before);
        assert_eq!(history.depth().redo, 0);
        assert!(history.redo(&mut terrain, "height").is_none());
    }
}
