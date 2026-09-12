//! A named graph of nodes, and the raster the graph bakes onto.

use glam::UVec2;
use serde::{Deserialize, Serialize};

use watershed::field::{FieldId, FieldRole};
use watershed::raster::{Raster, raster_coord, resolution};

use crate::terrain::graph::{FieldGraph, NodeOp};
use crate::terrain::regions::RegionOutput;

/// A named graph of nodes together with everything needed to evaluate it onto its
/// own raster, plus that raster once it has been baked.
///
/// The baked raster is not serialized: it is derived from the graph and is
/// re-obtained by baking, so a loaded document starts with every field empty and
/// sampling as `0.0`. Equality does compare it, so two fields differing only in
/// bake state are not equal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Field {
    /// The name this field is referenced by. Assigning it does not rewrite the
    /// graphs of other fields that reference the old name, nor the water spec that
    /// names it; renaming a field in a document is
    /// [`Edit::RenameField`](crate::edit::Edit::RenameField), which rewrites both in
    /// one change.
    pub id: FieldId,
    /// What the bake may do with the field. See [`FieldRole`] for the constraints
    /// a document carrying this has to satisfy.
    #[serde(default)]
    pub role: FieldRole,
    /// Resolution, as the [`raster`](watershed::raster) shift: one texel per cell at 0,
    /// per `2^shift` cells above that.
    pub shift: u8,
    /// The interval baked values are clamped into. Accepted in either order; read
    /// it through [`Field::bounds`] rather than directly.
    pub range: (f32, f32),
    /// Carried through the document and never read by this crate. It is for
    /// whatever consumes a baked terrain to decide what an exported field means.
    #[serde(default)]
    pub export: bool,
    /// Whether the map lights this field's colour ramp by the field's own normal.
    ///
    /// Nothing a bake reads looks at this or at [`Field::light_azimuth`], so
    /// toggling either never makes the bake stale.
    #[serde(default)]
    pub hillshade: bool,
    /// The compass bearing the hillshade light comes from, in degrees: 0 is north
    /// and 90 is east. Read only while [`Field::hillshade`] is set. Defaults to the
    /// northwest.
    #[serde(default = "default_light_azimuth")]
    pub light_azimuth: f32,
    /// Whether the map draws an iso-line at every multiple of
    /// [`Field::contour_interval`] over this field's colour ramp.
    ///
    /// Nothing a bake reads looks at this or at [`Field::contour_interval`], so
    /// toggling either never makes the bake stale.
    #[serde(default)]
    pub contours: bool,
    /// The spacing between iso-lines, in the field's own units. Read only while
    /// [`Field::contours`] is set. Defaults to a tenth, which divides the default
    /// range into ten bands.
    #[serde(default = "default_contour_interval")]
    pub contour_interval: f32,
    /// What the field's value is built out of. Evaluation order is derived from the
    /// edges, not stored.
    pub graph: FieldGraph,
    #[serde(skip)]
    baked: Raster<f32>,
}

const DEFAULT_LIGHT_AZIMUTH: f32 = 315.0;
const DEFAULT_CONTOUR_INTERVAL: f32 = 0.1;

fn default_light_azimuth() -> f32 {
    DEFAULT_LIGHT_AZIMUTH
}

fn default_contour_interval() -> f32 {
    DEFAULT_CONTOUR_INTERVAL
}

impl Field {
    /// A field named `id` with an empty graph: role `Custom`, shift 0, range
    /// `0.0..=1.0`, not exported, not hillshaded, lit from the northwest, without
    /// contours at a tenth-unit interval, and unbaked — so it samples as `0.0`
    /// until it is baked.
    pub fn new(id: impl Into<FieldId>) -> Self {
        Self {
            id: id.into(),
            role: FieldRole::Custom,
            shift: 0,
            range: (0.0, 1.0),
            export: false,
            hillshade: false,
            light_azimuth: DEFAULT_LIGHT_AZIMUTH,
            contours: false,
            contour_interval: DEFAULT_CONTOUR_INTERVAL,
            graph: FieldGraph::new(),
            baked: Raster::default(),
        }
    }

    /// Sets the role. Does not check the document-wide constraints on
    /// [`FieldRole`]; a conflict surfaces at plan time.
    pub fn with_role(mut self, role: FieldRole) -> Self {
        self.role = role;
        self
    }

    /// Sets [`Field::export`].
    pub fn with_export(mut self, export: bool) -> Self {
        self.export = export;
        self
    }

    /// Sets the resolution shift. A shift on the `Height` field is rejected at plan
    /// time, not here.
    pub fn with_shift(mut self, shift: u8) -> Self {
        self.shift = shift;
        self
    }

    /// Sets the clamp interval. Either order is accepted — see [`Field::bounds`].
    pub fn with_range(mut self, range: (f32, f32)) -> Self {
        self.range = range;
        self
    }

    /// Adds one node of `op` and reads the field from it.
    ///
    /// The one-node graph a field starts as, and what most of the test suite wants:
    /// a field that is exactly one op.
    pub fn with_op(mut self, op: NodeOp) -> Self {
        self.graph.add_node(op, [0.0, 0.0]);
        self
    }

    /// A field whose value is the sum of `ops`, in the order given.
    ///
    /// What a stack of layers blending onto zero came to, and so what most of the
    /// suite wants: the ops are added left to right and the field reads the total.
    pub fn with_sum(mut self, ops: impl IntoIterator<Item = NodeOp>) -> Self {
        let mut under: Option<crate::terrain::graph::NodeId> = None;
        for op in ops {
            let id = self.graph.node_with(op, &[]);
            under = Some(match under {
                None => id,
                Some(under) => self.graph.node_with(
                    NodeOp::Binary(crate::terrain::graph::Binary::Add),
                    &[under, id],
                ),
            });
        }
        if let Some(output) = under {
            self.graph
                .set_output(Some(output))
                .expect("the node was just added to this graph");
        }
        self
    }

    /// Replaces the whole graph.
    pub fn with_graph(mut self, graph: FieldGraph) -> Self {
        self.graph = graph;
        self
    }

    /// A copy of everything a person authored and nothing that was derived from it:
    /// the graph with every shader node's values dropped, and no bake. A paint or
    /// external raster is authored data and comes along.
    ///
    /// What a history snapshot is made of — the bake and the shader values are
    /// re-obtained by baking and dispatching, so a copy that carried them would cost
    /// the size of the document per edit.
    pub fn authored(&self) -> Self {
        let mut graph = self.graph.clone();
        for node in &mut graph.nodes {
            if let NodeOp::Shader(shader) = &mut node.op {
                shader.clear();
            }
        }
        Self {
            id: self.id.clone(),
            role: self.role,
            shift: self.shift,
            range: self.range,
            export: self.export,
            hillshade: self.hillshade,
            light_azimuth: self.light_azimuth,
            contours: self.contours,
            contour_interval: self.contour_interval,
            graph,
            baked: Raster::default(),
        }
    }

    /// [`Field::range`] as `(low, high)`, swapped if it was stored backwards. A
    /// backwards range is not an error anywhere; this is the only correct way to
    /// read it.
    pub fn bounds(&self) -> (f32, f32) {
        if self.range.0 <= self.range.1 {
            self.range
        } else {
            (self.range.1, self.range.0)
        }
    }

    /// Texel dimensions this field bakes to in a `size`-cell document. Never zero
    /// on either axis.
    pub fn resolution(&self, size: UVec2) -> UVec2 {
        resolution(size, self.shift)
    }

    /// The values from the last bake. Empty for a field that has never been baked,
    /// was released, or came from a loaded document — read it through
    /// [`Field::sample`] unless the texel grid itself is what you want.
    pub fn baked(&self) -> &Raster<f32> {
        &self.baked
    }

    /// The baked raster, writable in place. Its dimensions are the bake's
    /// invariant, so a caller replacing texels must not change its length.
    pub(crate) fn baked_mut(&mut self) -> &mut Raster<f32> {
        &mut self.baked
    }

    /// Moves the baked raster out, leaving the field unbaked and sampling as
    /// `0.0`. Pair with [`Field::put_baked`] to borrow a raster across a bake that
    /// needs the rest of the document mutably.
    pub(crate) fn take_baked(&mut self) -> Raster<f32> {
        std::mem::take(&mut self.baked)
    }

    /// Installs `raster` as the bake result, dropping whatever was there. Nothing
    /// checks it against [`Field::resolution`].
    pub(crate) fn put_baked(&mut self, raster: Raster<f32>) {
        self.baked = raster;
    }

    /// Whether this field's values name a class rather than measure a quantity,
    /// which is what decides how [`Field::sample`] interpolates.
    ///
    /// Derived from the graph, not declared: true when the node the field's value
    /// actually comes from emits region ids or cover classes.
    ///
    /// It is the *effective* output that decides — a bypassed node at the output is
    /// followed to what it passes through — so a `Regions` node under an arithmetic
    /// node no longer switches the whole field to nearest sampling, where a `Regions`
    /// layer anywhere in a stack once did.
    pub fn is_categorical(&self) -> bool {
        let Some(id) = self.graph.effective_output() else {
            return false;
        };
        matches!(
            self.graph.node(id).map(|node| &node.op),
            Some(NodeOp::Regions {
                output: RegionOutput::RegionId | RegionOutput::CoverClass,
                ..
            })
        )
    }

    /// The baked value at a position in document cells, where a cell centre is at
    /// `x + 0.5`. Reads `0.0` for an unbaked field, and clamps rather than failing
    /// outside the document.
    ///
    /// A coarse field is interpolated between its texels, so it reads as a smooth
    /// surface and not as blocks. A [categorical](Field::is_categorical) field is
    /// read to the nearest texel instead: halfway between two region ids is not a
    /// third region, so it must be one of the two.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = raster_coord(x, self.shift);
        let v = raster_coord(y, self.shift);
        if self.is_categorical() {
            self.baked.sample_nearest(u, v)
        } else {
            self.baked.sample_bilinear(u, v)
        }
    }

    /// The fields this one reads, through the nodes reachable from its output only;
    /// bypassing or unwiring a node removes its dependencies.
    ///
    /// This is what bake ordering and cycle detection run on, so a cycle that exists
    /// only through an unreachable node is not a cycle and the document plans.
    /// Duplicates are not removed and the order is evaluation order.
    pub fn dependencies(&self) -> impl Iterator<Item = &FieldId> {
        self.graph.dependencies().into_iter()
    }

    /// Every field this one's graph names in a `FieldRef`, whether or not the node is
    /// wired to anything, with duplicates kept.
    ///
    /// Wider than [`Field::dependencies`], which reports only what the output reaches:
    /// an unconnected reference reads nothing yet, but it is a declared read, and the
    /// editor refuses one that could not be wired up later. A bypassed node is left out
    /// of both, because bypass is how a reference is turned off.
    pub fn declared_reads(&self) -> impl Iterator<Item = &FieldId> {
        self.graph
            .nodes
            .iter()
            .filter(|node| !node.bypassed)
            .filter_map(|node| node.op.dependency())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::graph::{FieldGraph, NodeOp, Remap};

    // Shift 0 is what the `Height` field is pinned to, so a document's cell grid and
    // its height raster have to be the same grid.
    #[test]
    fn a_field_at_shift_zero_holds_one_texel_per_cell() {
        let field = Field::new("height");
        assert_eq!(field.resolution(UVec2::new(64, 32)), UVec2::new(64, 32));
    }

    // The point of a shift is the allocation it saves; this pins that the saving is
    // quadratic in the shift and not linear.
    #[test]
    fn a_coarse_field_holds_one_texel_per_block() {
        let field = Field::new("moisture").with_shift(4);
        assert_eq!(
            field.resolution(UVec2::new(4096, 4096)),
            UVec2::new(256, 256)
        );
    }

    // Nothing rejects a backwards range, so `bounds` is the only thing standing
    // between one and a clamp that empties the field.
    #[test]
    fn a_backwards_range_is_read_in_the_order_a_clamp_needs() {
        let field = Field::new("height").with_range((1.0, -1.0));
        assert_eq!(field.bounds(), (-1.0, 1.0));
    }

    // Pins both halves of what bake ordering is computed from: every reference the
    // output reaches counts, and one it cannot reach counts for nothing.
    #[test]
    fn a_field_reports_the_dependencies_its_output_reaches_and_no_others() {
        let mut graph = FieldGraph::new();
        let base = graph.node_with(NodeOp::Constant(0.5), &[]);
        let relief = graph.node_with(NodeOp::FieldRef(FieldId::from("relief")), &[]);
        let ridge = graph.node_with(NodeOp::FieldRef(FieldId::from("ridge")), &[]);
        let weight = graph.node_with(NodeOp::Remap(Remap::IDENTITY), &[ridge]);
        let mixed = graph.node_with(NodeOp::Lerp, &[base, relief, weight]);
        graph.node_with(NodeOp::FieldRef(FieldId::from("hidden")), &[]);
        graph.set_output(Some(mixed)).unwrap();

        let field = Field::new("height").with_graph(graph);
        let deps: Vec<_> = field.dependencies().map(|id| id.as_str()).collect();
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&"relief") && deps.contains(&"ridge"));
        assert!(!deps.contains(&"hidden"));
    }

    // A history snapshot is `authored()`, so a display property missing from it is a
    // property undo silently skips.
    #[test]
    fn a_new_field_is_unlit_from_the_northwest_and_carries_both_into_a_snapshot() {
        let mut field = Field::new("height");
        assert!(!field.hillshade);
        assert_eq!(field.light_azimuth, 315.0);

        field.hillshade = true;
        field.light_azimuth = 90.0;
        let authored = field.authored();
        assert!(authored.hillshade);
        assert_eq!(authored.light_azimuth, 90.0);
    }

    // Same reason as the hillshade pair above: a contour property dropped by
    // `authored()` is one undo would silently skip.
    #[test]
    fn a_new_field_has_no_contours_at_a_tenth_and_carries_both_into_a_snapshot() {
        let mut field = Field::new("height");
        assert!(!field.contours);
        assert_eq!(field.contour_interval, 0.1);

        field.contours = true;
        field.contour_interval = 0.25;
        let authored = field.authored();
        assert!(authored.contours);
        assert_eq!(authored.contour_interval, 0.25);
    }

    // Every loaded document is in this state until it is baked, so sampling one has
    // to be a defined read rather than a panic on an empty raster.
    #[test]
    fn an_unbaked_field_samples_as_zero() {
        let field = Field::new("height");
        assert_eq!(field.sample(12.5, 3.5), 0.0);
    }
}
