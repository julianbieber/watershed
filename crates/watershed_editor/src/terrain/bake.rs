//! Turning an authored document into evaluated grids: the order the fields have to
//! be visited in, the ways of driving that order, and what the result is handed
//! back as.

use std::collections::HashMap;

use glam::{UVec2, Vec2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use watershed::channel::{ChannelError, ChannelMeta, plan_layers};
use watershed::field::{FieldId, FieldRole};

use crate::gpu::{DispatchGlobals, ShaderRuntime, dispatch_key};
use crate::terrain::field::Field;
use crate::terrain::water::{WaterError, WaterSpec, WaterState};
use watershed::meta::WaterInfo;
use watershed::raster::{Raster, raster_coord, resolution, texel_center};
use watershed::terrain::{FieldInfo, LayerTexels, Terrain, TerrainLayer};

/// Everything structurally wrong with a document, and the one fault that is not.
///
/// Planning is what these are for: a document that plans can be baked, and a bake
/// that has started cannot fail any of these ways —
/// [`PlanError::ShaderDispatch`] excepted, which is raised while a field is being
/// evaluated, because the raster a sampling shader reads is produced inside the bake.
#[derive(Debug, Error)]
pub enum PlanError {
    /// The document has no cells.
    #[error("terrain size has a zero component: {0} by {1}")]
    ZeroSize(u32, u32),
    /// Two fields carry the same id, so a reference to it is ambiguous.
    #[error("two fields share the id `{0}`")]
    DuplicateField(String),
    /// A shader's `@layer` names a field the document does not carry. Names both,
    /// because the read is in the reader and the mistake may be in either.
    #[error("field `{referenced}`, read by `{reader}`, is not in the document")]
    UnknownField {
        /// The name that could not be resolved.
        referenced: String,
        /// The field whose shader names it.
        reader: String,
    },
    /// The fields cannot be ordered. Carries the cycle as a `->` chain of names.
    #[error("fields depend on each other in a cycle: {0}")]
    Cycle(String),
    /// Two fields claim `Height` or two claim `Moisture`, so a role lookup would be
    /// ambiguous.
    #[error("two fields claim the role `{0}`")]
    DuplicateRole(FieldRole),
    /// The document declares water and no field holds `Height` to solve it over.
    #[error("water is declared and no field holds the role `height`")]
    MissingHeightField,
    /// The `Height` field is coarser than the document; the water solve reads one
    /// texel per cell and will not resample.
    #[error("field `{0}` holds the role `height` at shift {1}")]
    CoarseHeight(String, u8),
    /// A field's shader could not be dispatched. The one fault here that is raised
    /// during evaluation rather than before it.
    #[error("the shader of field `{field}` did not run: {reason}")]
    ShaderDispatch {
        /// The field whose shader it is.
        field: String,
        /// What the dispatch answered.
        reason: String,
    },
}

/// What can go wrong once a plan exists.
///
/// Every structural fault was already caught by [`TerrainSpec::plan_bake`], so what
/// is left is a caller driving the plan out of step, or the water solve — the one
/// step whose inputs are not fully settled at plan time.
#[derive(Debug, Error)]
pub enum BakeError {
    /// [`Bake::advance`] was called on a finished plan, or a water step was reached
    /// on a document whose water spec has since been removed.
    #[error("the plan has no step left to advance")]
    NoStepRemaining,
    /// [`Bake::finish`] was called before every step had run, carrying how many are
    /// left. Finishing consumes the bake either way, so the work already done is lost
    /// with it.
    #[error("{0} step(s) of the plan have not run")]
    StepsRemaining(u32),
    /// The water step. See [`WaterError`].
    #[error(transparent)]
    WaterSolve(#[from] WaterError),
    /// A field step. Reachable if the document was edited between planning and
    /// baking. See [`PlanError`].
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// A field's values could not be stored in the channel they were destined for.
    /// See [`ChannelError`].
    #[error(transparent)]
    Channel(#[from] ChannelError),
}

/// An authored document: an extent, the fields in it, and optionally a description
/// of the water over them.
///
/// This is the editable side of a terrain and the thing that is saved. Everything
/// derived — a field's baked raster, the solved water — is held here too but is not
/// part of the serialized form; the recipe that re-derives it is, which is why the
/// water spec survives a save where the water itself does not.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TerrainSpec {
    /// The extent in cells. Every field covers all of it, whatever its own shift.
    pub size: UVec2,
    /// The fields, in declaration order. Not bake order — see
    /// [`TerrainSpec::bake_order`] — but the order a consumer reads them back in.
    pub fields: Vec<Field>,
    /// What water this document wants, if any. Serialized, so a document reloaded
    /// without its solved water can solve it again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water_spec: Option<WaterSpec>,
    /// The seed every shader of this document is told about. Serialized, so a
    /// reloaded document dispatches to the same values.
    #[serde(default)]
    pub seed: u32,
    #[serde(skip)]
    pub(crate) water: Option<WaterState>,
    #[serde(skip)]
    pub(crate) runtime: ShaderRuntime,
}

impl TerrainSpec {
    /// An empty document of that extent — no fields, no water.
    pub fn new(size: UVec2) -> Self {
        Self {
            size,
            fields: Vec::new(),
            water_spec: None,
            seed: 0,
            water: None,
            runtime: ShaderRuntime::default(),
        }
    }

    /// Installs what a field's shader is dispatched through. Derived state: it is not
    /// part of what the document is, it is not saved, and installing it is not an
    /// edit.
    pub fn set_shader_runtime(&mut self, runtime: ShaderRuntime) {
        self.runtime = runtime;
    }

    /// What a field's shader is currently dispatched through. A default runtime holds
    /// no device, and every field under it reads the values its shader already holds.
    pub fn shader_runtime(&self) -> &ShaderRuntime {
        &self.runtime
    }

    /// Every field that cannot bake, with the reason, in declaration order.
    ///
    /// A field reading a name the document does not carry, a field reading one of
    /// those, and every field of a cycle between fields that could otherwise bake. The
    /// first two are left unbaked by [`TerrainSpec::bake_in_place`] while the rest of
    /// document bakes; a cycle still fails the bake, and its fields' reason is
    /// `cycle: ` followed by the chain. Derived from the document each
    /// call, never stored. A document with two fields of one id answers no faults.
    pub fn field_faults(&self) -> Vec<(FieldId, String)> {
        let Ok(index_of) = self.index_fields() else {
            return Vec::new();
        };
        let mut faults = self.unreadable(&index_of);
        let blocked: Vec<bool> = faults.iter().map(Option::is_some).collect();
        let dependencies = self.resolve_readable(&index_of, &blocked);
        if let Err(PlanError::Cycle(chain)) = topological_order(&dependencies, &self.fields) {
            for name in chain.split(" -> ") {
                if let Some(&at) = index_of.get(name)
                    && faults[at].is_none()
                {
                    faults[at] = Some(format!("cycle: {chain}"));
                }
            }
        }
        self.fields
            .iter()
            .zip(faults)
            .filter_map(|(field, fault)| Some((field.id.clone(), fault?)))
            .collect()
    }

    /// Appends a field. Order here is declaration order, not evaluation order: a
    /// field may reference one added after it.
    pub fn with_field(mut self, field: Field) -> Self {
        self.fields.push(field);
        self
    }

    /// The field of that exact name, or `None`. Case-sensitive.
    pub fn field(&self, id: &str) -> Option<&Field> {
        self.fields.iter().find(|field| field.id.as_str() == id)
    }

    /// As [`TerrainSpec::field`], mutable. Editing through this can invalidate the
    /// bakes and the water; nothing here notices.
    pub fn field_mut(&mut self, id: &str) -> Option<&mut Field> {
        self.fields.iter_mut().find(|field| field.id.as_str() == id)
    }

    /// A field's baked value at a position in document cells, where a cell centre is
    /// at `x + 0.5`.
    ///
    /// `None` means the document has no such field — a distinction worth keeping,
    /// because a field that exists but is unbaked or released samples as `Some(0.0)`
    /// and a caller that conflated the two could not tell a typo from a released
    /// field.
    pub fn sample(&self, id: &str, x: f32, y: f32) -> Option<f32> {
        self.field(id).map(|field| field.sample(x, y))
    }

    /// Bakes every field over the whole document, in dependency order, leaving the
    /// results on the fields. Does not solve water.
    ///
    /// Fields are reallocated to their declared resolutions first, so this also
    /// fixes a document whose shifts have been edited.
    ///
    /// A field reading a name the document does not carry, and every field reading
    /// one of those, is left unbaked — sampling as `0.0` — while the rest bakes;
    /// [`TerrainSpec::field_faults`] says which and why. A cycle between fields, a
    /// duplicate id, a zero extent and a shader that fails to dispatch still fail the
    /// whole bake.
    pub fn bake_in_place(&mut self) -> Result<(), PlanError> {
        if self.size.x == 0 || self.size.y == 0 {
            return Err(PlanError::ZeroSize(self.size.x, self.size.y));
        }

        let index_of = self.index_fields()?;
        let blocked: Vec<bool> = self
            .unreadable(&index_of)
            .iter()
            .map(Option::is_some)
            .collect();
        let dependencies = self.resolve_readable(&index_of, &blocked);
        let order: Vec<usize> = topological_order(&dependencies, &self.fields)?
            .into_iter()
            .filter(|&index| !blocked[index])
            .collect();
        self.reallocate_rasters(&blocked);

        let mut baked: Vec<Raster<f32>> = self
            .fields
            .iter_mut()
            .map(|field| field.take_baked())
            .collect();

        let mut dispatched: Vec<(usize, (Raster<f32>, u64))> = Vec::new();
        let mut result = Ok(());
        for &target in &order {
            match self.evaluate(target, &index_of, &mut baked) {
                Ok(made) => dispatched.extend(made.map(|made| (target, made))),
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }

        for (field, raster) in self.fields.iter_mut().zip(baked) {
            field.put_baked(raster);
        }
        for (target, (values, key)) in dispatched {
            self.fields[target].shader.put_dispatch(values, key);
        }
        result
    }

    /// The order a bake visits the fields in: every field after the ones it reads.
    ///
    /// This is the whole of what a caller needs to drive a bake a stage at a time, and it
    /// is a *plan* rather than a running bake — nothing is borrowed, so the document
    /// stays readable between stages and the caller decides the pacing. Anything that
    /// would make a bake fail late (a cycle, a missing field, a duplicate id) fails here
    /// instead, before a single texel is written.
    ///
    /// Ids rather than indices, because a caller driving a bake a stage at a time
    /// may edit the document between stages, and an index into `fields` would not
    /// survive a field being added or removed where a name does.
    ///
    /// The order is deterministic: the same document gives the same order on every
    /// machine and in every run.
    pub fn bake_order(&self) -> Result<Vec<FieldId>, PlanError> {
        let index_of = self.index_fields()?;
        let dependencies = self.resolve_dependencies(&index_of)?;
        let order = topological_order(&dependencies, &self.fields)?;
        Ok(order
            .into_iter()
            .map(|index| self.fields[index].id.clone())
            .collect())
    }
    /// Bake one field over the whole document, assuming everything it reads is baked.
    ///
    /// **The assumption is the caller's to keep**, and [`TerrainSpec::bake_order`] is how:
    /// walking that order calls this on a field only after its dependencies. Called out
    /// of order it does not fail — it reads whatever those fields currently hold, which
    /// for an unbaked one is zero. That is the same fallback a document has before any
    /// bake at all, and it is what makes a partially baked document *displayable* rather
    /// than an error state.
    ///
    /// Only the named field is allocated, where [`TerrainSpec::bake_in_place`]
    /// allocates every field. That is what makes [`TerrainSpec::release`] worth
    /// anything: a released field stays released across the stages that follow it, so
    /// a staged bake peaks at its widest live set rather than at the sum of the
    /// document.
    pub fn bake_field(&mut self, id: &str) -> Result<(), PlanError> {
        if self.size.x == 0 || self.size.y == 0 {
            return Err(PlanError::ZeroSize(self.size.x, self.size.y));
        }
        let index_of = self.index_fields()?;
        let reader = FieldId::from(id);
        let target = lookup(&index_of, &reader, &reader)?;
        let wanted = resolution(self.size, self.fields[target].shift);
        if self.fields[target].baked().size() != wanted {
            *self.fields[target].baked_mut() = Raster::new(wanted, 0.0);
        }
        let mut baked: Vec<Raster<f32>> = self
            .fields
            .iter_mut()
            .map(|field| field.take_baked())
            .collect();

        let result = self.evaluate(target, &index_of, &mut baked);

        for (field, raster) in self.fields.iter_mut().zip(baked) {
            field.put_baked(raster);
        }
        if let Some((values, key)) = result? {
            self.fields[target].shader.put_dispatch(values, key);
        }
        Ok(())
    }

    /// Drop a field's baked raster, keeping the shader values that would rebuild it.
    ///
    /// **This is what makes a staged bake affordable rather than merely visible.** A
    /// document's fields do not all have to be resident at once: an intermediate is dead
    /// as soon as everything downstream of it has been baked, and at a whole-world size a
    /// single shift-0 field is tens of megabytes. Releasing as the order advances turns
    /// the peak from the sum of every field into the widest live set.
    ///
    /// A released field samples as zero, exactly as one that has never been baked — so
    /// releasing something still to be read is not an error, it is a wrong answer, and
    /// the caller owns that distinction the same way it owns the bake order.
    pub fn release(&mut self, id: &str) -> bool {
        match self.field_mut(id) {
            Some(field) => {
                *field.baked_mut() = Raster::default();
                true
            }
            None => false,
        }
    }

    /// How many bytes the baked rasters currently hold.
    ///
    /// The bakes only: a field's shader values are not counted. This number is
    /// the part that [`TerrainSpec::release`] moves and is what a staged bake watches.
    pub fn baked_bytes(&self) -> usize {
        self.fields
            .iter()
            .map(|field| {
                let size = field.baked().size();
                size.x as usize * size.y as usize * size_of::<f32>()
            })
            .sum()
    }

    fn evaluate(
        &self,
        target: usize,
        index_of: &HashMap<String, usize>,
        baked: &mut [Raster<f32>],
    ) -> Result<Option<(Raster<f32>, u64)>, PlanError> {
        let field = &self.fields[target];
        let texels = resolution(self.size, field.shift);
        let mut fresh = None;
        if let Some(program) = self.runtime.program(&field.file()) {
            let params = program.layout.pack(&field.shader.params);
            let key = dispatch_key(&program.source, &params, texels);
            let reusable = program.layers.is_empty()
                && field.shader.stamp() == Some(key)
                && field.shader.values().size() == texels;
            if !reusable {
                let layers: Vec<Option<&Raster<f32>>> = program
                    .layers
                    .iter()
                    .map(|read| index_of.get(&read.layer).map(|&at| &baked[at]))
                    .collect();
                let globals = DispatchGlobals {
                    document: self.size,
                    texels,
                    origin: UVec2::ZERO,
                    shift: field.shift as u32,
                    seed: self.runtime.seed().unwrap_or_default(),
                };
                let values = self
                    .runtime
                    .run(program, &params, globals, &layers)
                    .and_then(|values| {
                        Raster::from_vec(texels, values).ok_or_else(|| {
                            "the dispatch produced the wrong number of texels".to_owned()
                        })
                    })
                    .map_err(|reason| PlanError::ShaderDispatch {
                        field: field.id.to_string(),
                        reason,
                    })?;
                fresh = Some((values, key));
            }
        }

        let source = fresh
            .as_ref()
            .map_or(field.shader.values(), |(values, _)| values);
        let (low, high) = field.bounds();
        let shift = field.shift;
        let target_raster = &mut baked[target];
        let size = target_raster.size();
        for j in 0..size.y {
            let v = raster_coord(texel_center(j, shift), shift);
            for i in 0..size.x {
                let u = raster_coord(texel_center(i, shift), shift);
                target_raster.set(i, j, source.sample_bilinear(u, v).clamp(low, high));
            }
        }
        Ok(fresh)
    }

    fn index_fields(&self) -> Result<HashMap<String, usize>, PlanError> {
        let mut index_of = HashMap::with_capacity(self.fields.len());
        for (index, field) in self.fields.iter().enumerate() {
            if index_of.insert(field.id.to_string(), index).is_some() {
                return Err(PlanError::DuplicateField(field.id.to_string()));
            }
        }
        Ok(index_of)
    }

    fn resolve_dependencies(
        &self,
        index_of: &HashMap<String, usize>,
    ) -> Result<Vec<Vec<usize>>, PlanError> {
        let mut dependencies = vec![Vec::new(); self.fields.len()];
        for (index, field) in self.fields.iter().enumerate() {
            for id in field.dependencies() {
                let referenced = lookup(index_of, id, &field.id)?;
                if !dependencies[index].contains(&referenced) {
                    dependencies[index].push(referenced);
                }
            }
        }
        Ok(dependencies)
    }

    fn unreadable(&self, index_of: &HashMap<String, usize>) -> Vec<Option<String>> {
        let mut faults: Vec<Option<String>> = self
            .fields
            .iter()
            .map(|field| {
                field
                    .dependencies()
                    .find_map(|id| lookup(index_of, id, &field.id).err())
                    .map(|error| error.to_string())
            })
            .collect();
        let mut moved = true;
        while moved {
            moved = false;
            for (index, field) in self.fields.iter().enumerate() {
                if faults[index].is_some() {
                    continue;
                }
                let stuck = field.dependencies().find(|id| {
                    index_of
                        .get(id.as_str())
                        .is_some_and(|&at| faults[at].is_some())
                });
                if let Some(name) = stuck {
                    faults[index] = Some(format!(
                        "field `{}` reads `{name}`, which cannot bake",
                        field.id
                    ));
                    moved = true;
                }
            }
        }
        faults
    }

    fn resolve_readable(
        &self,
        index_of: &HashMap<String, usize>,
        blocked: &[bool],
    ) -> Vec<Vec<usize>> {
        let mut dependencies = vec![Vec::new(); self.fields.len()];
        for (index, field) in self.fields.iter().enumerate() {
            if blocked[index] {
                continue;
            }
            for id in field.dependencies() {
                if let Some(&referenced) = index_of.get(id.as_str())
                    && !dependencies[index].contains(&referenced)
                {
                    dependencies[index].push(referenced);
                }
            }
        }
        dependencies
    }

    fn reallocate_rasters(&mut self, blocked: &[bool]) {
        let size = self.size;
        for (field, &blocked) in self.fields.iter_mut().zip(blocked) {
            if blocked {
                *field.baked_mut() = Raster::default();
                continue;
            }
            let wanted = resolution(size, field.shift);
            if field.baked().size() != wanted {
                *field.baked_mut() = Raster::new(wanted, 0.0);
            }
        }
    }
}

fn lookup(
    index_of: &HashMap<String, usize>,
    referenced: &FieldId,
    reader: &FieldId,
) -> Result<usize, PlanError> {
    index_of
        .get(referenced.as_str())
        .copied()
        .ok_or_else(|| PlanError::UnknownField {
            referenced: referenced.to_string(),
            reader: reader.to_string(),
        })
}

#[derive(Clone, Copy, PartialEq)]
enum Mark {
    Unvisited,
    InProgress,
    Done,
}

fn topological_order(
    dependencies: &[Vec<usize>],
    fields: &[Field],
) -> Result<Vec<usize>, PlanError> {
    let mut marks = vec![Mark::Unvisited; dependencies.len()];
    let mut order = Vec::with_capacity(dependencies.len());
    let mut stack = Vec::new();
    for index in 0..dependencies.len() {
        visit(
            index,
            dependencies,
            fields,
            &mut marks,
            &mut order,
            &mut stack,
        )?;
    }
    Ok(order)
}

fn visit(
    index: usize,
    dependencies: &[Vec<usize>],
    fields: &[Field],
    marks: &mut [Mark],
    order: &mut Vec<usize>,
    stack: &mut Vec<usize>,
) -> Result<(), PlanError> {
    match marks[index] {
        Mark::Done => return Ok(()),
        Mark::InProgress => {
            let start = stack.iter().position(|&i| i == index).unwrap_or(0);
            let mut names: Vec<String> = stack[start..]
                .iter()
                .map(|&i| fields[i].id.to_string())
                .collect();
            names.push(fields[index].id.to_string());
            return Err(PlanError::Cycle(names.join(" -> ")));
        }
        Mark::Unvisited => {}
    }
    marks[index] = Mark::InProgress;
    stack.push(index);
    for &referenced in &dependencies[index] {
        visit(referenced, dependencies, fields, marks, order, stack)?;
    }
    stack.pop();
    marks[index] = Mark::Done;
    order.push(index);
    Ok(())
}

/// What one step of a plan does.
///
/// A step is the smallest unit of work whose result is complete: a whole field, or
/// the whole water solve. Nothing smaller is a step — a band of rows leaves a field
/// half-written, which nothing downstream may read, so it would not be a point a
/// caller could stop at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// Evaluate one field over the whole document.
    Field,
    /// Solve the water. Always last, and only present if the document declares
    /// water.
    Water,
}

/// One step of a [`BakePlan`].
#[derive(Clone, Debug, PartialEq)]
pub struct BakeStep {
    /// Whether this evaluates a field or solves the water.
    pub kind: StepKind,
    /// The field this step bakes. For a water step, the `Height` field it is solved
    /// over — empty if the document has none.
    pub field: String,
    /// Fields to release once this step has run, freeing their baked rasters.
    ///
    /// Honoured by [`Bake::advance`], but [`TerrainSpec::plan_bake`] never populates
    /// it: a finished [`Terrain`] has to answer at every cell of every field it
    /// names, so nothing a plan produces may be released before the plan ends.
    pub releases: Vec<String>,
}

/// The steps a document has to be taken through, settled before any of them runs.
///
/// A plan is a value, not a running bake: it borrows nothing, so a caller can read
/// off how many steps there are and which field each names — enough to size a
/// progress bar, or to decide the work is too large — while the document stays
/// readable. It is a snapshot, and editing the document afterwards does not update
/// it.
///
/// Fields already baked at their declared resolution are left out, so a plan for a
/// partially baked document is shorter than one for a fresh one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BakePlan {
    steps: Vec<BakeStep>,
}

impl BakePlan {
    /// The steps, in the order they must run.
    pub fn steps(&self) -> &[BakeStep] {
        &self.steps
    }

    /// How many steps. This is what a progress display counts against.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether there is nothing to do — a document with no fields, or one already
    /// fully baked and wanting no water.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// What [`Bake::advance`] says about the step it just ran.
///
/// The step ran either way — this reports whether another one follows, so a caller
/// loops on the return value rather than testing the bake before each call and
/// racing its own edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakeProgress {
    /// A step ran and at least one more remains.
    Advanced,
    /// The last step ran. A further [`Bake::advance`] is
    /// [`BakeError::NoStepRemaining`].
    Finished,
}

/// Where a bake has got to, for a caller showing progress.
#[derive(Clone, Debug, PartialEq)]
pub struct BakeReport {
    /// Steps completed so far, so `0` before the first has run.
    pub step: u32,
    /// Steps in the plan.
    pub total: u32,
    /// The field the last completed step baked; empty before the first has run.
    pub field: String,
    /// How much the baked rasters currently hold. The bakes only — fields' shader
    /// values and the solved water are not counted — so this is the number a staged
    /// bake can actually move, and it falls when a field is released.
    pub live_bytes: u64,
}

/// A document part way through being baked: it owns the spec, the plan, and
/// whatever its completed steps have written.
///
/// Constructing one allocates nothing — the plan is a list of names, and a field's
/// raster is allocated by the step that fills it — so a caller may plan a bake it
/// then decides not to run.
#[derive(Debug)]
pub struct Bake {
    spec: TerrainSpec,
    plan: BakePlan,
    next_step: u32,
    last_field: String,
}

impl Bake {
    /// The plan this bake is following. Fixed for the life of the bake.
    pub fn plan(&self) -> &BakePlan {
        &self.plan
    }

    /// The document as it stands, with the rasters the completed steps have written.
    /// Readable between steps, which is what lets a caller display a partial bake.
    pub fn spec(&self) -> &TerrainSpec {
        &self.spec
    }

    /// Where the bake has got to. Valid at any point: before the first step it
    /// reports step `0`, no field, and whatever the document was already holding.
    pub fn report(&self) -> BakeReport {
        BakeReport {
            step: self.next_step,
            total: self.plan.steps.len() as u32,
            field: self.last_field.clone(),
            live_bytes: self.spec.baked_bytes() as u64,
        }
    }

    /// Runs the next step and releases whatever that step named.
    ///
    /// Advancing past the last step is [`BakeError::NoStepRemaining`] rather than a
    /// second `Finished`, so a loop that has lost count is stopped rather than
    /// spinning. On an error the step's work may be partly done and the step counter
    /// does not advance.
    pub fn advance(&mut self) -> Result<BakeProgress, BakeError> {
        let index = self.next_step as usize;
        let Some(step) = self.plan.steps.get(index).cloned() else {
            return Err(BakeError::NoStepRemaining);
        };

        match step.kind {
            StepKind::Field => self.spec.bake_field(&step.field)?,
            StepKind::Water => {
                let water_spec = self
                    .spec
                    .water_spec
                    .clone()
                    .ok_or(BakeError::NoStepRemaining)?;
                self.spec.solve_water(&water_spec)?;
            }
        }

        for released in &step.releases {
            self.spec.release(released);
        }

        self.next_step += 1;
        self.last_field = step.field;
        if self.next_step as usize >= self.plan.steps.len() {
            Ok(BakeProgress::Finished)
        } else {
            Ok(BakeProgress::Advanced)
        }
    }

    /// The finished [`Terrain`], dropping the document that produced it — the
    /// parameter values and the values the fields' shaders hold.
    ///
    /// For a consumer that will only read. A caller that will edit and re-bake wants
    /// [`Bake::finish_keeping_spec`], since nothing rebuilds a document from a
    /// `Terrain`.
    ///
    /// [`BakeError::StepsRemaining`] if the plan has not been run to the end.
    pub fn finish(self) -> Result<Terrain, BakeError> {
        let (terrain, _) = self.finish_keeping_spec()?;
        Ok(terrain)
    }

    /// As [`Bake::finish`], handing the document back beside the terrain — for an
    /// editor, which needs the recipe to keep editing and the terrain to display.
    ///
    /// The two are independent afterwards: the terrain holds copies of the baked
    /// rasters, so editing the document does not disturb it.
    pub fn finish_keeping_spec(self) -> Result<(Terrain, TerrainSpec), BakeError> {
        let remaining = self.plan.steps.len() as u32 - self.next_step;
        if remaining > 0 {
            return Err(BakeError::StepsRemaining(remaining));
        }

        let terrain = quantize(&self.spec)?;
        Ok((terrain, self.spec))
    }
}

fn quantize(spec: &TerrainSpec) -> Result<Terrain, BakeError> {
    let readable: Vec<&Field> = spec
        .fields
        .iter()
        .filter(|field| field.baked().size() == field.resolution(spec.size))
        .collect();

    let shifts: Vec<u8> = readable.iter().map(|field| field.shift).collect();
    let (placements, layer_shifts) = plan_layers(&shifts);

    let mut layers: Vec<LayerBuild> = layer_shifts
        .iter()
        .map(|shift| LayerBuild::new(Some(*shift), resolution(spec.size, *shift)))
        .collect();

    let mut fields = Vec::with_capacity(readable.len());
    for (field, place) in readable.iter().zip(&placements) {
        let meta = value_range(field.baked().data());

        let layer = &mut layers[place.layer as usize];
        layer.push(
            meta,
            field.baked().data().iter().map(|value| meta.encode(*value)),
        );

        fields.push(FieldInfo {
            name: field.id.to_string(),
            role: field.role,
            shift: field.shift,
            categorical: false,
            layer: place.layer,
            channel: place.channel,
        });
    }

    let water = spec
        .water
        .as_ref()
        .filter(|state| state.size() == spec.size)
        .map(|state| {
            let index = layers.len();
            layers.push(water_layer(state, spec.size));
            WaterInfo {
                lakes: state.lakes(),
                layer: index as u8,
            }
        });

    Ok(Terrain::new(
        spec.size,
        fields,
        layers.into_iter().map(LayerBuild::finish).collect(),
        water,
    ))
}

fn value_range(values: &[f32]) -> ChannelMeta {
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        low = low.min(value);
        high = high.max(value);
    }
    if low > high {
        return ChannelMeta::linear(0.0, 0.0);
    }
    ChannelMeta::linear(low, high)
}

struct LayerBuild {
    shift: Option<u8>,
    size: UVec2,
    channels: Vec<ChannelMeta>,
    columns: Vec<Vec<u8>>,
}

impl LayerBuild {
    fn new(shift: Option<u8>, size: UVec2) -> Self {
        Self {
            shift,
            size,
            channels: Vec::new(),
            columns: Vec::new(),
        }
    }

    fn push(&mut self, meta: ChannelMeta, bytes: impl Iterator<Item = u8>) {
        self.channels.push(meta);
        self.columns.push(bytes.collect());
    }

    fn finish(self) -> TerrainLayer {
        let channels = self.columns.len();
        let texels = (self.size.x as usize) * (self.size.y as usize);
        let mut interleaved = vec![0u8; texels * channels];
        for (index, column) in self.columns.iter().enumerate() {
            for (texel, byte) in column.iter().enumerate() {
                interleaved[texel * channels + index] = *byte;
            }
        }
        let texels = LayerTexels::from_bytes(self.size, channels, interleaved)
            .expect("a layer is built at its own extent with one byte per channel");
        TerrainLayer::new(self.shift, self.channels, texels)
    }
}

fn water_layer(state: &WaterState, size: UVec2) -> LayerBuild {
    let mut layer = LayerBuild::new(Some(0), size);

    let deepest = state
        .depth()
        .data()
        .iter()
        .copied()
        .filter(|depth| depth.is_finite())
        .fold(0.0f32, f32::max);
    let depth_meta = ChannelMeta::log(deepest);
    layer.push(
        depth_meta,
        state.depth().data().iter().map(|depth| {
            let byte = depth_meta.encode(*depth);
            if *depth > 0.0 { byte.max(1) } else { byte }
        }),
    );

    let unit = ChannelMeta::unit();
    let flow: Vec<Vec2> = (0..size.y)
        .flat_map(|y| (0..size.x).map(move |x| (x, y)))
        .map(|(x, y)| state.flow_vector(x, y).unwrap_or(Vec2::ZERO))
        .collect();
    layer.push(unit, flow.iter().map(|vector| unit.encode(vector.x)));
    layer.push(unit, flow.iter().map(|vector| unit.encode(vector.y)));

    let largest = (0..size.y)
        .flat_map(|y| (0..size.x).map(move |x| (x, y)))
        .map(|(x, y)| state.accumulation(x, y))
        .fold(0.0f32, f32::max);
    let accum_meta = ChannelMeta::log(largest);
    layer.push(
        accum_meta,
        (0..size.y)
            .flat_map(|y| (0..size.x).map(move |x| (x, y)))
            .map(|(x, y)| accum_meta.encode(state.accumulation(x, y))),
    );

    layer
}

impl TerrainSpec {
    /// The steps this document has to be taken through.
    ///
    /// Every structural fault — a zero size, a duplicate id, an unresolvable
    /// reference, a cycle, a role conflict — is caught here, before a single raster
    /// is allocated, so a document that plans can be baked and a running bake fails
    /// only on the water solve or on an edit made since.
    pub fn plan_bake(&self) -> Result<BakePlan, PlanError> {
        if self.size.x == 0 || self.size.y == 0 {
            return Err(PlanError::ZeroSize(self.size.x, self.size.y));
        }
        self.validate_roles()?;

        let index_of = self.index_fields()?;
        let dependencies = self.resolve_dependencies(&index_of)?;
        let order = topological_order(&dependencies, &self.fields)?;

        let mut steps: Vec<BakeStep> = Vec::with_capacity(order.len() + 1);
        for &index in &order {
            let field = &self.fields[index];
            if field.baked().size() == resolution(self.size, field.shift) {
                continue;
            }
            steps.push(BakeStep {
                kind: StepKind::Field,
                field: field.id.to_string(),
                releases: Vec::new(),
            });
        }

        if self.water_spec.is_some() {
            steps.push(BakeStep {
                kind: StepKind::Water,
                field: self
                    .field_with_role(FieldRole::Height)
                    .map(|field| field.id.to_string())
                    .unwrap_or_default(),
                releases: Vec::new(),
            });
        }

        Ok(BakePlan { steps })
    }

    /// Takes the document into a [`Bake`] the caller drives a step at a time.
    ///
    /// Plans first, so this fails on everything [`TerrainSpec::plan_bake`] does.
    /// Allocates nothing beyond the plan itself; the rasters are allocated by the
    /// steps that fill them.
    pub fn begin_bake(self) -> Result<Bake, PlanError> {
        let plan = self.plan_bake()?;
        Ok(Bake {
            spec: self,
            plan,
            next_step: 0,
            last_field: String::new(),
        })
    }

    /// Plans, runs every step, and finishes — the whole of
    /// [`TerrainSpec::begin_bake`], [`Bake::advance`] and [`Bake::finish`] in one
    /// call, for a caller with no progress to report.
    ///
    /// Consumes the document; use [`TerrainSpec::begin_bake`] to keep it.
    pub fn bake(self) -> Result<Terrain, BakeError> {
        let mut bake = self.begin_bake()?;
        while !bake.plan().is_empty() && bake.next_step < bake.plan().len() as u32 {
            bake.advance()?;
        }
        bake.finish()
    }

    /// The field holding `role`. [`FieldRole::Custom`] always answers `None`,
    /// because any number of fields may hold it.
    pub fn field_with_role(&self, role: FieldRole) -> Option<&Field> {
        if role == FieldRole::Custom {
            return None;
        }
        self.fields.iter().find(|field| field.role == role)
    }

    /// Checks the three things a role assignment has to satisfy: at most one field
    /// per named role, a `Height` field at shift 0, and a `Height` field present if
    /// water is declared.
    ///
    /// Called by [`TerrainSpec::plan_bake`], and worth calling directly by anything
    /// that lets a role be changed, so the conflict is reported where it was made
    /// rather than at the next bake.
    pub fn validate_roles(&self) -> Result<(), PlanError> {
        for role in [FieldRole::Height, FieldRole::Moisture] {
            if self.fields.iter().filter(|f| f.role == role).count() > 1 {
                return Err(PlanError::DuplicateRole(role));
            }
        }
        if let Some(height) = self.field_with_role(FieldRole::Height)
            && height.shift != 0
        {
            return Err(PlanError::CoarseHeight(height.id.to_string(), height.shift));
        }
        if self.water_spec.is_some() && self.field_with_role(FieldRole::Height).is_none() {
            return Err(PlanError::MissingHeightField);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(size: UVec2) -> Raster<f32> {
        let span = (size.x + size.y) as f32;
        Raster::from_vec(
            size,
            (0..size.x * size.y)
                .map(|n| ((n % size.x) + (n / size.x)) as f32 / span)
                .collect(),
        )
        .unwrap()
    }

    fn two_field_document() -> TerrainSpec {
        TerrainSpec::new(UVec2::new(96, 80))
            .with_field(
                Field::new("moisture")
                    .with_shift(3)
                    .holding(ramp(UVec2::new(12, 10))),
            )
            .with_field(
                Field::new("height")
                    .with_shift(0)
                    .holding(ramp(UVec2::new(96, 80)))
                    .reading(&["moisture"]),
            )
    }

    // The whole point of the staged bake: a caller that walks the order one field at a
    // time has to end up with exactly the document a single `bake()` would have written,
    // or the intermediate results it showed were of a different world.
    #[test]
    fn baking_a_stage_at_a_time_writes_what_one_bake_would_have() {
        let mut whole = two_field_document().with_field(
            Field::new("relief")
                .with_range((-1.0, 1.0))
                .holding(ramp(UVec2::new(96, 80)))
                .reading(&["height"]),
        );
        let mut staged = whole.clone();

        whole.bake_in_place().unwrap();
        for id in staged.bake_order().unwrap() {
            staged.bake_field(id.as_str()).unwrap();
        }

        for field in &whole.fields {
            let one = field.baked();
            let other = staged.field(field.id.as_str()).unwrap().baked();
            assert_eq!(one.size(), other.size(), "{} changed size", field.id);
            assert!(
                one.data().iter().zip(other.data()).all(|(a, b)| a == b),
                "{} differs between a staged bake and a whole one",
                field.id
            );
        }
    }

    // Releasing is what keeps a staged bake's peak below the sum of its fields, and it
    // has to leave the document able to rebuild what it dropped.
    #[test]
    fn a_released_field_reads_as_zero_and_bakes_back() {
        let mut terrain = two_field_document();
        terrain.bake_in_place().unwrap();

        let before = terrain.baked_bytes();
        let sampled = terrain.sample("height", 12.5, 9.5).unwrap();
        assert!(
            sampled != 0.0,
            "the sample to compare against is already zero"
        );

        assert!(terrain.release("height"));
        assert!(terrain.baked_bytes() < before);
        assert_eq!(terrain.sample("height", 12.5, 9.5), Some(0.0));

        terrain.bake_field("height").unwrap();
        assert_eq!(terrain.baked_bytes(), before);
        assert_eq!(terrain.sample("height", 12.5, 9.5), Some(sampled));
    }

    // The guard on the whole point of releasing: a staged bake that drops what it no
    // longer needs must actually *hold* less, not hand the raster straight back on the
    // next stage.
    #[test]
    fn a_staged_bake_does_not_re_allocate_what_the_caller_released() {
        let mut terrain = two_field_document();
        let order = terrain.bake_order().unwrap();
        terrain.bake_field(order[0].as_str()).unwrap();

        let with_first = terrain.baked_bytes();
        terrain.release(order[0].as_str());
        let released = terrain.baked_bytes();
        assert!(released < with_first);

        terrain.bake_field(order[1].as_str()).unwrap();
        assert_eq!(
            terrain.field(order[0].as_str()).unwrap().baked().size(),
            UVec2::ZERO,
            "a released field came back when the next stage baked"
        );
    }

    // A staged bake releases by name, and the names come from a document the caller
    // may have edited since; a typo has to be a `false` it can notice, not a crash.
    #[test]
    fn releasing_a_field_that_is_not_there_says_so_rather_than_panicking() {
        let mut terrain = two_field_document();
        assert!(!terrain.release("nowhere"));
    }

    // A stage list has to be an order rather than a listing: a field that reads another
    // cannot come first, or the caller walking it would bake against zeros.
    #[test]
    fn the_stage_order_puts_a_field_after_everything_it_reads() {
        let terrain = two_field_document();
        let order = terrain.bake_order().unwrap();
        let position = |id: &str| order.iter().position(|got| got.as_str() == id).unwrap();
        assert!(position("moisture") < position("height"));
        assert_eq!(order.len(), terrain.fields.len());
    }

    // The failures a bake can only discover late are the ones a caller most wants early,
    // because a staged bake has already drawn half a world by the time it hits one.
    #[test]
    fn a_stage_list_refuses_a_document_a_bake_would_refuse() {
        let cyclic = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("a").reading(&["b"]))
            .with_field(Field::new("b").reading(&["a"]));
        assert!(matches!(cyclic.bake_order(), Err(PlanError::Cycle(_))));

        let dangling =
            TerrainSpec::new(UVec2::splat(16)).with_field(Field::new("a").reading(&["gone"]));
        assert!(matches!(
            dangling.bake_order(),
            Err(PlanError::UnknownField { .. })
        ));
    }

    // Baking a field the document does not carry is the caller's mistake and has to say
    // so, rather than quietly doing nothing.
    #[test]
    fn baking_a_field_that_is_not_there_says_which_one() {
        let mut terrain = two_field_document();
        assert!(matches!(
            terrain.bake_field("nowhere"),
            Err(PlanError::UnknownField { .. })
        ));
    }

    // The baseline the staging tests compare against: two fields
    // at different shifts, one reading the other, each allocated at its own resolution
    // and filled with finite values that actually vary.
    #[test]
    fn a_two_field_document_bakes_each_field_at_its_own_resolution() {
        let mut terrain = two_field_document();
        terrain.bake_in_place().unwrap();

        let moisture = terrain.field("moisture").unwrap();
        assert_eq!(moisture.baked().size(), UVec2::new(12, 10));
        let height = terrain.field("height").unwrap();
        assert_eq!(height.baked().size(), UVec2::new(96, 80));

        let values = height.baked().data();
        assert!(values.iter().all(|value| value.is_finite()));
        let lowest = values.iter().copied().fold(f32::INFINITY, f32::min);
        let highest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(highest - lowest > 0.05, "{lowest} to {highest}");
    }
    // Every test here, and any document baked before the editor has seen a render
    // device, bakes without a runtime: a field whose shader has produced nothing has
    // to read as `0.0` rather than failing the bake.
    #[test]
    fn a_shader_with_no_runtime_bakes_as_zero_rather_than_failing() {
        let mut terrain = TerrainSpec::new(UVec2::splat(8)).with_field(Field::new("height"));
        terrain.bake_in_place().expect("the bake was refused");
        assert!(
            terrain
                .field("height")
                .unwrap()
                .baked()
                .data()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    // The failure mode a naive walk has here is non-termination, which is far worse
    // than a wrong answer; the error also names the cycle, because a document with
    // several fields gives no other clue which reference to remove.
    #[test]
    fn a_dependency_cycle_is_an_error_rather_than_a_hang() {
        let mut terrain = TerrainSpec::new(UVec2::new(8, 8))
            .with_field(Field::new("a").reading(&["b"]))
            .with_field(Field::new("b").reading(&["a"]));
        let error = terrain.bake_in_place().unwrap_err();
        assert!(matches!(error, PlanError::Cycle(_)), "{error}");
    }

    // Caught at plan time and named on both sides: the dangling reference lives in the
    // reader, so an error naming only the missing field would not say where to look.
    #[test]
    fn a_reference_to_a_field_that_is_not_there_is_an_error() {
        let terrain =
            TerrainSpec::new(UVec2::new(8, 8)).with_field(Field::new("a").reading(&["gone"]));
        let error = terrain.plan_bake().unwrap_err();
        assert!(
            matches!(&error, PlanError::UnknownField { referenced, reader }
                if referenced == "gone" && reader == "a"),
            "{error}"
        );
    }

    // References are by name, so two fields sharing one makes every reference to it
    // ambiguous — it has to be refused rather than resolved to whichever came first.
    #[test]
    fn two_fields_may_not_share_an_id() {
        let mut terrain = TerrainSpec::new(UVec2::new(8, 8))
            .with_field(Field::new("height"))
            .with_field(Field::new("height"));
        let error = terrain.bake_in_place().unwrap_err();
        assert!(matches!(error, PlanError::DuplicateField(_)), "{error}");
    }

    // A zero extent would make every allocation and every rectangle degenerate; it is
    // refused at the top rather than producing an empty document that looks baked.
    #[test]
    fn a_document_with_no_extent_is_an_error() {
        let mut terrain = TerrainSpec::new(UVec2::new(0, 16));
        assert!(matches!(
            terrain.bake_in_place().unwrap_err(),
            PlanError::ZeroSize(0, 16)
        ));
    }
    // The clamp is applied once to what the shader produced, so a shader may leave the
    // range and still land inside it.
    #[test]
    fn a_field_is_clamped_to_the_range_it_declares() {
        let mut terrain = TerrainSpec::new(UVec2::new(8, 8))
            .with_field(Field::new("height").with_range((0.0, 1.0)).held(5.0));
        terrain.bake_in_place().unwrap();
        assert_eq!(terrain.sample("height", 4.5, 4.5).unwrap(), 1.0);
    }

    // An extent that is not a multiple of the shift rounds up to a raster whose last
    // row and column are only partly covered by the document; those texels still have
    // to be written, or a sample near the far edge reads whatever the allocation held.
    #[test]
    fn a_bake_leaves_no_texel_of_a_field_untouched() {
        let mut terrain = TerrainSpec::new(UVec2::new(37, 23))
            .with_field(Field::new("height").with_shift(2).held(0.5));
        terrain.bake_in_place().unwrap();
        let field = terrain.field("height").unwrap();
        assert_eq!(field.baked().size(), UVec2::new(10, 6));
        assert!(field.baked().data().iter().all(|value| *value == 0.5));
    }
    // The baked rasters, a shader's values and the fields its file reads are skipped by
    // serde, so this pins that everything else survives and that a decoded document is
    // unbaked rather than half-baked.
    #[test]
    fn a_document_round_trips_through_serde_without_its_bakes() {
        let terrain = two_field_document();
        let encoded = serde_json::to_string(&terrain).unwrap();
        let decoded: TerrainSpec = serde_json::from_str(&encoded).unwrap();

        let mut expected = terrain.clone();
        for field in &mut expected.fields {
            *field = field.authored();
            field.shader.layers.clear();
        }
        assert_eq!(decoded, expected);
        assert!(decoded.field("height").unwrap().baked().is_empty());
    }

    fn roled_document() -> TerrainSpec {
        TerrainSpec::new(UVec2::new(64, 64))
            .with_field(
                Field::new("moisture")
                    .with_role(FieldRole::Moisture)
                    .with_shift(3)
                    .holding(ramp(UVec2::new(8, 8))),
            )
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .holding(ramp(UVec2::new(64, 64))),
            )
    }

    // The plan is what a caller reads before a texel is written, so a step per field in
    // dependency order is the whole of what it promises.
    #[test]
    fn a_plan_carries_one_step_per_field_in_the_order_a_bake_visits_them() {
        let plan = roled_document().plan_bake().unwrap();
        let fields: Vec<_> = plan.steps().iter().map(|step| step.field.clone()).collect();
        assert_eq!(fields, vec!["moisture", "height"]);
        assert!(plan.steps().iter().all(|step| step.kind == StepKind::Field));
    }

    // Water reads a baked height, so its step has to come after every field step; a
    // plan that ordered it anywhere else would solve over an empty raster.
    #[test]
    fn a_spec_that_declares_water_plans_a_water_step_last() {
        let mut spec = roled_document();
        spec.water_spec = Some(WaterSpec::new("height").with_moisture("moisture"));

        let plan = spec.plan_bake().unwrap();
        assert_eq!(plan.steps().last().unwrap().kind, StepKind::Water);
        assert_eq!(plan.steps().len(), 3);
    }

    // Advancing answers Finished on the last step and nothing after it — the caller drives
    // its loop on that rather than asking the bake a second question.
    #[test]
    fn advancing_answers_finished_on_the_last_step_and_refuses_after_it() {
        let mut bake = roled_document().begin_bake().unwrap();
        assert_eq!(bake.advance().unwrap(), BakeProgress::Advanced);
        assert_eq!(bake.advance().unwrap(), BakeProgress::Finished);
        assert!(matches!(
            bake.advance().unwrap_err(),
            BakeError::NoStepRemaining
        ));
    }

    // What a progress display reads. The field named is the one that has *finished*,
    // not the one about to start, which is the distinction an off-by-one here would
    // blur.
    #[test]
    fn a_report_names_the_field_whose_step_ran_and_counts_the_rest() {
        let mut bake = roled_document().begin_bake().unwrap();
        bake.advance().unwrap();

        let report = bake.report();
        assert_eq!((report.step, report.total), (1, 2));
        assert_eq!(report.field, "moisture");
        assert!(report.live_bytes > 0);
    }

    // A Terrain exists only when every step has run, or it would carry a field that reads
    // as zero everywhere while looking exactly like one that was baked.
    #[test]
    fn finishing_before_every_step_has_run_is_refused() {
        let mut bake = roled_document().begin_bake().unwrap();
        bake.advance().unwrap();

        assert!(matches!(
            bake.finish().unwrap_err(),
            BakeError::StepsRemaining(1)
        ));
    }

    // The one-call spelling and the stepped one are the same bake, or the progress a
    // caller watched was of a different world.
    #[test]
    fn stepping_a_bake_writes_what_the_one_call_spelling_writes() {
        let stepped = {
            let mut bake = roled_document().begin_bake().unwrap();
            while bake.advance().unwrap() == BakeProgress::Advanced {}
            bake.finish().unwrap()
        };
        let whole = roled_document().bake().unwrap();

        for view in whole.fields() {
            let other = stepped.field(view.name()).unwrap();
            assert_eq!(view.bytes(), other.bytes(), "{} differs", view.name());
        }
    }

    // Role validation is part of planning precisely so it costs nothing: a document
    // with a role conflict must fail before it allocates a field's worth of memory.
    #[test]
    fn two_fields_claiming_one_role_is_refused_before_a_raster_is_allocated() {
        let mut spec = roled_document();
        spec.fields[0].role = FieldRole::Height;

        assert!(matches!(
            spec.plan_bake().unwrap_err(),
            PlanError::DuplicateRole(FieldRole::Height)
        ));
    }

    // The solve reads its height one texel per cell and will not resample one, so a coarse
    // height is refused at plan time rather than discovered at the water step.
    #[test]
    fn a_coarse_height_field_is_refused_at_plan_time() {
        let mut spec = roled_document();
        spec.fields[1].shift = 2;

        assert!(matches!(
            spec.plan_bake().unwrap_err(),
            PlanError::CoarseHeight(_, 2)
        ));
    }

    // The water step names the height field by role, so a document declaring water
    // with no `Height` field would otherwise plan happily and fail part way through
    // the bake.
    #[test]
    fn declaring_water_without_a_height_field_is_refused_at_plan_time() {
        let mut spec = roled_document();
        spec.fields[1].role = FieldRole::Custom;
        spec.water_spec = Some(WaterSpec::new("height"));

        assert!(matches!(
            spec.plan_bake().unwrap_err(),
            PlanError::MissingHeightField
        ));
    }

    // A spec whose file carried every bake plans no steps at all — that is what a
    // bakes-only export is, and it reaches a Terrain without evaluating a texel.
    #[test]
    fn a_spec_that_carries_every_bake_plans_no_steps() {
        let mut spec = roled_document();
        spec.bake_in_place().unwrap();

        assert!(spec.plan_bake().unwrap().is_empty());
    }

    // A document written before roles existed loads with every field defaulted to Custom,
    // so a water spec it carried names a height field the plan can no longer find.
    #[test]
    fn a_document_that_carries_no_roles_refuses_to_plan_its_water() {
        let mut spec =
            TerrainSpec::new(UVec2::new(32, 32)).with_field(Field::new("height").held(0.5));
        spec.water_spec = Some(WaterSpec::new("height"));

        assert!(matches!(
            spec.plan_bake().unwrap_err(),
            PlanError::MissingHeightField
        ));
    }

    // A file's `@layer` name is the only thing that says one field needs another, so
    // the order has to follow it even when the reader is declared first.
    #[test]
    fn a_field_whose_shader_names_another_is_baked_after_it() {
        let terrain = TerrainSpec::new(UVec2::splat(8))
            .with_field(
                Field::new("reader")
                    .holding(ramp(UVec2::splat(8)))
                    .reading(&["source"]),
            )
            .with_field(Field::new("source").held(0.5));
        let order: Vec<_> = terrain
            .bake_order()
            .unwrap()
            .into_iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(order, ["source", "reader"]);
        let steps: Vec<_> = terrain
            .plan_bake()
            .unwrap()
            .steps()
            .iter()
            .map(|step| step.field.clone())
            .collect();
        assert_eq!(steps, ["source", "reader"]);
    }

    // A mistyped name in one file must not stop the rest of the document from baking;
    // the reader alone is left unbaked, and the fault names what it could not find.
    #[test]
    fn a_field_naming_no_field_is_left_unbaked_while_the_rest_bakes() {
        let mut terrain = TerrainSpec::new(UVec2::new(96, 80))
            .with_field(
                Field::new("moisture")
                    .with_shift(3)
                    .holding(ramp(UVec2::new(12, 10))),
            )
            .with_field(
                Field::new("height")
                    .holding(ramp(UVec2::new(96, 80)))
                    .reading(&["nowhere"]),
            );
        terrain.bake_in_place().expect("the bake was refused");

        assert_eq!(
            terrain.field("moisture").unwrap().baked().size(),
            UVec2::new(12, 10)
        );
        assert!(terrain.field("height").unwrap().baked().is_empty());
        let faults = terrain.field_faults();
        assert_eq!(faults.len(), 1, "{faults:?}");
        assert_eq!(faults[0].0.as_str(), "height");
        assert!(faults[0].1.contains("nowhere"), "{}", faults[0].1);
    }

    // Two files naming each other cannot be ordered, so the bake fails rather than
    // spinning, and both fields carry the chain that says which names to change.
    #[test]
    fn two_shaders_naming_each_other_fail_the_bake_and_both_carry_the_chain() {
        let mut terrain = TerrainSpec::new(UVec2::splat(8))
            .with_field(Field::new("a").reading(&["b"]))
            .with_field(Field::new("b").reading(&["a"]));
        let error = terrain.bake_in_place().unwrap_err();
        assert!(
            matches!(&error, PlanError::Cycle(chain) if chain == "a -> b -> a"),
            "{error}"
        );
        let faults = terrain.field_faults();
        assert_eq!(faults.len(), 2, "{faults:?}");
        assert!(
            faults
                .iter()
                .all(|(_, fault)| fault.contains("a -> b -> a"))
        );
    }
}
