//! Worked example of consuming a terrain from an application: opening a terrain
//! directory, locating a world position as a chunk and a texel within it, loading
//! only that chunk, and reading fields back by role, by name, by cell and by layer.
//!
//! Nothing here bakes. A terrain directory holds the values already, so a consuming
//! project reads them straight out of it — which is the whole reason the format
//! stores images beside its metadata rather than a recipe to re-derive them from.

use watershed::{FieldRole, FieldView, IoError, Terrain, TerrainDir};

fn main() -> Result<(), IoError> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "terrain".to_owned());

    let dir = TerrainDir::open(&path)?;
    let grid = dir.grid();
    println!(
        "{}x{} chunks of {}x{} cells",
        grid.chunks().x,
        grid.chunks().y,
        grid.chunk_size().x,
        grid.chunk_size().y
    );

    let Some(at) = grid.locate(512.5, 512.5) else {
        println!("512.5,512.5 is outside the terrain");
        return Ok(());
    };
    let terrain = dir.load_chunk(at.chunk)?;
    println!("512.5,512.5 is texel {} of chunk {}", at.texel, at.chunk);

    report_fields(&terrain);
    report_layers(&terrain);
    sample_a_cell(&terrain, at.texel.x, at.texel.y);

    if let Some(fields) = WorldFields::resolve(&terrain) {
        let tile = fields.height_tile(0, 0, 32);
        println!("tile: {} heights", tile.len());
        println!(
            "biome at {}: {:?}",
            at.texel,
            fields.biome_at(at.texel.x, at.texel.y)
        );
        println!(
            "temperature at {}: {}",
            at.texel,
            fields.temperature_at(at.texel.x as f32 + 0.5, at.texel.y as f32 + 0.5)
        );
    }

    Ok(())
}

fn report_fields(terrain: &Terrain) {
    println!("chunk of {}x{} cells", terrain.width(), terrain.height());
    for view in terrain.fields() {
        println!(
            "  {:12} {:8} {}x{} texels  {}..{}  {:?} at byte {} of every {}",
            view.name(),
            view.role(),
            view.texel_width(),
            view.texel_height(),
            view.range_low(),
            view.range_high(),
            view.format(),
            view.offset(),
            view.stride(),
        );
    }
}

fn report_layers(terrain: &Terrain) {
    for index in 0..terrain.layer_count() {
        let Some(layer) = terrain.layer(index) else {
            continue;
        };
        println!(
            "  layer {index}: {} {:?} channel(s), {} bytes",
            layer.channels(),
            layer.format(),
            layer.bytes().len(),
        );
    }
}

fn sample_a_cell(terrain: &Terrain, x: u32, y: u32) {
    let Some(height) = terrain.field_with_role(FieldRole::Height) else {
        return;
    };
    let Some(ground) = height.value_at(x, y) else {
        return;
    };
    println!("cell {x},{y}: ground {ground}");
    println!(
        "  {} distinct heights in this chunk",
        distinct_levels(&height, terrain.width(), terrain.height())
    );

    for view in terrain
        .fields()
        .filter(|view| view.role() == FieldRole::Custom)
    {
        if let Some(value) = view.value_at(x, y) {
            println!("  {} {value}", view.name());
        }
    }

    if let Some(water) = terrain.water()
        && let Some(depth) = water.depth_at(x, y)
        && depth > 0.0
    {
        println!("  water {depth} deep, surface at {}", ground + depth);
        if let Some(flow) = water.flow_at(x, y) {
            println!("  flowing towards {flow}");
        }
    }
}

fn distinct_levels(field: &FieldView, width: u32, height: u32) -> usize {
    let mut levels = std::collections::BTreeSet::new();
    for y in 0..height {
        for x in 0..width {
            if let Some(value) = field.value_at(x, y) {
                levels.insert(value.to_bits());
            }
        }
    }
    levels.len()
}

struct WorldFields<'a> {
    height: FieldView<'a>,
    temperature: FieldView<'a>,
    biome: FieldView<'a>,
}

impl<'a> WorldFields<'a> {
    fn resolve(terrain: &'a Terrain) -> Option<Self> {
        Some(Self {
            height: terrain.field_with_role(FieldRole::Height)?,
            temperature: terrain.field("temperature")?,
            biome: terrain.field("biome")?,
        })
    }

    fn height_tile(&self, min_x: u32, min_y: u32, side: u32) -> Vec<f32> {
        let mut tile = Vec::with_capacity((side * side) as usize);
        for y in min_y..min_y + side {
            for x in min_x..min_x + side {
                tile.push(self.height.value_at(x, y).unwrap_or(0.0));
            }
        }
        tile
    }

    fn biome_at(&self, x: u32, y: u32) -> Option<u16> {
        self.biome.value_at(x, y).map(|value| value as u16)
    }

    fn temperature_at(&self, x: f32, y: f32) -> f32 {
        self.temperature.sample(x, y)
    }
}
