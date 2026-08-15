// This exmaple shows how to integrate a terrain into an application.

use watershed::{BakeProgress, Error, FieldRole, FieldView, Terrain, TerrainSpec};

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "terrain.watershed".to_owned());
    let quiet = args.next().as_deref() == Some("--quiet");

    let terrain = if quiet {
        load(&path)?
    } else {
        load_showing_progress(&path)?
    };

    report_fields(&terrain);
    sample_a_cell(&terrain, 512, 512);

    if let Some(fields) = WorldFields::resolve(&terrain) {
        let tile = fields.height_tile(0, 0, 32);
        println!("tile: {} heights", tile.len());
        println!("biome at 512,512: {:?}", fields.biome_at(512, 512));
        println!(
            "temperature at 512.5,512.5: {}",
            fields.temperature_at(512.5, 512.5)
        );
    }

    Ok(())
}

fn load(path: &str) -> Result<Terrain, Error> {
    Ok(TerrainSpec::load_from_path(path)?.bake()?)
}

fn load_showing_progress(path: &str) -> Result<Terrain, Error> {
    let spec = TerrainSpec::load_from_path(path)?;
    let mut bake = spec.begin_bake()?;

    println!("{path}: {} step(s)", bake.plan().steps().len());

    while !bake.plan().is_empty() {
        let progress = bake.advance()?;
        let report = bake.report();
        println!(
            "  {}/{}  {}  {} MiB live",
            report.step,
            report.total,
            report.field,
            report.live_bytes >> 20,
        );
        if progress == BakeProgress::Finished {
            break;
        }
    }

    Ok(bake.finish()?)
}

fn report_fields(terrain: &Terrain) {
    println!("{}x{} cells", terrain.width(), terrain.height());
    for view in terrain.fields() {
        println!(
            "  {:12} {:8} {}x{} texels  {}..{}",
            view.name(),
            view.role(),
            view.texel_width(),
            view.texel_height(),
            view.range_low(),
            view.range_high(),
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
    }
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
