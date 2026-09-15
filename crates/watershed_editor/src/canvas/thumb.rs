//! The picture on a layer's card: what raster it is drawn from, and what keeps those
//! pictures in step with the document.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use watershed::raster::{Raster, Texel};

use super::CanvasShape;
use crate::document::Document;
use crate::terrain::TerrainSpec;

const TEXELS: u32 = 64;

/// The square on a card that shows the layer the card stands for.
#[derive(Component)]
pub struct CardThumb {
    /// The layer the picture is drawn from, by name.
    pub layer: String,
    blank: Color,
    drawn: Option<u64>,
}

impl CardThumb {
    /// A thumbnail for the layer named `layer`, showing `blank` until there is a raster
    /// to draw and again if it loses one.
    pub fn new(layer: String, blank: Color) -> Self {
        Self {
            layer,
            blank,
            drawn: None,
        }
    }
}

/// Redraws each card's picture from the bake of the layer it names.
///
/// A card keeps the flat square until its layer has been baked once, and goes back to
/// it if the document stops carrying that layer.
pub fn sync_thumbnails(
    document: Res<Document>,
    shape: Res<CanvasShape>,
    mut images: ResMut<Assets<Image>>,
    mut thumbs: Query<(&mut CardThumb, &mut Sprite)>,
) {
    if !document.is_changed() && !shape.is_changed() {
        return;
    }
    let Some(terrain) = document.terrain() else {
        return;
    };
    for (mut thumb, mut sprite) in &mut thumbs {
        match layer_picture(terrain, &thumb.layer) {
            Some(bytes) => {
                let mark = hash(&bytes);
                if thumb.drawn == Some(mark) {
                    continue;
                }
                sprite.image = images.add(image_of(bytes));
                sprite.color = Color::WHITE;
                thumb.drawn = Some(mark);
            }
            None => {
                if thumb.drawn.is_none() {
                    continue;
                }
                sprite.image = Handle::default();
                sprite.color = thumb.blank;
                thumb.drawn = None;
            }
        }
    }
}

fn layer_picture(terrain: &TerrainSpec, layer: &str) -> Option<Vec<u8>> {
    terrain
        .layer(layer)
        .and_then(|layer| picture(layer.baked()))
}

fn picture<T: Texel>(raster: &Raster<T>) -> Option<Vec<u8>> {
    if raster.is_empty() {
        return None;
    }
    let mut sampled = vec![0.0f32; (TEXELS * TEXELS) as usize];
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for row in 0..TEXELS {
        let source_y = (TEXELS - 1 - row) * raster.height() / TEXELS;
        for column in 0..TEXELS {
            let source_x = column * raster.width() / TEXELS;
            let value = raster
                .get(source_x, source_y)
                .map_or(0.0, |texel| texel.to_f32());
            sampled[(row * TEXELS + column) as usize] = value;
            low = low.min(value);
            high = high.max(value);
        }
    }
    let span = high - low;
    let mut bytes = Vec::with_capacity(sampled.len() * 4);
    for value in sampled {
        let t = if span < 1e-6 {
            0.5
        } else {
            (value - low) / span
        };
        let colour = crate::material::sequential(t);
        bytes.push((colour.x.clamp(0.0, 1.0) * 255.0) as u8);
        bytes.push((colour.y.clamp(0.0, 1.0) * 255.0) as u8);
        bytes.push((colour.z.clamp(0.0, 1.0) * 255.0) as u8);
        bytes.push(255);
    }
    Some(bytes)
}

fn image_of(bytes: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: TEXELS,
            height: TEXELS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::nearest();
    image
}

fn hash(bytes: &[u8]) -> u64 {
    let mut mark = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        mark ^= *byte as u64;
        mark = mark.wrapping_mul(0x0000_0100_0000_01b3);
    }
    mark
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::Layer;

    fn ramp(size: u32) -> Raster<f32> {
        let mut raster = Raster::new(UVec2::splat(size), 0.0);
        for y in 0..size {
            for x in 0..size {
                raster.set(x, y, y as f32 / (size - 1) as f32);
            }
        }
        raster
    }

    fn row(bytes: &[u8], index: u32) -> [u8; 3] {
        let at = (index * TEXELS * 4) as usize;
        [bytes[at], bytes[at + 1], bytes[at + 2]]
    }

    // The raster's row zero is its bottom and an image's row zero is its top, which
    // `layer.wesl` says and flips `v` for. Without the flip every card's picture is
    // upside down against the map showing the same layer.
    #[test]
    fn the_rasters_bottom_row_lands_in_the_images_last_row() {
        let bytes = picture(&ramp(16)).expect("a picture");
        let top = row(&bytes, 0);
        let bottom = row(&bytes, TEXELS - 1);
        assert_ne!(top, bottom);
        let dark = |pixel: [u8; 3]| pixel.iter().map(|c| *c as u32).sum::<u32>();
        assert!(
            dark(bottom) > dark(top),
            "top {top:?} bottom {bottom:?} are not the way round the ramp draws"
        );
    }

    // A layer that has not varied yet — a shader holding one value — has a zero span,
    // and normalising against it would divide by zero.
    #[test]
    fn a_flat_raster_draws_one_colour_and_does_not_divide_by_zero() {
        let bytes = picture(&Raster::new(UVec2::splat(8), 0.5)).expect("a picture");
        assert_eq!(bytes.len(), (TEXELS * TEXELS * 4) as usize);
        assert!(bytes.chunks(4).all(|pixel| pixel == &bytes[..4]));
    }

    fn two_layer_terrain() -> TerrainSpec {
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_layer(Layer::new("base").with_range((0.0, 1.0)).holding(ramp(16)))
            .with_layer(Layer::new("height").reading(&["base"]));
        terrain.bake_in_place().expect("a bake");
        terrain
    }

    // A layer's card is drawn from that layer's own bake, which is the same raster the
    // map shows when the layer is opened — and a name the document does not carry draws
    // nothing rather than whatever layer happens to be first.
    #[test]
    fn a_card_draws_its_layers_baked_raster() {
        let terrain = two_layer_terrain();
        let drawn = layer_picture(&terrain, "base").expect("a picture of `base`");
        let expected = picture(terrain.layer("base").unwrap().baked()).expect("a picture");
        assert_eq!(drawn, expected);
        assert!(
            layer_picture(&terrain, "nowhere").is_none(),
            "a layer the document does not carry drew something"
        );
    }
}
