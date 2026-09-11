//! The picture on a node's card: what a node that holds a raster of its own is shown
//! as, and what keeps that picture in step with the values the node last produced.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use watershed::raster::{Raster, Texel};

use super::{CanvasShape, open_graph};
use crate::document::Document;
use crate::terrain::graph::{GraphNode, NodeId, NodeOp};

const TEXELS: u32 = 64;

/// The square on a card that shows what the node produced.
#[derive(Component)]
pub struct CardThumb {
    /// The node the picture is read from.
    pub node: NodeId,
    blank: Color,
    drawn: Option<u64>,
}

impl CardThumb {
    /// A thumbnail for `node`, showing `blank` until the node has a raster to draw
    /// and again if it loses it.
    pub fn new(node: NodeId, blank: Color) -> Self {
        Self {
            node,
            blank,
            drawn: None,
        }
    }
}

/// Redraws each card's picture from the node's own raster.
///
/// Only a node that holds a raster — a shader, a painted or an imported one — has a
/// picture; every other op keeps the flat accent square, because drawing one would
/// mean a preview bake of a whole field per card per edit.
pub fn sync_thumbnails(
    document: Res<Document>,
    shape: Res<CanvasShape>,
    mut images: ResMut<Assets<Image>>,
    mut thumbs: Query<(&mut CardThumb, &mut Sprite)>,
) {
    if !document.is_changed() && !shape.is_changed() {
        return;
    }
    let Some(graph) = open_graph(&document) else {
        return;
    };
    for (mut thumb, mut sprite) in &mut thumbs {
        let picture = graph.node(thumb.node).and_then(node_picture);
        match picture {
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

fn node_picture(node: &GraphNode) -> Option<Vec<u8>> {
    match &node.op {
        NodeOp::Shader(shader) => picture(shader.values()),
        NodeOp::External(raster) => picture(raster),
        NodeOp::Paint(raster) => picture(raster),
        _ => None,
    }
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
    // `field.wgsl` says and flips `v` for. Without the flip every card's picture is
    // upside down against the map showing the same node.
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

    // A node that has not varied yet — a constant shader, a fresh paint layer — has a
    // zero span, and normalising against it would divide by zero.
    #[test]
    fn a_flat_raster_draws_one_colour_and_does_not_divide_by_zero() {
        let bytes = picture(&Raster::new(UVec2::splat(8), 0.5)).expect("a picture");
        assert_eq!(bytes.len(), (TEXELS * TEXELS * 4) as usize);
        assert!(bytes.chunks(4).all(|pixel| pixel == &bytes[..4]));
    }

    // A paint layer is bytes, not floats, so it reaches the picture through `Texel`
    // and its two ends have to land at the two ends of the ramp.
    #[test]
    fn a_byte_raster_is_read_through_texel() {
        let mut raster = Raster::new(UVec2::splat(4), 0u8);
        for x in 0..4 {
            raster.set(x, 3, 255);
        }
        let bytes = picture(&raster).expect("a picture");
        assert_ne!(row(&bytes, 0), row(&bytes, TEXELS - 1));
    }

    // A node with no raster of its own keeps the flat accent square rather than
    // drawing a picture of nothing.
    #[test]
    fn an_op_with_no_raster_has_no_picture() {
        let node = GraphNode::new(NodeId(0), NodeOp::Constant(0.5), [0.0, 0.0]);
        assert!(node_picture(&node).is_none());
    }
}
