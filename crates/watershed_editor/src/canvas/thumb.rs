//! The picture on a node's card: which nodes are drawn rather than left flat, what
//! raster each is drawn from, and what keeps those pictures in step with the document.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use watershed::raster::{Raster, Texel};

use super::CanvasShape;
use crate::document::Document;
use crate::terrain::TerrainSpec;
use crate::terrain::graph::{GraphNode, NodeId, NodeOp};

const TEXELS: u32 = 64;

/// Which raster a card's picture is drawn from.
///
/// The two kinds of card the canvas draws: a node of the open field's graph, drawn
/// from whatever that node produced, and a field of the document, drawn from that
/// field's own bake.
pub enum ThumbSource {
    /// A node of the open field's graph.
    Node(NodeId),
    /// A field of the document, by name.
    Field(String),
}

/// The square on a card that shows what the card stands for.
#[derive(Component)]
pub struct CardThumb {
    /// What the picture is read from.
    pub source: ThumbSource,
    blank: Color,
    drawn: Option<u64>,
}

impl CardThumb {
    /// A thumbnail for `source`, showing `blank` until there is a raster to draw and
    /// again if it loses one.
    pub fn new(source: ThumbSource, blank: Color) -> Self {
        Self {
            source,
            blank,
            drawn: None,
        }
    }
}

/// Redraws each card's picture from the raster behind it.
///
/// A shader node is drawn from the raster it holds, and a reference is drawn from the
/// bake of the field it names — which costs nothing extra, since that field is already
/// baked. A shader node holding no values yet keeps the flat accent square. A field
/// card is drawn from that field's own bake, and so keeps the flat square until the
/// field has been baked once.
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
        let picture = source_picture(terrain, document.active(), &thumb.source);
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

fn source_picture(terrain: &TerrainSpec, active: &str, source: &ThumbSource) -> Option<Vec<u8>> {
    match source {
        ThumbSource::Node(id) => terrain
            .field(active)
            .and_then(|field| field.graph.node(*id))
            .and_then(|node| node_picture(terrain, node)),
        ThumbSource::Field(name) => terrain.field(name).and_then(|field| picture(field.baked())),
    }
}

fn node_picture(terrain: &TerrainSpec, node: &GraphNode) -> Option<Vec<u8>> {
    match &node.op {
        NodeOp::Shader(shader) => picture(shader.values()),
        NodeOp::FieldRef(id) => terrain
            .field(id.as_str())
            .and_then(|field| picture(field.baked())),
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
    use crate::terrain::Field;
    use crate::terrain::shader::ShaderLayer;
    use watershed::FieldId;

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

    // A node that has not varied yet — a shader holding one value — has a
    // zero span, and normalising against it would divide by zero.
    #[test]
    fn a_flat_raster_draws_one_colour_and_does_not_divide_by_zero() {
        let bytes = picture(&Raster::new(UVec2::splat(8), 0.5)).expect("a picture");
        assert_eq!(bytes.len(), (TEXELS * TEXELS * 4) as usize);
        assert!(bytes.chunks(4).all(|pixel| pixel == &bytes[..4]));
    }

    fn two_field_terrain() -> TerrainSpec {
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(
                Field::new("base")
                    .with_range((0.0, 1.0))
                    .with_op(NodeOp::holding(ramp(16))),
            )
            .with_field(Field::new("height").with_op(NodeOp::FieldRef(FieldId::from("base"))));
        terrain.bake_in_place().expect("a bake");
        terrain
    }

    // A shader node that has never been dispatched holds no values yet, so it keeps
    // the flat accent square rather than drawing a picture of nothing.
    #[test]
    fn an_op_with_no_raster_has_no_picture() {
        let terrain = two_field_terrain();
        let node = GraphNode::new(
            NodeId(0),
            NodeOp::Shader(ShaderLayer::new("x.wgsl")),
            [0.0, 0.0],
        );
        assert!(node_picture(&terrain, &node).is_none());
    }

    // The first acceptance criterion: a reference card stops being a blank square and
    // shows the field it names, drawn from that field's own bake — so an edit to the
    // referenced field moves the thumbnail with it.
    #[test]
    fn a_reference_draws_the_baked_raster_of_the_field_it_names() {
        let terrain = two_field_terrain();
        let node = GraphNode::new(
            NodeId(0),
            NodeOp::FieldRef(FieldId::from("base")),
            [0.0, 0.0],
        );
        let drawn = node_picture(&terrain, &node).expect("a picture of `base`");
        let expected = picture(terrain.field("base").unwrap().baked()).expect("a picture");
        assert_eq!(drawn, expected);
    }

    // Acceptance criterion two's half that is a picture: a field's card is drawn from
    // that field's own bake, which is the same raster the map shows when the field is
    // opened.
    #[test]
    fn a_field_source_draws_that_fields_baked_raster() {
        let terrain = two_field_terrain();
        let drawn = source_picture(&terrain, "height", &ThumbSource::Field("base".to_owned()))
            .expect("a picture of `base`");
        let expected = picture(terrain.field("base").unwrap().baked()).expect("a picture");
        assert_eq!(drawn, expected);
        assert!(
            source_picture(
                &terrain,
                "height",
                &ThumbSource::Field("nowhere".to_owned())
            )
            .is_none(),
            "a field the document does not carry drew something"
        );
    }

    // A reference the document cannot resolve draws nothing rather than panicking or
    // drawing whatever field happens to be first; the broken card is what says so.
    #[test]
    fn a_reference_to_a_field_that_is_not_there_has_no_picture() {
        let terrain = two_field_terrain();
        let node = GraphNode::new(
            NodeId(0),
            NodeOp::FieldRef(FieldId::from("nowhere")),
            [0.0, 0.0],
        );
        assert!(node_picture(&terrain, &node).is_none());
    }
}
