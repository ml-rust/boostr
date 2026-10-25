pub mod detect;
pub mod detectors;
pub mod types;

pub use detect::{
    detect_architecture, detect_architecture_from_names, detect_architecture_from_shapes,
};
pub use detectors::ShapeOf;
pub use types::{DetectedConfig, LayerType, ModelFormat};
