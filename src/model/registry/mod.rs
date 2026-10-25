//! Model registry for loading models by name.

mod accessors;
mod dims;
mod load;
mod model;
mod unsupported;

pub use model::LoadedModel;
pub use unsupported::unsupported_model_refusal;
