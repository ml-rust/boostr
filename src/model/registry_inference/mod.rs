//! Inference forward passes and expert weight management for [`crate::model::registry::LoadedModel`].

mod cuda_graph;
mod generic;
mod recurrent;
