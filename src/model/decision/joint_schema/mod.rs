//! The Clef joint-schema head: every field of a record scored jointly from
//! all final-normed hidden rows and the output-projection rows of each
//! option's tokens. Sizes and dtype (`config`), weights (`weights`,
//! `load`), the building blocks (`blocks`, `routing`, `decoder_layer`),
//! fields (`item`), index layout and reductions (`layout`), the forward
//! (`forward`), and the head as a `DecisionHead` (`head`).

pub mod blocks;
pub mod config;
pub mod decoder_layer;
pub mod forward;
pub mod head;
pub mod item;
pub mod layout;
pub mod load;
pub mod routing;
pub mod weights;

pub use config::{HeadDtype, JointSchemaConfig};
pub use forward::Stages;
pub use head::JointSchemaHead;
pub use item::{FieldKind, JointOption, JointSchemaItem, full_sequence_plan, joint_answer_options};
pub use load::JointScalars;
