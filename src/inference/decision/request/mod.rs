//! Building decision requests on the host: a plan from full prompt rows
//! (`rows`), many prefix-free requests as one run (`merge`), and the
//! merged result cut back per request (`split`).

pub mod merge;
pub mod rows;
pub mod split;

pub use merge::{DecisionRequest, GatherIndexed, MergedPart, MergedRequest, merge_requests};
pub use rows::{RowSlot, RowsPlan, plan_from_rows};
pub use split::split_result;
