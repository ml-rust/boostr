//! Paged-attention forward — decode against a `LayeredPagedKvCache`.

mod attend;
mod forward;
mod packed_rope;
#[cfg(test)]
mod test_support;
