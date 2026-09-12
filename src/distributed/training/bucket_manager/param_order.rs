//! Leaf-parameter ordering derived from the autograd graph.

use std::collections::HashSet;
use std::sync::Arc;

use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::TensorId;

/// Extract leaf parameter IDs from a computation graph in backward traversal order.
///
/// Performs a topological sort of the graph (same as backward), collects
/// leaf node IDs (those with no `grad_fn`), and returns them in the order
/// they are encountered during backward (reverse topological order).
///
/// This ordering is optimal for bucket construction: gradients computed
/// first during backward should be in the same bucket so the bucket fills
/// quickly and allreduce can start early.
pub fn param_order_from_graph<R: Runtime>(loss: &Var<R>) -> Vec<TensorId> {
    let mut topo = Vec::new();
    let mut visited = HashSet::new();

    fn dfs<R: Runtime>(
        id: TensorId,
        grad_fn: Option<Arc<dyn numr::autograd::GradFn<R>>>,
        visited: &mut HashSet<TensorId>,
        topo: &mut Vec<(TensorId, bool)>, // (id, is_leaf)
    ) {
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        let input_ids: Vec<TensorId> = grad_fn
            .as_ref()
            .map(|gf| gf.inputs().to_vec())
            .unwrap_or_default();

        if let Some(gf) = &grad_fn {
            for (input_id, input_grad_fn) in input_ids.iter().zip(gf.input_grad_fns()) {
                dfs(*input_id, input_grad_fn, visited, topo);
            }
        }

        topo.push((id, grad_fn.is_none()));
    }

    dfs(loss.id(), loss.grad_fn().cloned(), &mut visited, &mut topo);

    // Reverse topological order, keep only leaves
    topo.into_iter()
        .rev()
        .filter(|(_, is_leaf)| *is_leaf)
        .map(|(id, _)| id)
        .collect()
}
