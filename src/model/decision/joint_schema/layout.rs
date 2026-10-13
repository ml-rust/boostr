//! The host-built index layout of one record's fields ([`Layout`]) and the
//! row reductions the head composes from numr ops: span means
//! ([`span_means`]), row dot products ([`row_dots`]) and L2 normalisation
//! ([`l2_normalize`]). Reductions accumulate in F32, as torch reduces a
//! half-width tensor, and return the compute dtype.

use super::item::JointSchemaItem;
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::ops::Range;

/// Where each field's options sit once all options are concatenated in
/// field order, then head option order.
pub(super) struct Layout {
    /// Each field's instruction span.
    pub(super) questions: Vec<Range<usize>>,
    /// Each option's span, all fields concatenated.
    pub(super) options: Vec<Range<usize>>,
    /// Each option's slice of the concatenated option ids.
    pub(super) lexical: Vec<Range<usize>>,
    /// The field of each option.
    pub(super) owner: Vec<i64>,
    /// The option count of each field.
    pub(super) counts: Vec<usize>,
    /// The type-embedding row of each field.
    pub(super) types: Vec<i64>,
    /// The largest option count.
    pub(super) widest: usize,
}

impl Layout {
    pub(super) fn new(items: &[JointSchemaItem]) -> Self {
        let mut options = Vec::new();
        let mut lexical = Vec::new();
        let mut owner = Vec::new();
        let mut next = 0usize;
        for (q, item) in items.iter().enumerate() {
            for option in &item.options {
                options.push(option.span.clone());
                lexical.push(next..next + option.ids.len());
                next += option.ids.len();
                owner.push(q as i64);
            }
        }
        let counts: Vec<usize> = items.iter().map(|i| i.options.len()).collect();
        Self {
            questions: items.iter().map(|i| i.question.clone()).collect(),
            options,
            lexical,
            owner,
            widest: counts.iter().copied().max().unwrap_or(0),
            counts,
            types: items.iter().map(|i| i.kind.type_id() as i64).collect(),
        }
    }

    /// The option count over all fields.
    pub(super) fn total(&self) -> usize {
        self.options.len()
    }

    /// For a `[fields, widest]` padding of the concatenated options: the
    /// option at each padded slot, `total()` for a pad slot, and the padded
    /// slot of each option.
    pub(super) fn padding(&self) -> (Vec<i64>, Vec<i64>) {
        let total = self.total();
        let mut slots = Vec::with_capacity(self.counts.len() * self.widest);
        let mut real = Vec::with_capacity(total);
        let mut first = 0usize;
        for (q, &k) in self.counts.iter().enumerate() {
            slots.extend((first..first + k).map(|o| o as i64));
            slots.extend(std::iter::repeat_n(total as i64, self.widest - k));
            real.extend((0..k).map(|j| (q * self.widest + j) as i64));
            first += k;
        }
        (slots, real)
    }
}

/// `t` in `dtype`; a clone when it already is.
pub(super) fn cast_to<R, C>(client: &C, t: &Tensor<R>, dtype: DType) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    if t.dtype() == dtype {
        return Ok(t.clone());
    }
    Ok(client.cast(t, dtype)?)
}

/// The mean of the rows of `x` `[rows, D]` over each span, `[spans.len(),
/// D]` in `dtype`: the span rows are gathered, summed in F32 by one matmul
/// against a 0/1 owner matrix, divided by the span lengths, then cast.
/// Every span is non-empty and inside `x`.
pub(super) fn span_means<R, C>(
    client: &C,
    x: &Tensor<R>,
    spans: &[Range<usize>],
    dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let device = x.device();
    let total: usize = spans.iter().map(|s| s.len()).sum();
    let mut positions = Vec::with_capacity(total);
    let mut owner = vec![0.0f32; spans.len() * total];
    let mut lengths = Vec::with_capacity(spans.len());
    for (s, span) in spans.iter().enumerate() {
        for p in span.clone() {
            owner[s * total + positions.len()] = 1.0;
            positions.push(p as i64);
        }
        lengths.push(span.len() as f32);
    }
    let positions = Tensor::<R>::from_slice(&positions, &[total], device)?;
    let owner = Tensor::<R>::from_slice(&owner, &[spans.len(), total], device)?;
    let lengths = Tensor::<R>::from_slice(&lengths, &[spans.len(), 1], device)?;
    let rows = cast_to(client, &client.index_select(x, 0, &positions)?, DType::F32)?;
    let sums = client.matmul(&owner, &rows)?;
    cast_to(client, &client.div(&sums, &lengths)?, dtype)
}

/// `sum_d a[n, d] * b[n, d]` for each row, `[N]` in `a`'s dtype, products
/// and sum in F32.
pub(super) fn row_dots<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let a32 = cast_to(client, a, DType::F32)?;
    let b32 = cast_to(client, b, DType::F32)?;
    let dots = client.sum(&client.mul(&a32, &b32)?, &[1], false)?;
    cast_to(client, &dots, a.dtype())
}

/// Each row of `x` `[N, D]` over `max(||row||, eps)`, torch
/// `F.normalize`: the norm in F32, rounded to `x`'s dtype, the division in
/// `x`'s dtype.
pub(super) fn l2_normalize<R, C>(client: &C, x: &Tensor<R>, eps: f64) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let x32 = cast_to(client, x, DType::F32)?;
    let squares = client.sum(&client.mul(&x32, &x32)?, &[1], true)?;
    let norms = client.clamp(&client.sqrt(&squares)?, eps, f64::INFINITY)?;
    let norms = cast_to(client, &norms, x.dtype())?;
    Ok(client.div(x, &norms)?)
}

#[cfg(test)]
mod tests {
    use super::super::item::{FieldKind, JointOption};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn item(kind: FieldKind, question: Range<usize>, spans: &[Range<usize>]) -> JointSchemaItem {
        JointSchemaItem {
            kind,
            question,
            options: spans
                .iter()
                .map(|s| JointOption {
                    span: s.clone(),
                    ids: s.clone().map(|p| p as u32).collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn layout_concatenates_and_pads() {
        let items = [
            item(FieldKind::Choice, 0..2, &[2..4, 4..5, 5..8]),
            item(FieldKind::Noul, 8..9, &[9..10, 10..12]),
        ];
        let layout = Layout::new(&items);
        assert_eq!(layout.total(), 5);
        assert_eq!(layout.owner, vec![0, 0, 0, 1, 1]);
        assert_eq!(layout.counts, vec![3, 2]);
        assert_eq!(layout.types, vec![1, 0]);
        assert_eq!(layout.widest, 3);
        assert_eq!(layout.lexical, vec![0..2, 2..3, 3..6, 6..7, 7..9]);
        let (slots, real) = layout.padding();
        assert_eq!(slots, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(real, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn means_dots_and_norms() {
        let (client, device) = cpu_setup();
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&data, &[4, 3], &device).unwrap();
        let means = span_means(&client, &x, &[1..3, 0..1, 0..4], DType::F32).unwrap();
        assert_eq!(means.shape(), &[3, 3]);
        assert_eq!(
            means.to_vec::<f32>(),
            vec![4.5, 5.5, 6.5, 0.0, 1.0, 2.0, 4.5, 5.5, 6.5]
        );

        let a =
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 0.0, 0.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 5.0, 6.0], &[2, 2], &device).unwrap();
        assert_eq!(
            row_dots(&client, &a, &b).unwrap().to_vec::<f32>(),
            vec![11.0, 0.0]
        );
        let n = l2_normalize(&client, &a, 1e-12).unwrap().to_vec::<f32>();
        assert_eq!(n, vec![0.6, 0.8, 0.0, 0.0]);
    }
}
