//! The Noul yes-probability map and the abstention band decision.

use crate::error::{Error, Result};

/// The Noul abstention band. `p_yes <= low` decides "no", `p_yes >= high`
/// decides "yes", and `low < p_yes < high` abstains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoulBand {
    /// Upper edge of "no" (inclusive).
    pub low: f64,
    /// Lower edge of "yes" (inclusive).
    pub high: f64,
}

impl Default for NoulBand {
    /// The 0.2 / 0.8 band.
    fn default() -> Self {
        Self {
            low: 0.2,
            high: 0.8,
        }
    }
}

/// The map applied to the Noul yes probability before the band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum NoulMap {
    /// Report the posterior unchanged.
    #[default]
    Raw,
    /// Commit every answer outside the 0.2 / 0.8 band:
    /// `p' = 0.8 + 0.1 * (p - 0.5)` for `p >= 0.5`, and the mirror
    /// `p' = 0.2 - 0.1 * (0.5 - p)` below 0.5. `p = 0.5` maps to 0.8.
    Commit,
}

impl NoulMap {
    /// The mapped yes probability.
    pub fn apply(self, p: f64) -> f64 {
        match self {
            Self::Raw => p,
            Self::Commit if p >= 0.5 => 0.8 + 0.1 * (p - 0.5),
            Self::Commit => 0.2 - 0.1 * (0.5 - p),
        }
    }
}

/// The decision for `p_yes`. With no band, `p_yes >= 0.5` is "yes". With a
/// band, `None` inside it. Errors on a band outside `0 <= low <= high <= 1`.
pub fn decide(p_yes: f64, band: Option<NoulBand>) -> Result<Option<bool>> {
    let Some(band) = band else {
        return Ok(Some(p_yes >= 0.5));
    };
    let valid = band.low.is_finite()
        && band.high.is_finite()
        && band.low >= 0.0
        && band.high <= 1.0
        && band.low <= band.high;
    if !valid {
        return Err(Error::InvalidArgument {
            arg: "noul_band",
            reason: format!("need 0 <= low <= high <= 1, got {band:?}"),
        });
    }
    Ok(if p_yes <= band.low {
        Some(false)
    } else if p_yes >= band.high {
        Some(true)
    } else {
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_band_decides_at_one_half() {
        assert_eq!(decide(0.5, None).unwrap(), Some(true));
        assert_eq!(decide(0.4999, None).unwrap(), Some(false));
    }

    #[test]
    fn band_edges_are_inclusive() {
        let b = Some(NoulBand::default());
        assert_eq!(decide(0.2, b).unwrap(), Some(false));
        assert_eq!(decide(0.8, b).unwrap(), Some(true));
        assert_eq!(decide(0.5, b).unwrap(), None);
        assert_eq!(decide(0.21, b).unwrap(), None);
        assert_eq!(decide(0.79, b).unwrap(), None);
        assert_eq!(decide(0.0, b).unwrap(), Some(false));
        assert_eq!(decide(1.0, b).unwrap(), Some(true));
    }

    #[test]
    fn invalid_band() {
        for (low, high) in [(0.8, 0.2), (-0.1, 0.8), (0.2, 1.1), (f64::NAN, 0.8)] {
            assert!(decide(0.5, Some(NoulBand { low, high })).is_err());
        }
    }

    #[test]
    fn commit_map_values() {
        let m = NoulMap::Commit;
        assert_eq!(m.apply(0.5), 0.8);
        assert!((m.apply(1.0) - 0.85).abs() < 1e-12);
        assert!((m.apply(0.0) - 0.15).abs() < 1e-12);
        assert!((m.apply(0.49) - 0.199).abs() < 1e-12);
        assert!((m.apply(0.7) - 0.82).abs() < 1e-12);
        assert_eq!(NoulMap::Raw.apply(0.37), 0.37);
    }

    #[test]
    fn commit_map_never_abstains() {
        let b = Some(NoulBand::default());
        for i in 0..=1000 {
            let p = NoulMap::Commit.apply(i as f64 / 1000.0);
            assert!(decide(p, b).unwrap().is_some(), "p = {p}");
        }
    }
}
