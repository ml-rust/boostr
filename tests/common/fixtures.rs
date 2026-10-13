//! Model-fixture resolution and the standard skip notice.

use std::path::PathBuf;

/// Resolve a model fixture: `$<specific_var>`, else `$BOOSTR_MODELS_DIR/<relative>`
/// (pass `relative = ""` when the specific var's fallback is the models root
/// itself). Returns `None` when neither is set or the resolved path is absent
/// on disk, so callers skip.
pub fn model_fixture(specific_var: &str, relative: &str) -> Option<PathBuf> {
    let path = match std::env::var(specific_var) {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => {
            let root = std::env::var("BOOSTR_MODELS_DIR")
                .ok()
                .filter(|r| !r.is_empty())?;
            let root = PathBuf::from(root);
            if relative.is_empty() {
                root
            } else {
                root.join(relative)
            }
        }
    };
    path.exists().then_some(path)
}

/// Print the standard skip notice naming both the specific env var and
/// `BOOSTR_MODELS_DIR`, so the reader knows how to enable the test.
pub fn skip_notice(what: &str, specific_var: &str) {
    eprintln!("skipping: {what} unavailable; set {specific_var} or BOOSTR_MODELS_DIR");
}
