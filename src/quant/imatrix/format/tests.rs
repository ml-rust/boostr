//! Round trip, and every way a file is refused rather than misread.

use super::*;

fn entry(name: &str, rows: u64, sums: &[f32]) -> ImportanceEntry {
    ImportanceEntry {
        name: name.to_string(),
        rows,
        sums: sums.to_vec(),
    }
}

fn sample() -> ImportanceMatrix {
    let mut matrix = ImportanceMatrix::new(4096);
    matrix
        .insert(entry(
            "model.layers.0.mlp.down_proj.weight",
            512,
            &[1.0, 2.0],
        ))
        .unwrap();
    matrix
        .insert(entry(
            "model.layers.0.self_attn.q_proj.weight",
            512,
            &[3.0, 4.0, 5.0],
        ))
        .unwrap();
    matrix
}

#[test]
fn round_trip_preserves_every_field() {
    let matrix = sample();
    let parsed = ImportanceMatrix::from_bytes(&matrix.to_bytes()).unwrap();
    assert_eq!(parsed, matrix);
    assert_eq!(parsed.token_count(), 4096);
    assert_eq!(parsed.len(), 2);
}

#[test]
fn entries_are_written_in_name_order() {
    let matrix = sample();
    let ordered: Vec<&str> = matrix.entries().map(|e| e.name.as_str()).collect();
    let mut expected = ordered.clone();
    expected.sort_unstable();
    assert_eq!(ordered, expected);
}

#[test]
fn two_matrices_with_the_same_content_produce_identical_bytes() {
    // Insertion order differs; the file must not.
    let mut reversed = ImportanceMatrix::new(4096);
    reversed
        .insert(entry(
            "model.layers.0.self_attn.q_proj.weight",
            512,
            &[3.0, 4.0, 5.0],
        ))
        .unwrap();
    reversed
        .insert(entry(
            "model.layers.0.mlp.down_proj.weight",
            512,
            &[1.0, 2.0],
        ))
        .unwrap();
    assert_eq!(reversed.to_bytes(), sample().to_bytes());
}

#[test]
fn header_is_the_documented_length() {
    assert_eq!(
        sample().to_bytes()[..IMATRIX_MAGIC.len()],
        IMATRIX_MAGIC[..]
    );
    assert_eq!(IMATRIX_HEADER_LEN, 32);
}

#[test]
fn duplicate_name_is_refused() {
    let mut matrix = sample();
    let err = matrix
        .insert(entry("model.layers.0.mlp.down_proj.weight", 1, &[9.0]))
        .unwrap_err();
    assert!(err.to_string().contains("recorded twice"), "{err}");
}

#[test]
fn negative_and_non_finite_values_are_refused() {
    let mut matrix = ImportanceMatrix::new(1);
    assert!(matrix.insert(entry("a", 1, &[-1.0])).is_err());
    assert!(matrix.insert(entry("b", 1, &[f32::NAN])).is_err());
    assert!(matrix.insert(entry("c", 1, &[f32::INFINITY])).is_err());
}

#[test]
fn wrong_magic_is_refused() {
    let mut bytes = sample().to_bytes();
    bytes[0] = b'X';
    let err = ImportanceMatrix::from_bytes(&bytes).unwrap_err();
    assert!(err.to_string().contains("magic"), "{err}");
}

#[test]
fn unknown_version_is_refused() {
    let mut bytes = sample().to_bytes();
    bytes[8..12].copy_from_slice(&(IMATRIX_VERSION + 1).to_le_bytes());
    assert!(ImportanceMatrix::from_bytes(&bytes).is_err());
}

#[test]
fn set_reserved_flag_is_refused() {
    let mut bytes = sample().to_bytes();
    bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
    let err = ImportanceMatrix::from_bytes(&bytes).unwrap_err();
    assert!(err.to_string().contains("flags"), "{err}");
}

#[test]
fn truncated_file_is_refused() {
    let bytes = sample().to_bytes();
    for cut in [
        4,
        IMATRIX_HEADER_LEN,
        IMATRIX_HEADER_LEN + 8,
        bytes.len() - 1,
    ] {
        assert!(
            ImportanceMatrix::from_bytes(&bytes[..cut]).is_err(),
            "accepted a file truncated to {cut} byte(s)"
        );
    }
}

#[test]
fn trailing_bytes_are_refused() {
    let mut bytes = sample().to_bytes();
    bytes.push(0);
    let err = ImportanceMatrix::from_bytes(&bytes).unwrap_err();
    assert!(err.to_string().contains("trailing"), "{err}");
}

#[test]
fn a_width_disagreement_is_an_error_not_a_finding() {
    let expected = vec![("model.layers.0.mlp.down_proj.weight".to_string(), 3)];
    let err = sample().check_against(&expected).unwrap_err();
    assert!(err.to_string().contains("column"), "{err}");
}

#[test]
fn check_separates_present_absent_and_unknown() {
    let expected = vec![
        ("model.layers.0.mlp.down_proj.weight".to_string(), 2),
        ("model.layers.0.mlp.up_proj.weight".to_string(), 2),
    ];
    let check = sample().check_against(&expected).unwrap();
    assert_eq!(check.present, vec!["model.layers.0.mlp.down_proj.weight"]);
    assert_eq!(check.absent, vec!["model.layers.0.mlp.up_proj.weight"]);
    assert_eq!(
        check.unknown,
        vec!["model.layers.0.self_attn.q_proj.weight"]
    );
}

#[test]
fn an_unmeasured_tensor_has_no_entry_at_all() {
    let matrix = sample();
    assert!(matrix.get("model.layers.0.mlp.up_proj.weight").is_none());
}

#[test]
fn mean_square_divides_by_rows_and_refuses_an_empty_count() {
    assert_eq!(
        entry("a", 4, &[8.0, 4.0]).mean_square(),
        Some(vec![2.0, 1.0])
    );
    assert_eq!(entry("a", 0, &[8.0]).mean_square(), None);
}

#[test]
fn file_round_trips_through_a_path() {
    let dir = std::env::temp_dir().join(format!("boostr-imatrix-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("importance.bstrimtx");
    sample().write_to_path(&path).unwrap();
    assert_eq!(ImportanceMatrix::read_from_path(&path).unwrap(), sample());
    std::fs::remove_dir_all(&dir).ok();
}
