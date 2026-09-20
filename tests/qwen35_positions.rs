//! IMROPE position rule for `qwen35` prompts holding images, checked
//! against hand-derived `[4, S]` arrays.
//!
//! ```bash
//! cargo test --test qwen35_positions
//! ```

use boostr::model::qwen35::{ImageGrid, Segment, VisionMarkers, plan_positions_only};

const M: VisionMarkers = VisionMarkers {
    vision_start: 248_053,
    vision_end: 248_054,
    image_pad: 248_056,
};

const VS: u32 = M.vision_start;
const VE: u32 = M.vision_end;
const PAD: u32 = M.image_pad;

/// `[text, img(3x2), text, img(2x5), text]`.
const IDS: [u32; 11] = [10, 11, VS, PAD, VE, 12, VS, PAD, VE, 13, 14];
const GRIDS: [ImageGrid; 2] = [ImageGrid { nx: 3, ny: 2 }, ImageGrid { nx: 2, ny: 5 }];

#[test]
fn two_images_exact_positions() {
    let layout = plan_positions_only(&IDS, &GRIDS, &M, 0).unwrap();
    // 9 text rows + 6 + 10 image rows.
    assert_eq!(layout.seq_len, 25);
    // 10 11 VS | image 1 at 3, 3 + max(3, 2) = 6 | VE 12 VS at 6 7 8 |
    // image 2 at 9, 9 + max(2, 5) = 14 | VE 13 14 at 14 15 16.
    assert_eq!(layout.next_rope_pos, 17);
    assert_eq!(
        layout.segments,
        vec![
            Segment::Text { start: 0, end: 3 },
            Segment::Image { index: 0 },
            Segment::Text { start: 4, end: 7 },
            Segment::Image { index: 1 },
            Segment::Text { start: 8, end: 11 },
        ]
    );

    let t: [i32; 25] = [
        0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 14, 15, 16,
    ];
    let h: [i32; 25] = [
        0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 15, 16,
    ];
    let w: [i32; 25] = [
        0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 14, 15, 16,
    ];
    let e = [0i32; 25];
    assert_eq!(layout.stream(0), &t, "t stream");
    assert_eq!(layout.stream(1), &h, "h stream");
    assert_eq!(layout.stream(2), &w, "w stream");
    assert_eq!(layout.stream(3), &e, "e stream");

    let mut expected = Vec::with_capacity(100);
    expected.extend_from_slice(&t);
    expected.extend_from_slice(&h);
    expected.extend_from_slice(&w);
    expected.extend_from_slice(&e);
    assert_eq!(layout.positions, expected);
}

#[test]
fn start_offset_shifts_every_stream() {
    let base = plan_positions_only(&IDS, &GRIDS, &M, 0).unwrap();
    let shifted = plan_positions_only(&IDS, &GRIDS, &M, 100).unwrap();
    assert_eq!(shifted.next_rope_pos, base.next_rope_pos + 100);
    for s in 0..3 {
        let want: Vec<i32> = base.stream(s).iter().map(|p| p + 100).collect();
        assert_eq!(shifted.stream(s), &want[..], "stream {s}");
    }
    assert_eq!(shifted.stream(3), base.stream(3));
}

#[test]
fn text_only_equals_index() {
    let ids: Vec<u32> = (0..7).map(|i| 500 + i).collect();
    let layout = plan_positions_only(&ids, &[], &M, 0).unwrap();
    assert_eq!(layout.seq_len, 7);
    assert_eq!(layout.next_rope_pos, 7);
    let index: Vec<i32> = (0..7).collect();
    for s in 0..3 {
        assert_eq!(layout.stream(s), &index[..], "stream {s}");
    }
    assert_eq!(layout.stream(3), &[0; 7]);
}

#[test]
fn image_count_mismatch_names_both_counts() {
    let err = plan_positions_only(&IDS, &GRIDS[..1], &M, 0).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("2 image_pad"), "{text}");
    assert!(text.contains("1 images"), "{text}");
}
