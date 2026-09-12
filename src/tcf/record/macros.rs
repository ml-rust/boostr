//! `define_record!`: one invocation per TCF v1 record, listing every field
//! AND every reserved range in a single ordered table with literal offsets
//! taken verbatim from FORMAT.md.
//!
//! One table generates four things that cannot drift apart:
//!
//! - the struct (a `reserved` row produces no field),
//! - `decode`, bounds-checked and reserved-checked,
//! - `encode`, which writes every reserved range as zero,
//! - `LAYOUT`, a const `(offset, size, name)` array.
//!
//! `LAYOUT` is then walked by a const block at compile time: a gap, an
//! overlap, a typo'd offset, or a wrong total size is a build error, not a
//! test failure. `tools/verify.py` closes the remaining gap by diffing the
//! same table against the record byte-tables in FORMAT.md, so a
//! consistently-copied transcription error cannot pass either.

/// Generate a TCF v1 record from its spec offset table. See the module docs.
///
/// Row forms, in spec order:
///
/// ```text
/// field  name: Type => offset, size;                    a plain field
/// flags  name: Type => offset, size;                    unknown bits rejected (Section 8.1.5)
/// strref name: (name_off => offset, name_len => offset); a string-table pair (Section 6)
/// reserved => offset, size;                             MUST be zero (Section 4)
/// ```
///
/// `mode = record` implements [`crate::tcf::record::traits::Record`];
/// `mode = singleton` emits the same method names as inherent items, for
/// `Header` (Section 5).
#[macro_export]
macro_rules! define_record {
    (
        $(#[$meta:meta])*
        pub struct $Name:ident;
        size = $size:expr;
        mode = $mode:ident;
        validate = $validate:path;
        rows { $($rows:tt)* }
    ) => {
        $crate::define_record!(@munch
            [$(#[$meta])*] $Name, $size, $mode, $validate, io(bytes, out),
            fields {} names {} decode {} encode {} layout {}
            rest { $($rows)* }
        );
    };

    (
        $(#[$meta:meta])*
        pub struct $Name:ident;
        size = $size:expr;
        mode = $mode:ident;
        rows { $($rows:tt)* }
    ) => {
        $crate::define_record!(
            $(#[$meta])*
            pub struct $Name;
            size = $size;
            mode = $mode;
            validate = $crate::tcf::record::field::no_validate;
            rows { $($rows)* }
        );
    };

    // --- row: a plain field -------------------------------------------------
    (@munch
        [$(#[$meta:meta])*] $Name:ident, $size:expr, $mode:ident, $validate:path, io($b:ident, $o:ident),
        fields { $($f:tt)* } names { $($nm:tt)* } decode { $($d:tt)* } encode { $($e:tt)* } layout { $($l:tt)* }
        rest { $(#[$fmeta:meta])* field $fname:ident: $Ty:ty => $off:expr, $len:expr; $($rest:tt)* }
    ) => {
        $crate::define_record!(@munch
            [$(#[$meta])*] $Name, $size, $mode, $validate, io($b, $o),
            fields { $($f)* $(#[$fmeta])* pub $fname: $Ty, }
            names { $($nm)* $fname, }
            decode {
                $($d)*
                const _: () = assert!(
                    <$Ty as $crate::tcf::record::field::RecordField>::LEN == $len,
                    concat!(stringify!($Name), ".", stringify!($fname), " width differs from the spec table"),
                );
                let $fname = <$Ty as $crate::tcf::record::field::RecordField>::read(
                    $b, $off, stringify!($Name))?;
            }
            encode {
                $($e)*
                $crate::tcf::record::field::RecordField::write($fname, $o, $off, stringify!($Name))?;
            }
            layout { $($l)* ($off, $len, stringify!($fname)), }
            rest { $($rest)* }
        );
    };

    // --- row: a flag field, unknown bits rejected (Section 8.1.5) -------------------
    (@munch
        [$(#[$meta:meta])*] $Name:ident, $size:expr, $mode:ident, $validate:path, io($b:ident, $o:ident),
        fields { $($f:tt)* } names { $($nm:tt)* } decode { $($d:tt)* } encode { $($e:tt)* } layout { $($l:tt)* }
        rest { $(#[$fmeta:meta])* flags $fname:ident: $Ty:ty => $off:expr, $len:expr; $($rest:tt)* }
    ) => {
        $crate::define_record!(@munch
            [$(#[$meta])*] $Name, $size, $mode, $validate, io($b, $o),
            fields { $($f)* $(#[$fmeta])* pub $fname: $Ty, }
            names { $($nm)* $fname, }
            decode {
                $($d)*
                const _: () = assert!(
                    <$Ty as $crate::tcf::record::field::RecordFlags>::LEN == $len,
                    concat!(stringify!($Name), ".", stringify!($fname), " width differs from the spec table"),
                );
                let $fname = <$Ty as $crate::tcf::record::field::RecordFlags>::read_checked(
                    $b,
                    $off,
                    stringify!($Name),
                    concat!(stringify!($Name), ".", stringify!($fname)),
                )?;
            }
            encode {
                $($e)*
                $crate::tcf::record::field::RecordFlags::write($fname, $o, $off, stringify!($Name))?;
            }
            layout { $($l)* ($off, $len, stringify!($fname)), }
            rest { $($rest)* }
        );
    };

    // --- row: a string-table (off, len) pair, two spec rows, one field (Section 6) --
    (@munch
        [$(#[$meta:meta])*] $Name:ident, $size:expr, $mode:ident, $validate:path, io($b:ident, $o:ident),
        fields { $($f:tt)* } names { $($nm:tt)* } decode { $($d:tt)* } encode { $($e:tt)* } layout { $($l:tt)* }
        rest {
            $(#[$fmeta:meta])*
            strref $fname:ident: ($offname:ident => $off:expr, $lenname:ident => $lenoff:expr);
            $($rest:tt)*
        }
    ) => {
        $crate::define_record!(@munch
            [$(#[$meta])*] $Name, $size, $mode, $validate, io($b, $o),
            fields { $($f)* $(#[$fmeta])* pub $fname: $crate::tcf::record::field::StringRef, }
            names { $($nm)* $fname, }
            decode {
                $($d)*
                let $fname = $crate::tcf::record::field::StringRef::read(
                    $b, $off, $lenoff, stringify!($Name))?;
            }
            encode {
                $($e)*
                $crate::tcf::record::field::StringRef::write($fname, $o, $off, $lenoff, stringify!($Name))?;
            }
            layout { $($l)* ($off, 8, stringify!($offname)), ($lenoff, 4, stringify!($lenname)), }
            rest { $($rest)* }
        );
    };

    // --- row: a reserved range, MUST be zero (Section 4) ----------------------------
    (@munch
        [$(#[$meta:meta])*] $Name:ident, $size:expr, $mode:ident, $validate:path, io($b:ident, $o:ident),
        fields { $($f:tt)* } names { $($nm:tt)* } decode { $($d:tt)* } encode { $($e:tt)* } layout { $($l:tt)* }
        rest { reserved => $off:expr, $len:expr; $($rest:tt)* }
    ) => {
        $crate::define_record!(@munch
            [$(#[$meta])*] $Name, $size, $mode, $validate, io($b, $o),
            fields { $($f)* }
            names { $($nm)* }
            decode {
                $($d)*
                $crate::tcf::record::field::expect_zero(
                    $b,
                    $off,
                    $len,
                    concat!(stringify!($Name), ".reserved@", stringify!($off)),
                )?;
            }
            encode {
                $($e)*
                $crate::tcf::record::field::zero_range($o, $off, $len, stringify!($Name))?;
            }
            layout { $($l)* ($off, $len, "reserved"), }
            rest { $($rest)* }
        );
    };

    // --- end: an array record, implementing `Record` -------------------------
    (@munch
        [$(#[$meta:meta])*] $Name:ident, $size:expr, record, $validate:path, io($b:ident, $o:ident),
        fields { $($f:tt)* } names { $($nm:ident,)* } decode { $($d:tt)* } encode { $($e:tt)* } layout { $($l:tt)* }
        rest {}
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $Name { $($f)* }

        impl $Name {
            /// This record's `(offset, size, field)` table, verbatim from
            /// FORMAT.md. Walked by the const block below at compile
            /// time and diffed against the spec by `tools/verify.py`.
            pub const LAYOUT: &'static [(usize, usize, &'static str)] = &[ $($l)* ];
        }

        $crate::define_record!(@layout_check $Name, $size);

        impl $crate::tcf::record::traits::Record for $Name {
            const SIZE: usize = $size;

            fn decode($b: &[u8]) -> Result<Self, $crate::tcf::error::TcfError> {
                if $b.len() < $size {
                    return Err($crate::tcf::error::TcfError::SectionBounds {
                        section: stringify!($Name),
                    });
                }
                $($d)*
                let record = Self { $($nm,)* };
                $validate(&record)?;
                Ok(record)
            }

            fn encode(&self, $o: &mut [u8]) -> Result<(), $crate::tcf::error::TcfError> {
                if $o.len() < $size {
                    return Err($crate::tcf::error::TcfError::SectionBounds {
                        section: stringify!($Name),
                    });
                }
                let Self { $($nm,)* } = self;
                $($e)*
                Ok(())
            }
        }
    };

    // --- end: the singleton header, same methods as inherent items -----------
    (@munch
        [$(#[$meta:meta])*] $Name:ident, $size:expr, singleton, $validate:path, io($b:ident, $o:ident),
        fields { $($f:tt)* } names { $($nm:ident,)* } decode { $($d:tt)* } encode { $($e:tt)* } layout { $($l:tt)* }
        rest {}
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $Name { $($f)* }

        impl $Name {
            /// Exact on-disk size in bytes.
            pub const SIZE: usize = $size;

            /// This record's `(offset, size, field)` table, verbatim from
            /// FORMAT.md.
            pub const LAYOUT: &'static [(usize, usize, &'static str)] = &[ $($l)* ];

            /// Decode from the first `SIZE` bytes of `bytes`.
            ///
            /// # Errors
            /// `TcfError::SectionBounds` on a short slice,
            /// `TcfError::NonzeroReserved` on a non-zero reserved byte or an
            /// unknown flag bit, plus whatever the post-decode hook rejects.
            pub fn decode($b: &[u8]) -> Result<Self, $crate::tcf::error::TcfError> {
                if $b.len() < $size {
                    return Err($crate::tcf::error::TcfError::SectionBounds {
                        section: stringify!($Name),
                    });
                }
                $($d)*
                let record = Self { $($nm,)* };
                $validate(&record)?;
                Ok(record)
            }

            /// Encode into the first `SIZE` bytes of `out`, writing every
            /// reserved range as zero.
            ///
            /// # Errors
            /// `TcfError::SectionBounds` when `out` is shorter than `SIZE`.
            pub fn encode(&self, $o: &mut [u8]) -> Result<(), $crate::tcf::error::TcfError> {
                if $o.len() < $size {
                    return Err($crate::tcf::error::TcfError::SectionBounds {
                        section: stringify!($Name),
                    });
                }
                let Self { $($nm,)* } = self;
                $($e)*
                Ok(())
            }
        }

        $crate::define_record!(@layout_check $Name, $size);
    };

    // --- the compile-time layout walk ---------------------------------------
    (@layout_check $Name:ident, $size:expr) => {
        const _: () = {
            let layout = $Name::LAYOUT;
            let mut cursor = 0usize;
            let mut i = 0usize;
            while i < layout.len() {
                assert!(
                    layout[i].0 == cursor,
                    concat!(stringify!($Name), ": a row's offset leaves a gap or an overlap"),
                );
                cursor += layout[i].1;
                i += 1;
            }
            assert!(
                cursor == $size,
                concat!(stringify!($Name), ": the rows do not sum to the spec record size"),
            );
        };
    };
}
