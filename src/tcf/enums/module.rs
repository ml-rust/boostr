//! Enumerations decoded from `ModuleRecord` fields. FORMAT.md Section 7.

crate::define_enum_u16! {
    /// `ModuleRecord.module_role`. Section 7.1.
    pub enum ModuleRole as "module_role" {
        Other = 0,
        Embedding = 1,
        Attention = 2,
        Ffn = 3,
        Moe = 4,
        Ssm = 5,
        ConvStack = 6,
        Diffusion = 7,
        Codec = 8,
        NormGroup = 9,
    }
}

crate::define_enum_u16! {
    /// `ModuleRecord.state_dtype`: numeric type of recurrent state for
    /// stateful modules. Section 7.3. `None` means the module holds no recurrent
    /// state.
    pub enum StateDtype as "state_dtype" {
        None = 0,
        F32 = 1,
        F16 = 2,
        Bf16 = 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_role_roundtrips() {
        for raw in 0u16..=9 {
            let role = ModuleRole::try_from(raw).expect("defined value");
            assert_eq!(role.to_u16(), raw);
        }
        assert!(ModuleRole::try_from(10).is_err());
    }

    #[test]
    fn state_dtype_roundtrips() {
        assert_eq!(StateDtype::try_from(0), Ok(StateDtype::None));
        assert_eq!(StateDtype::try_from(1), Ok(StateDtype::F32));
        assert_eq!(StateDtype::try_from(2), Ok(StateDtype::F16));
        assert_eq!(StateDtype::try_from(3), Ok(StateDtype::Bf16));
        assert!(StateDtype::try_from(4).is_err());
    }
}
