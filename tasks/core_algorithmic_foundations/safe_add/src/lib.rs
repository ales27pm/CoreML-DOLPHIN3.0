//! Overflow-safe integer addition utilities validated with property-based tests.

/// Safely add two 64-bit integers, panicking on overflow.
#[must_use]
pub fn safe_add(a: i64, b: i64) -> i64 {
    a.checked_add(b).expect("overflow")
}

#[cfg(test)]
mod tests {
    use super::safe_add;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn addition_is_commutative(a in -1_000_000i64..1_000_000, b in -1_000_000i64..1_000_000) {
            prop_assert_eq!(safe_add(a, b), safe_add(b, a));
        }

        #[test]
        fn addition_inverse(a in -1_000_000i64..1_000_000) {
            prop_assert_eq!(safe_add(a, -a), 0);
        }
    }
}
