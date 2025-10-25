#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Computes the dot product of two equal-length vectors.
///
/// When compiled for WebAssembly with the `wasm` feature enabled, the function
/// is exported via `wasm-bindgen` so that browsers can call it directly. For
/// native targets it remains a regular Rust function which enables parity tests
/// without a wasm toolchain.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot product requires vectors of identical length"
    );
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::dot;

    #[test]
    fn dot_matches_manual_computation() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [0.5, 1.5, 2.5, 3.5];
        assert!((dot(&left, &right) - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "identical length")]
    fn mismatched_lengths_panic() {
        let left = [1.0, 2.0];
        let right = [1.0];
        let _ = dot(&left, &right);
    }
}
