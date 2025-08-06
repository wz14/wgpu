/// Library exports for NTT polynomial benchmark
/// 
/// This module provides the core functionality for benchmarking polynomial multiplication
/// using Number Theoretic Transform (NTT) on GPU compute shaders.

pub mod polynomial_benchmark;
pub mod validation;

pub use polynomial_benchmark::{NTTPolynomialMultiplier, BigInt260, generate_random_polynomial, POLY_DEGREE};
pub use validation::{cpu_polynomial_multiply, generate_test_polynomial, polynomials_approximately_equal};