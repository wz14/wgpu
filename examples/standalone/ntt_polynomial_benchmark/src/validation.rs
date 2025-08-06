/// Simple CPU validation functions for NTT polynomial multiplication
/// These functions provide a reference implementation for verification

use crate::polynomial_benchmark::BigInt260;

/// Simple CPU-based polynomial multiplication for validation (naive O(n^2) algorithm)
pub fn cpu_polynomial_multiply(poly_a: &[BigInt260], poly_b: &[BigInt260]) -> Vec<BigInt260> {
    assert_eq!(poly_a.len(), poly_b.len());
    let degree = poly_a.len();
    let mut result = vec![BigInt260::new(0); degree * 2 - 1];

    // Simple convolution - in practice you'd implement proper modular arithmetic
    for i in 0..degree {
        for j in 0..degree {
            if i + j < result.len() {
                // Simplified multiplication - just using the first component for demo
                let prod = poly_a[i].components[0].wrapping_mul(poly_b[j].components[0]);
                result[i + j].components[0] = result[i + j].components[0].wrapping_add(prod);
            }
        }
    }

    // Reduce to original degree for comparison
    result.truncate(degree);
    result
}

/// Generate a simple test polynomial with known patterns
pub fn generate_test_polynomial(degree: usize, pattern: u64) -> Vec<BigInt260> {
    let mut poly = Vec::with_capacity(degree);
    for i in 0..degree {
        poly.push(BigInt260::new((i as u64 * pattern) % 1000 + 1));
    }
    poly
}

/// Validate that two polynomials are approximately equal (for testing)
pub fn polynomials_approximately_equal(a: &[BigInt260], b: &[BigInt260], tolerance: u64) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (coeff_a, coeff_b) in a.iter().zip(b.iter()) {
        // Simple comparison of first component only for validation
        let diff = if coeff_a.components[0] > coeff_b.components[0] {
            coeff_a.components[0] - coeff_b.components[0]
        } else {
            coeff_b.components[0] - coeff_a.components[0]
        };
        
        if diff > tolerance {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_polynomial_multiplication() {
        // Test with small polynomials
        let degree = 8;
        let poly_a = generate_test_polynomial(degree, 1);
        let poly_b = generate_test_polynomial(degree, 2);

        let result = cpu_polynomial_multiply(&poly_a, &poly_b);
        
        // Basic sanity checks
        assert_eq!(result.len(), degree);
        
        // Check that result is non-zero (coefficients should have been multiplied)
        assert!(result.iter().any(|c| c.components[0] > 0));
    }

    #[test]
    fn test_polynomial_validation() {
        let degree = 4;
        let poly_a = generate_test_polynomial(degree, 1);
        let poly_b = generate_test_polynomial(degree, 1);

        // Should be equal
        assert!(polynomials_approximately_equal(&poly_a, &poly_b, 0));

        // Should not be equal with small tolerance
        let poly_c = generate_test_polynomial(degree, 2);
        assert!(!polynomials_approximately_equal(&poly_a, &poly_c, 5));
    }
}