/// Main binary for NTT polynomial multiplication benchmark
/// 
/// This runs the benchmark as a standalone application demonstrating
/// large polynomial multiplication using wgpu compute shaders.

use wgpu_example_ntt_polynomial_benchmark::polynomial_benchmark::{NTTPolynomialMultiplier, generate_random_polynomial, POLY_DEGREE};

#[tokio::main]
async fn main() {
    env_logger::init();
    
    println!("=== NTT Polynomial Multiplication Benchmark ===");
    println!("Polynomial degree: {}", POLY_DEGREE);
    println!("Modulus representation: {} x 64-bit components", 5);
    println!();

    let multiplier = NTTPolynomialMultiplier::new().await;
    
    // Generate test polynomials
    println!("Generating random polynomials...");
    let poly_a = generate_random_polynomial(POLY_DEGREE);
    let poly_b = generate_random_polynomial(POLY_DEGREE);

    println!("Starting polynomial multiplication...");
    let start_time = std::time::Instant::now();
    let result = multiplier.multiply_polynomials(&poly_a, &poly_b);
    let total_time = start_time.elapsed();

    println!("✓ Multiplication completed!");
    println!("Total execution time: {:?}", total_time);
    println!("Result polynomial has {} coefficients", result.len());
    
    // Verify result (basic sanity check)
    println!("First few result coefficients:");
    for (i, coeff) in result.iter().take(5).enumerate() {
        println!("  coeff[{}] = {:?}", i, coeff.components);
    }

    // Performance analysis
    let ops_per_second = (POLY_DEGREE as f64 * POLY_DEGREE as f64) / total_time.as_secs_f64();
    println!();
    println!("Performance Analysis:");
    println!("- Effective operations per second: {:.2e}", ops_per_second);
    println!("- Time per coefficient: {:.2} µs", total_time.as_micros() as f64 / POLY_DEGREE as f64);
}