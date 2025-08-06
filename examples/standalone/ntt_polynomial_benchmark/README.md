# NTT Polynomial Multiplication Benchmark

This is a standalone Rust application that demonstrates polynomial multiplication using Number Theoretic Transform (NTT) on GPU compute shaders with wgpu.

## Features

- **Large Polynomial Multiplication**: Handles polynomials of degree 8192 
- **Large Modulus Support**: Designed for modulus up to 2^260 using multi-precision arithmetic
- **GPU Acceleration**: Uses wgpu compute shaders for parallel processing
- **Comprehensive Benchmarking**: Includes performance measurement and analysis

## Technical Details

### Multi-Precision Arithmetic
- Uses 5 x 64-bit components to represent integers up to 2^320 (sufficient for 2^260 modulus)
- Implements modular arithmetic operations in GPU shaders
- Supports Montgomery multiplication for efficient modular operations

### NTT Implementation
- Forward Number Theoretic Transform for coefficient to evaluation conversion
- Pointwise multiplication in evaluation domain
- Inverse NTT for conversion back to coefficient representation
- Uses precomputed twiddle factors (roots of unity)

### GPU Compute Architecture
- Workgroup size optimized for GPU architectures (64 threads per workgroup)
- Memory-efficient buffer management
- Proper synchronization barriers for parallel algorithms

## Building and Running

```bash
# Build the project
cargo build --release

# Run the benchmark application
cargo run --release

# Run criterion benchmarks
cargo bench
```

## Usage Example

```rust
use wgpu_example_ntt_polynomial_benchmark::{
    NTTPolynomialMultiplier, generate_random_polynomial, POLY_DEGREE
};

#[tokio::main]
async fn main() {
    let multiplier = NTTPolynomialMultiplier::new().await;
    
    let poly_a = generate_random_polynomial(POLY_DEGREE);
    let poly_b = generate_random_polynomial(POLY_DEGREE);
    
    let result = multiplier.multiply_polynomials(&poly_a, &poly_b);
    println!("Multiplication completed: {} coefficients", result.len());
}
```

## Performance Characteristics

The benchmark measures:
- Total execution time for polynomial multiplication
- Operations per second 
- Time per coefficient
- GPU memory transfer overhead
- Compute shader execution time

For polynomials of degree 8192, typical performance on modern GPUs can achieve millions of operations per second, significantly outperforming CPU implementations for large problem sizes.

## Modulus Considerations

While the example uses a practical modulus for demonstration, the architecture is designed to scale to 2^260:

1. **Prime Selection**: For NTT to work, the modulus must be prime and satisfy p ≡ 1 (mod 2n)
2. **Root of Unity**: Must find a primitive nth root of unity modulo p  
3. **Montgomery Form**: Large moduli benefit from Montgomery reduction for efficiency

## Future Enhancements

- [ ] True 2^260 modulus with proper prime finding
- [ ] Montgomery ladder for modular exponentiation
- [ ] Batch processing for multiple polynomial pairs
- [ ] Memory pool optimization
- [ ] Cross-platform GPU backend selection