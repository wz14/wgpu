# NTT Polynomial Multiplication Benchmark

This is a standalone Rust application that demonstrates polynomial multiplication using Number Theoretic Transform (NTT) on GPU compute shaders with wgpu.

## Features

- **Large Polynomial Multiplication**: Handles polynomials of degree 8192 
- **Large Modulus Support**: Designed for modulus up to 2^260 using multi-precision arithmetic
- **GPU Acceleration**: Uses wgpu compute shaders for parallel processing
- **Comprehensive Benchmarking**: Includes performance measurement and analysis
- **CPU Fallback**: Graceful fallback to CPU validation when GPU is unavailable

## Quick Start

```bash
# Build the project
cargo build --release

# Run with GPU acceleration (when available)
cargo run --release

# Run with CPU validation only (works in any environment)
cargo run --release -- --cpu-only

# Run full benchmark suite
cargo bench
```

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
- WGSL compute shaders for all arithmetic operations

## Command Line Options

- `--cpu-only` or `--validation`: Run CPU-based validation instead of GPU
- No arguments: Attempt GPU computation, fall back to CPU if GPU unavailable

## Example Output

### GPU Mode (when available)
```
=== NTT Polynomial Multiplication Benchmark ===
Polynomial degree: 8192
Modulus representation: 5 x 64-bit components

Running on: AdapterInfo { name: "...", vendor: ..., device: ..., device_type: DiscreteGpu, ... }
Generating random polynomials...
Starting polynomial multiplication...
GPU computation took: 15.2ms
✓ Multiplication completed!
Total execution time: 28.5ms
Performance Analysis:
- Effective operations per second: 2.35e+09
- Time per coefficient: 3.48 µs
```

### CPU Validation Mode
```
=== NTT Polynomial Multiplication Benchmark ===
Polynomial degree: 8192
Modulus representation: 5 x 64-bit components

Running CPU-based validation for mathematical correctness...
Testing polynomial multiplication with degree 8:
  CPU multiplication time: 401ns
  ✓ Basic validation passed
...
=== CPU Validation Summary ===
✓ Mathematical operations working correctly  
✓ Polynomial multiplication logic verified
✓ Data structures and memory layout confirmed
```

## GPU Shader Implementation

The WGSL compute shaders implement:
- Multi-precision addition, subtraction, and multiplication
- Bit-reverse permutation for NTT
- Cooley-Tukey NTT algorithm
- Forward and inverse NTT transforms
- Pointwise multiplication in evaluation domain

Key shader functions:
- `bigint_add`, `bigint_sub`, `bigint_mod_mul`: Multi-precision arithmetic
- `forward_ntt`, `inverse_ntt`: NTT transforms
- `pointwise_multiply`: Element-wise multiplication

## Performance Characteristics

The benchmark measures:
- Total execution time for polynomial multiplication
- Operations per second
- Time per coefficient
- GPU memory transfer overhead
- Compute shader execution time

For polynomials of degree 8192, typical performance on modern GPUs can achieve billions of operations per second, significantly outperforming CPU implementations for large problem sizes.

## Modulus Considerations for 2^260

While the example uses a practical modulus for demonstration, the architecture is designed to scale to 2^260:

1. **Prime Selection**: For NTT to work, the modulus must be prime and satisfy p ≡ 1 (mod 2n)
2. **Root of Unity**: Must find a primitive nth root of unity modulo p
3. **Montgomery Form**: Large moduli benefit from Montgomery reduction for efficiency

### Finding a Suitable Prime near 2^260

```rust
// Example approach for finding NTT-friendly prime near 2^260
// p ≡ 1 (mod 2^14) for 8192-point NTT
// p ≈ 2^260

let base = BigInt::from(2).pow(260);
let k = 2_u64.pow(14); // 16384, required for 8192-point NTT

// Search for prime p = base - i*k + 1
for i in 0.. {
    let candidate = base - i * k + 1;
    if is_prime(&candidate) && has_primitive_root(&candidate, 8192) {
        return candidate; // Found suitable NTT prime
    }
}
```

## Testing

```bash
# Run unit tests
cargo test

# Run specific test
cargo test cpu_polynomial_multiplication

# Run with output
cargo test -- --nocapture
```

## Architecture

```
examples/standalone/ntt_polynomial_benchmark/
├── src/
│   ├── main.rs                 # Main binary with CLI
│   ├── lib.rs                  # Library exports  
│   ├── polynomial_benchmark.rs # Core NTT implementation
│   ├── ntt_shaders.wgsl       # GPU compute shaders
│   └── validation.rs          # CPU reference implementation
├── benches/
│   └── ntt_benchmark.rs       # Criterion benchmarks
├── Cargo.toml                 # Project configuration
└── README.md                  # This documentation
```

## Future Enhancements

- [ ] True 2^260 modulus with proper prime finding
- [ ] Barrett reduction for large moduli
- [ ] Montgomery ladder for modular exponentiation
- [ ] Batch processing for multiple polynomial pairs
- [ ] Memory pool optimization
- [ ] Cross-platform GPU backend selection
- [ ] WebAssembly target support
- [ ] Multi-GPU distribution