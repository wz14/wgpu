use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use wgpu_example_ntt_polynomial_benchmark::{NTTPolynomialMultiplier, generate_random_polynomial};

fn benchmark_ntt_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT Polynomial Multiplication");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    // Test different polynomial degrees
    let degrees = vec![1024, 2048, 4096, 8192];

    for degree in degrees {
        group.bench_with_input(
            BenchmarkId::new("GPU NTT", degree),
            &degree,
            |b, &degree| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let multiplier = rt.block_on(async { NTTPolynomialMultiplier::new().await });

                b.iter(|| {
                    let poly_a = generate_random_polynomial(degree);
                    let poly_b = generate_random_polynomial(degree);
                    multiplier.multiply_polynomials(&poly_a, &poly_b)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_ntt_multiplication);
criterion_main!(benches);