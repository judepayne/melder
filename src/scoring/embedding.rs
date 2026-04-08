//! Embedding scorer: cosine similarity of L2-normalized vectors.
//!
//! `fastembed` returns L2-normalized vectors, so dot product = cosine similarity.
//! We clamp to [0.0, 1.0] since negative cosine similarity is not meaningful
//! for our matching use case.

/// Dot product of two f32 slices — single implementation used everywhere.
///
/// When the `simd` feature is enabled, dispatches to SimSIMD's
/// hardware-accelerated inner product (NEON / SVE / AVX2 / AVX-512).
/// Otherwise falls back to an iterator loop that LLVM auto-vectorizes.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "dot_product_f32: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    #[cfg(feature = "simd")]
    {
        use simsimd::SpatialSimilarity;
        // simsimd returns Option<f64>; None only on length mismatch.
        f32::dot(a, b).unwrap_or(0.0) as f32
    }
    #[cfg(not(feature = "simd"))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Compute the dot product of two f32 slices, returning f64 for score
/// compatibility.
///
/// Thin wrapper around [`dot_product_f32`] with a widening cast.
pub fn dot_product(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");
    if a.len() != b.len() {
        return 0.0;
    }
    dot_product_f32(a, b) as f64
}

/// L2-normalize a vector in-place.
///
/// After normalization, the vector has unit length (L2 norm = 1.0).
/// If the vector has zero norm (all zeros), it is left unchanged.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two vectors, clamped to [0.0, 1.0].
///
/// Assumes both vectors are L2-normalized (as returned by fastembed).
/// For normalized vectors, dot product equals cosine similarity.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    dot_product(a, b).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;

    #[test]
    fn identical_vectors() {
        let v = vec![1.0_f32, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 0.001);
    }

    #[test]
    fn forty_five_degrees() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.707_f32, 0.707];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.707).abs() < 0.01, "expected ~0.707, got {}", sim);
    }

    #[test]
    fn l2_normalize_simple() {
        let mut v = vec![3.0_f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 0.001, "got {}", v[0]);
        assert!((v[1] - 0.8).abs() < 0.001, "got {}", v[1]);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut v = vec![0.0_f32, 0.0, 0.0];
        l2_normalize(&mut v);
        // Should remain zeros (no division by zero)
        assert!((v[0]).abs() < f32::EPSILON);
        assert!((v[1]).abs() < f32::EPSILON);
        assert!((v[2]).abs() < f32::EPSILON);
    }

    #[test]
    fn l2_normalize_already_unit() {
        let mut v = vec![1.0_f32, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 0.001);
        assert!((v[1]).abs() < 0.001);
    }

    #[test]
    fn dot_product_simple() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let dp = dot_product(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((dp - 32.0).abs() < 0.001, "got {}", dp);
    }

    #[test]
    fn cosine_clamps_negative() {
        // Opposite vectors: cosine = -1.0, should be clamped to 0.0
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 0.001);
    }

    #[test]
    fn fastembed_self_similarity() {
        // Verify with actual fastembed vectors
        let pool = crate::encoder::EncoderPool::new(crate::encoder::EncoderOptions {
            model_name: "all-MiniLM-L6-v2".to_string(),
            pool_size: 1,
            quantized: false,
            gpu: false,
            encode_batch_size: None,
        })
        .expect("pool creation failed");
        let vecs = pool
            .encode(&["hello world", "hello world"])
            .expect("encode failed");

        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "self-similarity = {}, expected ~1.0",
            sim
        );
    }
}
