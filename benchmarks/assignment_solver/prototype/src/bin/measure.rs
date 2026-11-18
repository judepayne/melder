//! Reproducible std-only sparse assignment benchmark; emits JSON lines, not edge CSVs.

use std::env;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::time::Instant;

use assignment_solver::{Limits, Options, SCALE, WeightedEdge, quantize, solve_with_options};

const RECIPE: &str = "sparse-v1";
const DEFAULT_SEED: u64 = 0x005e_edb1;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64: wrapping arithmetic and iteration order are part of the recipe.
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn hash_word(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn generate(n: usize, degree: usize, shape: &str, seed: u64) -> Vec<WeightedEdge> {
    let mut rng = Rng(seed);
    let mut edges = Vec::new();
    let mut start = 0;
    let mut group = 0;
    while start < n {
        // Repeated, sharply skewed component sizes. Each group is connected by
        // its diagonal plus cyclic successor edges (except singleton groups).
        let width = if shape == "skewed" {
            [1, 4, 16, 64, 256, 1024][group % 6].min(n - start)
        } else {
            n
        };
        for local_b in 0..width {
            let b = start + local_b;
            let mut neighbors = Vec::new();
            if shape == "chain" {
                neighbors.push(local_b);
                if local_b + 1 < width {
                    neighbors.push(local_b + 1);
                }
            } else {
                let count = degree.min(width);
                neighbors.push(local_b);
                if count > 1 {
                    neighbors.push((local_b + 1) % width);
                }
                while neighbors.len() < count {
                    let a = (rng.next() % width as u64) as usize;
                    if !neighbors.contains(&a) {
                        neighbors.push(a);
                    }
                }
            }
            // Sort before drawing weights: this ordering is part of sparse-v1.
            neighbors.sort_unstable();
            for local_a in neighbors {
                let raw_units = if shape == "chain" {
                    if local_a == local_b {
                        750_000_000
                    } else {
                        500_000_000
                    }
                } else {
                    (rng.next() % 1_000_000_000 + 1) as i64
                };
                edges.push(WeightedEdge {
                    b,
                    a: start + local_a,
                    weight: raw_units,
                    payload: edges.len(),
                });
            }
        }
        start += width;
        group += 1;
    }
    edges
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        return Err(
            "usage: measure N DEGREE giant|skewed|chain [SEED_DECIMAL] [--workers N] [--batch-admissible] (N is vertices per side)"
                .into(),
        );
    }
    let n: usize = args[1].parse().map_err(|_| "invalid N")?;
    let degree: usize = args[2].parse().map_err(|_| "invalid DEGREE")?;
    let shape = args[3].as_str();
    let mut rest = args[4..].iter().peekable();
    let seed = match rest.peek() {
        Some(value) if !value.starts_with("--") => rest
            .next()
            .expect("peeked seed exists")
            .parse()
            .map_err(|_| "invalid SEED_DECIMAL")?,
        _ => DEFAULT_SEED,
    };
    let mut workers = NonZeroUsize::MIN;
    let mut batch_admissible = false;
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--workers" => {
                workers = rest
                    .next()
                    .ok_or("--workers requires a positive integer")?
                    .parse()
                    .map_err(|_| "--workers requires a positive integer")?;
            }
            "--batch-admissible" => batch_admissible = true,
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    if n == 0 || degree < 2 || !matches!(shape, "giant" | "skewed" | "chain") {
        return Err("N must be positive, DEGREE >= 2, shape giant|skewed|chain".into());
    }
    if SCALE != 1_000_000_000 {
        return Err("sparse-v1 requires SCALE = 1_000_000_000".into());
    }
    let start = Instant::now();
    let mut edges = generate(n, degree, shape, seed);
    let generation_s = start.elapsed().as_secs_f64();
    let start = Instant::now();
    for edge in &mut edges {
        let raw_units = edge.weight;
        edge.weight = quantize(raw_units as f64 / 1_000_000_000.0)
            .map_err(|error| format!("quantization: {error:?}"))?;
        if edge.weight != raw_units {
            return Err("raw integer units must round-trip through f64 and quantize".into());
        }
    }
    let conversion_s = start.elapsed().as_secs_f64();
    // FNV-1a64 is a recipe checksum, NOT a cryptographic identity. Hash the
    // dimensions, then each (b,a,raw-f64-bits,represented-weight) as LE u64s.
    // The integer-to-f64 conversion round-trips at this scale, so raw units
    // are recoverable from the represented weights without another edge array.
    let start = Instant::now();
    let mut digest = 0xcbf29ce484222325;
    hash_word(&mut digest, n as u64);
    hash_word(&mut digest, n as u64);
    for edge in &edges {
        hash_word(&mut digest, edge.b as u64);
        hash_word(&mut digest, edge.a as u64);
        hash_word(
            &mut digest,
            (edge.weight as f64 / 1_000_000_000.0).to_bits(),
        );
        hash_word(&mut digest, edge.weight as u64);
    }
    let digest_s = start.elapsed().as_secs_f64();
    println!(
        "{{\"event\":\"graph\",\"recipe\":\"{RECIPE}\",\"shape\":\"{shape}\",\"n_b\":{n},\"n_a\":{n},\"requested_degree\":{degree},\"seed\":{seed},\"edges\":{},\"scale\":{SCALE},\"raw_score_recipe\":\"integer units 1..1000000000 divided by 1e9 as f64; chain diagonal .75 successor .5\",\"conversion\":\"assignment_solver::quantize\",\"digest_algorithm\":\"fnv1a64-le-nb-na-then-b-a-raw_f64_bits-weight\",\"graph_digest\":\"{digest:016x}\",\"generation_s\":{generation_s},\"conversion_s\":{conversion_s},\"digest_s\":{digest_s}}}",
        edges.len(),
    );
    io::stdout().flush().map_err(|error| error.to_string())?;
    let start = Instant::now();
    let matching = solve_with_options(
        n,
        n,
        &edges,
        Limits::default(),
        Options {
            workers,
            batch_admissible,
        },
    )
    .map_err(|error| format!("solve: {error:?}"))?;
    let solve_s = start.elapsed().as_secs_f64();
    let validation_start = Instant::now();
    let mut used_b = vec![false; n];
    let mut used_a = vec![false; n];
    let mut objective = 0_i128;
    let mut selected_digest = 0xcbf29ce484222325;
    let mut payloads = matching.selected_payloads.clone();
    payloads.sort_unstable();
    for payload in &payloads {
        let edge = edges
            .get(*payload)
            .ok_or("selected payload outside input")?;
        if used_b[edge.b] || used_a[edge.a] {
            return Err("selected matching violates bijection".into());
        }
        used_b[edge.b] = true;
        used_a[edge.a] = true;
        objective += i128::from(edge.weight);
        hash_word(&mut selected_digest, *payload as u64);
    }
    if objective != matching.objective_units {
        return Err("selected objective does not match reported objective".into());
    }
    if shape == "chain" && objective != n as i128 * 750_000_000 {
        return Err("analytic chain objective mismatch".into());
    }
    let validation_s = validation_start.elapsed().as_secs_f64();
    let stats = &matching.stats;
    println!(
        "{{\"event\":\"result\",\"status\":\"ok\",\"workers\":{workers},\"batch_admissible\":{batch_admissible},\"solve_s\":{solve_s},\"solve_wall_s\":{solve_s},\"phase_timing_semantics\":\"summed component elapsed phases plus serial overhead; not parallel wall time\",\"validation_s\":{validation_s},\"selected\":{},\"selected_payload_digest\":\"{selected_digest:016x}\",\"objective_units\":\"{}\",\"components\":{},\"largest_component_vertices\":{},\"largest_component_edges\":{},\"augmentations\":{},\"shortest_path_searches\":{},\"batched_augmentations\":{},\"build_s\":{},\"search_s\":{},\"extract_s\":{}}}",
        payloads.len(),
        matching.objective_units,
        stats.components,
        stats.largest_component_vertices,
        stats.largest_component_edges,
        stats.augmentations,
        stats.shortest_path_searches,
        stats.batched_augmentations,
        stats.build_time.as_secs_f64(),
        stats.search_time.as_secs_f64(),
        stats.extract_time.as_secs_f64(),
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_preserve_fixture_objective() {
        let edges = generate(100, 5, "skewed", DEFAULT_SEED);
        let baseline = solve_with_options(
            100,
            100,
            &edges,
            Limits::default(),
            Options {
                workers: NonZeroUsize::MIN,
                batch_admissible: false,
            },
        )
        .expect("baseline fixture solves");
        for workers in [1, 2] {
            for batch_admissible in [false, true] {
                let matching = solve_with_options(
                    100,
                    100,
                    &edges,
                    Limits::default(),
                    Options {
                        workers: NonZeroUsize::new(workers).expect("positive test worker count"),
                        batch_admissible,
                    },
                )
                .expect("control fixture solves");
                assert_eq!(
                    matching.objective_units, baseline.objective_units,
                    "controls preserve objective: workers={workers}, batch={batch_admissible}",
                );
                assert!(
                    matching.stats.shortest_path_searches > 0,
                    "searches recorded"
                );
                if !batch_admissible {
                    assert_eq!(matching.stats.batched_augmentations, 0, "batching disabled");
                }
            }
        }
    }

    #[test]
    fn generator_is_unique_bounded_and_repeatable() {
        for shape in ["giant", "skewed", "chain"] {
            let first = generate(1100, 20, shape, DEFAULT_SEED);
            let second = generate(1100, 20, shape, DEFAULT_SEED);
            let mut pairs = std::collections::HashSet::new();
            assert_eq!(first.len(), second.len(), "same seed must reproduce edges");
            if shape == "giant" {
                assert_eq!(first.len(), 1100 * 20, "giant graph has exact degree");
            } else if shape == "chain" {
                assert_eq!(first.len(), 2 * 1100 - 1, "analytic chain edge count");
            }
            for (left, right) in first.iter().zip(&second) {
                assert_eq!(
                    quantize(left.weight as f64 / 1_000_000_000.0)
                        .expect("generated raw score is finite and in range"),
                    left.weight,
                    "generated raw score round-trips to its represented units",
                );
                assert!(left.b < 1100 && left.a < 1100, "bounded endpoints");
                assert!(pairs.insert((left.b, left.a)), "unique pair");
                assert_eq!(
                    (left.b, left.a, left.weight),
                    (right.b, right.a, right.weight),
                    "same seed reproduces raw weights"
                );
            }
        }
    }
}
