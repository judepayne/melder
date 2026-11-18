//! Independent optional-matching oracles and deterministic graph fixtures.
//!
//! Seeds, case indices, and complete failing inputs are included in assertions so
//! failures can be reproduced without a generated fixture or external oracle.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use super::{Limits, Options, SCALE, WeightedEdge, quantize, solve_with_options};

// Deliberately no residual graph, potentials, quantization, or solver helpers:
// enumerate the unmatched branch and every unused receiver for each proposer.
fn oracle(n_b: usize, n_a: usize, edges: &[WeightedEdge]) -> i128 {
    fn visit(b: usize, rows: &[Vec<(usize, i64)>], used: &mut [bool]) -> i128 {
        if b == rows.len() {
            return 0;
        }
        let mut best = visit(b + 1, rows, used);
        for &(a, weight) in &rows[b] {
            if !used[a] {
                used[a] = true;
                best = best.max(i128::from(weight) + visit(b + 1, rows, used));
                used[a] = false;
            }
        }
        best
    }
    let mut rows = vec![Vec::new(); n_b];
    for edge in edges {
        rows[edge.b].push((edge.a, edge.weight));
    }
    visit(0, &rows, &mut vec![false; n_a])
}

fn edge(b: usize, a: usize, weight: i64, payload: usize) -> WeightedEdge {
    WeightedEdge {
        b,
        a,
        weight,
        payload,
    }
}

fn checked(
    n_b: usize,
    n_a: usize,
    edges: &[WeightedEdge],
    expected: i128,
    context: &str,
    options: Options,
) -> BTreeSet<(usize, usize)> {
    let result = solve_with_options(n_b, n_a, edges, Limits::default(), options)
        .unwrap_or_else(|error| panic!("{context}: solve failed: {error:?}; graph={edges:?}"));
    let mut bs = BTreeSet::new();
    let mut aas = BTreeSet::new();
    let mut payloads = BTreeSet::new();
    let mut pairs = BTreeSet::new();
    let mut sum = 0_i128;
    for payload in &result.selected_payloads {
        assert!(
            payloads.insert(*payload),
            "{context}: repeated payload {payload}; {edges:?}"
        );
        let selected = edges
            .iter()
            .find(|e| e.payload == *payload)
            .unwrap_or_else(|| panic!("{context}: unknown payload {payload}; {edges:?}"));
        assert!(
            selected.b < n_b && selected.a < n_a,
            "{context}: endpoint out of range"
        );
        assert!(bs.insert(selected.b), "{context}: B reused; {edges:?}");
        assert!(aas.insert(selected.a), "{context}: A reused; {edges:?}");
        assert!(
            pairs.insert((selected.b, selected.a)),
            "{context}: pair repeated"
        );
        sum += i128::from(selected.weight);
    }
    assert_eq!(
        sum, result.objective_units,
        "{context}: extraction sum; {edges:?}"
    );
    assert_eq!(sum, expected, "{context}: oracle objective; {edges:?}");
    pairs
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // Fixed xorshift64 sequence; no platform- or dependency-specific RNG.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            values.swap(i, self.below(i + 1));
        }
    }
}

fn variants(
    n_b: usize,
    n_a: usize,
    edges: &mut [WeightedEdge],
    expected: i128,
    rng: &mut Rng,
    context: &str,
) -> BTreeSet<(usize, usize)> {
    edges.sort_by_key(|e| (e.b, e.a));
    let policies = [
        Options::default(),
        Options {
            batch_admissible: true,
            ..Options::default()
        },
    ];
    let baseline = policies.map(|options| checked(n_b, n_a, edges, expected, context, options));
    // Larger oracle cases contain many unequal components; vary worker count
    // as well as input order and payloads. Keep tiny-graph tests single-threaded.
    let worker_counts = if n_b > 16 && n_a > 16 { [2, 8] } else { [1, 1] };
    for workers in worker_counts {
        rng.shuffle(edges);
        let mut payloads: Vec<_> = (0..edges.len()).map(|i| 17 + i * 11).collect();
        rng.shuffle(&mut payloads);
        for (e, payload) in edges.iter_mut().zip(payloads) {
            e.payload = payload;
        }
        for (mut options, pairs) in policies.into_iter().zip(&baseline) {
            options.workers = NonZeroUsize::new(workers).expect("positive test worker count");
            assert_eq!(
                &checked(n_b, n_a, edges, expected, context, options),
                pairs,
                "{context} ({options:?}): canonical pairs changed after shuffle/payload remap; {edges:?}"
            );
        }
    }
    baseline[0].clone()
}

#[test]
fn equal_gain_chain_batches_multiple_augmentations() {
    let n = 16;
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push(edge(i, i, 75, 2 * i));
        if i + 1 < n {
            edges.push(edge(i, i + 1, 50, 2 * i + 1));
        }
    }
    let sequential = solve_with_options(n, n, &edges, Limits::default(), Options::default())
        .expect("sequential chain solves");
    let batched = solve_with_options(
        n,
        n,
        &edges,
        Limits::default(),
        Options {
            batch_admissible: true,
            ..Options::default()
        },
    )
    .expect("batched chain solves");
    assert_eq!(
        batched.objective_units,
        75 * n as i128,
        "diagonal is optimal"
    );
    assert_eq!(
        batched.selected_payloads, sequential.selected_payloads,
        "unique optimum is unchanged"
    );
    assert_eq!(
        batched.stats.batched_augmentations,
        n - 1,
        "one search supplies the remaining pairs"
    );
    assert_eq!(
        batched.stats.shortest_path_searches, 2,
        "initial plus termination search"
    );
    assert_eq!(
        sequential.stats.shortest_path_searches,
        n + 1,
        "the reference searches once per pair, then stops"
    );
}

#[test]
fn exhaustive_ternary_three_by_three_and_other_alphabets() {
    let mut rng = Rng(0x91ea_d173_4432_7811);
    // Each cell is absent, low, or high: all 3^9 possible 3x3 graphs.
    // The second alphabet independently exercises explicit zero-weight edges.
    for (n, low, high) in [(3, 1, 2), (3, 0, SCALE), (2, SCALE - 1, SCALE)] {
        for code in 0..3_usize.pow((n * n) as u32) {
            let mut digits = code;
            let mut edges = Vec::new();
            for b in 0..n {
                for a in 0..n {
                    let digit = digits % 3;
                    digits /= 3;
                    if digit != 0 {
                        edges.push(edge(b, a, if digit == 1 { low } else { high }, b * n + a));
                    }
                }
            }
            let expected = oracle(n, n, &edges);
            variants(
                n,
                n,
                &mut edges,
                expected,
                &mut rng,
                &format!("ternary n={n} low={low} high={high} code={code}"),
            );
        }
    }
}

#[test]
fn ten_thousand_seeded_rectangular_optional_graphs() {
    const SEED: u64 = 0xfedc_1234_9876_5abc;
    let mut rng = Rng(SEED);
    let weights = [
        0,
        1,
        2,
        SCALE / 2 - 1,
        SCALE / 2,
        SCALE / 2 + 1,
        SCALE - 1,
        SCALE,
    ];
    for case in 0..10_000 {
        let n_b = rng.below(8);
        let n_a = rng.below(8);
        let density = [10, 25, 45, 70][case % 4];
        let mut edges = Vec::new();
        for b in 0..n_b {
            for a in 0..n_a {
                if rng.below(100) < density {
                    edges.push(edge(b, a, weights[rng.below(weights.len())], b * 8 + a));
                }
            }
        }
        let expected = oracle(n_b, n_a, &edges);
        variants(
            n_b,
            n_a,
            &mut edges,
            expected,
            &mut rng,
            &format!("seed={SEED:#x} case={case} n_b={n_b} n_a={n_a}"),
        );
    }
}

#[test]
fn worked_optional_matching_examples() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for (name, entries, expected, cardinality) in [
        (
            "cardinality trap",
            vec![(0, 0, 90), (0, 1, 40), (1, 0, 40)],
            90,
            1,
        ),
        (
            "reassignment",
            vec![(0, 0, 96), (0, 1, 95), (1, 0, 94), (1, 1, 86)],
            189,
            2,
        ),
        (
            "zero gain stops",
            vec![(0, 0, 90), (0, 1, 45), (1, 0, 45)],
            90,
            1,
        ),
    ] {
        let mut edges: Vec<_> = entries
            .into_iter()
            .enumerate()
            .map(|(p, (b, a, w))| edge(b, a, w, p))
            .collect();
        assert_eq!(
            oracle(2, 2, &edges),
            expected,
            "{name}: independent fixture check"
        );
        let pairs = variants(2, 2, &mut edges, expected, &mut rng, name);
        assert_eq!(pairs.len(), cardinality, "{name}: optional stopping rule");
    }
}

#[test]
fn rectangular_ties_zeros_and_empty_sides() {
    let mut rng = Rng(0xa187_c098_4431_2211);
    for (n_b, n_a) in [(0, 0), (0, 7), (7, 0), (2, 2), (3, 6), (6, 3)] {
        for weight in [0, 1, SCALE] {
            let mut edges = Vec::new();
            for b in 0..n_b {
                for a in 0..n_a {
                    edges.push(edge(b, a, weight, b * n_a + a));
                }
            }
            variants(
                n_b,
                n_a,
                &mut edges,
                (n_b.min(n_a) as i128) * i128::from(weight),
                &mut rng,
                &format!("ties {n_b}x{n_a} weight={weight}"),
            );
        }
    }
}

#[test]
fn medium_disconnected_graphs_match_independent_component_oracles() {
    const SEED: u64 = 0x7123_4abc_89de_f013;
    let mut rng = Rng(SEED);
    for case in 0..24 {
        let (mut n_b, mut n_a, mut expected) = (0, 0, 0_i128);
        let mut edges = Vec::new();
        for _ in 0..32 {
            let bs = 1 + rng.below(6);
            let aas = 1 + rng.below(6);
            let mut local = Vec::new();
            for b in 0..bs {
                for a in 0..aas {
                    if rng.below(100) < 45 {
                        local.push(edge(b, a, rng.below(17) as i64, local.len()));
                    }
                }
            }
            expected += oracle(bs, aas, &local);
            for e in local {
                edges.push(edge(n_b + e.b, n_a + e.a, e.weight, edges.len()));
            }
            n_b += bs;
            n_a += aas;
        }
        variants(
            n_b,
            n_a,
            &mut edges,
            expected,
            &mut rng,
            &format!("medium seed={SEED:#x} case={case}"),
        );
    }
}

#[test]
fn analytic_chains_stars_dense_and_connected_sparse() {
    let mut rng = Rng(0x8812_abcd_3345_0987);
    for n in [1, 2, 7, 22, 23, 24, 128, 256] {
        let mut chain = Vec::new();
        for b in 0..n {
            chain.push(edge(b, b, 860_000_000, chain.len()));
            if b + 1 < n {
                chain.push(edge(b, b + 1, 900_000_000, chain.len()));
            }
        }
        let expected = (n as i128 * 860_000_000).max((n - 1) as i128 * 900_000_000);
        if n <= 7 {
            assert_eq!(oracle(n, n, &chain), expected, "small chain formula n={n}");
        }
        variants(
            n,
            n,
            &mut chain,
            expected,
            &mut rng,
            &format!("chain n={n}"),
        );
    }
    for transpose in [false, true] {
        let mut star: Vec<_> = (0..257)
            .map(|i| {
                let (b, a) = if transpose { (0, i) } else { (i, 0) };
                edge(b, a, (i % 31) as i64, i)
            })
            .collect();
        let (n_b, n_a) = if transpose { (1, 257) } else { (257, 1) };
        variants(n_b, n_a, &mut star, 30, &mut rng, "star");
    }
    // Dense, nonuniform component: the diagonal attains the per-row upper bound.
    let n = 40;
    let mut dense = Vec::new();
    for b in 0..n {
        for a in 0..n {
            dense.push(edge(
                b,
                a,
                if b == a {
                    1000
                } else {
                    ((b * 17 + a * 31) % 999) as i64
                },
                dense.len(),
            ));
        }
    }
    variants(
        n,
        n,
        &mut dense,
        n as i128 * 1000,
        &mut rng,
        "dense upper bound",
    );
    // One connected sparse component (not disconnected diagonal toy data).
    let n = 384;
    let mut sparse = Vec::new();
    for b in 0..n {
        for offset in [0, 1, 7, 23, 61] {
            sparse.push(edge(
                b,
                (b + offset) % n,
                if offset == 0 { 1000 } else { 999 },
                sparse.len(),
            ));
        }
    }
    variants(
        n,
        n,
        &mut sparse,
        n as i128 * 1000,
        &mut rng,
        "connected degree-five upper bound",
    );
}

#[test]
fn nine_decimal_precision_and_represented_sum_objective() {
    assert_eq!(SCALE, 1_000_000_000, "documented nine-decimal scale");
    for (raw, expected) in [
        (0.0, 0),
        (1.0, SCALE),
        (0.5000000001, 500_000_000),
        (0.5000000004, 500_000_000),
        (0.50000000049, 500_000_000),
        (0.50000000051, 500_000_001),
    ] {
        assert_eq!(
            quantize(raw).expect("finite in-range precision fixture"),
            expected,
            "raw={raw:.12}"
        );
    }
    for raw in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -f64::MAX,
    ] {
        assert!(quantize(raw).is_err(), "invalid conversion accepted: {raw}");
    }
    let diagonal = [0.50000000049, 0.50000000049];
    let cross = [0.50000000051, 0.49999999951];
    assert!(
        diagonal.iter().sum::<f64>() > cross.iter().sum::<f64>(),
        "fixture raw optimum is diagonal"
    );
    let mut edges = vec![
        edge(0, 0, quantize(diagonal[0]).expect("fixture"), 0),
        edge(1, 1, quantize(diagonal[1]).expect("fixture"), 1),
        edge(0, 1, quantize(cross[0]).expect("fixture"), 2),
        edge(1, 0, quantize(cross[1]).expect("fixture"), 3),
    ];
    let expected = oracle(2, 2, &edges);
    assert_eq!(
        expected, 1_000_000_001,
        "represented objective differs from raw objective"
    );
    let pairs = variants(
        2,
        2,
        &mut edges,
        expected,
        &mut Rng(0xcafe_9876_5432_1001),
        "represented sum",
    );
    assert_eq!(
        pairs,
        BTreeSet::from([(0, 1), (1, 0)]),
        "represented optimum is cross"
    );
}
