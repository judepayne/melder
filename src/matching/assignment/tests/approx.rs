//! Independent enumerative oracles, certificates, and canonical-pair regressions.
//! No production matching, component, or quantization helper is used as an oracle.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use super::{ApproxMatching, ApproxMode, Limits, SCALE, SolverError, WeightedEdge, solve_approx};

const MODES: [ApproxMode; 4] = [
    ApproxMode::ScoreFirst,
    ApproxMode::CountFirst,
    ApproxMode::Best,
    ApproxMode::Hybrid,
];

fn edge(b: usize, a: usize, weight: i64, payload: usize) -> WeightedEdge {
    WeightedEdge {
        b,
        a,
        weight,
        payload,
    }
}

// Enumerate every partial injection, including the unmatched branch and zero
// edges. Cardinality and weight are independent maxima, NOT lexicographic goals.
fn oracle(n_b: usize, n_a: usize, edges: &[WeightedEdge]) -> (i128, usize) {
    fn visit(b: usize, rows: &[Vec<(usize, i64)>], used: &mut [bool]) -> (i128, usize) {
        if b == rows.len() {
            return (0, 0);
        }
        let mut best = visit(b + 1, rows, used);
        for &(a, weight) in &rows[b] {
            if !used[a] {
                used[a] = true;
                let tail = visit(b + 1, rows, used);
                used[a] = false;
                best.0 = best.0.max(i128::from(weight) + tail.0);
                best.1 = best.1.max(1 + tail.1);
            }
        }
        best
    }
    let mut rows = vec![Vec::new(); n_b];
    for e in edges {
        rows[e.b].push((e.a, e.weight));
    }
    visit(0, &rows, &mut vec![false; n_a])
}

// Deliberately quadratic edge flood fill instead of the solver's union-find.
fn components(edges: &[WeightedEdge]) -> Vec<Vec<WeightedEdge>> {
    let mut remaining = edges.to_vec();
    let mut parts = Vec::new();
    while let Some(first) = remaining.pop() {
        let mut part = vec![first];
        loop {
            let before = part.len();
            let mut i = 0;
            while i < remaining.len() {
                if part
                    .iter()
                    .any(|e| e.b == remaining[i].b || e.a == remaining[i].a)
                {
                    part.push(remaining.swap_remove(i));
                } else {
                    i += 1;
                }
            }
            if part.len() == before {
                break;
            }
        }
        parts.push(part);
    }
    parts
}

fn active(edges: &[WeightedEdge]) -> usize {
    edges.iter().map(|e| e.b).collect::<BTreeSet<_>>().len()
        + edges.iter().map(|e| e.a).collect::<BTreeSet<_>>().len()
}

fn row_col_upper(edges: &[WeightedEdge]) -> i128 {
    let bs: BTreeSet<_> = edges.iter().map(|e| e.b).collect();
    let aas: BTreeSet<_> = edges.iter().map(|e| e.a).collect();
    let rows: i128 = bs
        .into_iter()
        .map(|b| {
            i128::from(
                edges
                    .iter()
                    .filter(|e| e.b == b)
                    .map(|e| e.weight)
                    .max()
                    .expect("active row"),
            )
        })
        .sum();
    let cols: i128 = aas
        .into_iter()
        .map(|a| {
            i128::from(
                edges
                    .iter()
                    .filter(|e| e.a == a)
                    .map(|e| e.weight)
                    .max()
                    .expect("active column"),
            )
        })
        .sum();
    rows.min(cols)
}

fn run(
    n_b: usize,
    n_a: usize,
    edges: &[WeightedEdge],
    mode: ApproxMode,
    workers: usize,
    limit: usize,
) -> ApproxMatching {
    solve_approx(
        n_b,
        n_a,
        edges,
        Limits::default(),
        mode,
        NonZeroUsize::new(workers).expect("positive workers"),
        limit,
    )
    .unwrap_or_else(|err| {
        panic!("{mode:?} workers={workers} limit={limit}: {err:?}; graph={edges:?}")
    })
}

fn pairs(result: &ApproxMatching, edges: &[WeightedEdge], context: &str) -> Vec<(usize, usize)> {
    let mut seen_payloads = BTreeSet::new();
    let mut bs = BTreeSet::new();
    let mut aas = BTreeSet::new();
    let mut pairs = Vec::new();
    let mut sum = 0_i128;
    for p in &result.selected_payloads {
        assert!(
            seen_payloads.insert(*p),
            "{context}: duplicate payload {p}; graph={edges:?}"
        );
        let e = edges
            .iter()
            .find(|e| e.payload == *p)
            .unwrap_or_else(|| panic!("{context}: unknown payload {p}; graph={edges:?}"));
        assert!(
            bs.insert(e.b),
            "{context}: reused B {}; pairs={pairs:?}; graph={edges:?}",
            e.b
        );
        assert!(
            aas.insert(e.a),
            "{context}: reused A {}; pairs={pairs:?}; graph={edges:?}",
            e.a
        );
        pairs.push((e.b, e.a));
        sum += i128::from(e.weight);
    }
    assert_eq!(
        sum, result.objective_units,
        "{context}: extraction; pairs={pairs:?}; graph={edges:?}"
    );
    assert!(
        pairs.windows(2).all(|p| p[0] < p[1]),
        "{context}: output not canonical; pairs={pairs:?}; graph={edges:?}"
    );
    pairs
}

fn check(
    n_b: usize,
    n_a: usize,
    edges: &[WeightedEdge],
    mode: ApproxMode,
    limit: usize,
    expected: (i128, usize),
    context: &str,
) -> ApproxMatching {
    let result = run(n_b, n_a, edges, mode, 1, limit);
    let selected = pairs(&result, edges, context);
    assert!(
        result.objective_units <= expected.0 && expected.0 <= result.upper_units,
        "{context}: mode={mode:?}: objective={} optimum={} upper={}; pairs={selected:?}; graph={edges:?}",
        result.objective_units,
        expected.0,
        result.upper_units
    );
    let parts = components(edges);
    let upper: i128 = parts.iter().map(|p| row_col_upper(p)).sum();
    assert!(
        result.upper_units <= upper,
        "{context}: loose certificate; pairs={selected:?}; graph={edges:?}"
    );
    if !matches!(mode, ApproxMode::Hybrid) {
        assert_eq!(
            result.upper_units, upper,
            "{context}: row/column certificate; pairs={selected:?}; graph={edges:?}"
        );
    }
    let exact = parts
        .iter()
        .filter(|p| matches!(mode, ApproxMode::Hybrid) && active(p) <= limit)
        .count();
    assert_eq!(
        result.exact_components, exact,
        "{context}: exact component count"
    );
    assert_eq!(
        result.approx_components,
        parts.len() - exact,
        "{context}: approximate component count"
    );
    assert_eq!(
        result.largest_component_vertices,
        parts.iter().map(|p| active(p)).max().unwrap_or(0),
        "{context}: active component size"
    );
    if matches!(mode, ApproxMode::CountFirst) {
        assert_eq!(
            selected.len(),
            expected.1,
            "{context}: maximum cardinality; pairs={selected:?}; graph={edges:?}"
        );
    }
    if matches!(mode, ApproxMode::Hybrid) && exact == parts.len() {
        assert_eq!(
            result.objective_units, expected.0,
            "{context}: exact hybrid optimum; pairs={selected:?}; graph={edges:?}"
        );
    }
    result
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
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

fn deterministic(n_b: usize, n_a: usize, edges: &[WeightedEdge], rng: &mut Rng, context: &str) {
    for mode in MODES {
        for limit in [0, 4, usize::MAX] {
            let baseline = run(n_b, n_a, edges, mode, 1, limit);
            let expected = pairs(&baseline, edges, context);
            let metadata = (
                baseline.objective_units,
                baseline.upper_units,
                baseline.exact_components,
                baseline.approx_components,
                baseline.largest_component_vertices,
            );
            for workers in [1, 2, 8] {
                let mut shuffled = edges.to_vec();
                rng.shuffle(&mut shuffled);
                let mut payloads: Vec<_> = (0..edges.len()).map(|i| 101 + i * 17).collect();
                rng.shuffle(&mut payloads);
                for (e, p) in shuffled.iter_mut().zip(payloads) {
                    e.payload = p;
                }
                let result = run(n_b, n_a, &shuffled, mode, workers, limit);
                let actual = pairs(&result, &shuffled, context);
                assert_eq!(
                    actual, expected,
                    "{context}: mode={mode:?} workers={workers} limit={limit}; remapped graph={shuffled:?}"
                );
                assert_eq!(
                    (
                        result.objective_units,
                        result.upper_units,
                        result.exact_components,
                        result.approx_components,
                        result.largest_component_vertices
                    ),
                    metadata,
                    "{context}: metadata changed; mode={mode:?} workers={workers}; pairs={actual:?}"
                );
            }
        }
    }
}

#[test]
fn ten_thousand_seeded_graphs_bound_exact_optional_optimum_and_hk_cardinality() {
    const SEED: u64 = 0x4f71_298a_bce3_650d;
    let mut rng = Rng(SEED);
    let weights = [
        0,
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
        let (n_b, n_a) = (rng.below(7), rng.below(7));
        let mut edges = Vec::new();
        for b in 0..n_b {
            for a in 0..n_a {
                if rng.below(100) < [10, 35, 65, 95][case % 4] {
                    edges.push(edge(b, a, weights[rng.below(weights.len())], b * 7 + a));
                }
            }
        }
        let context = format!("seed={SEED:#x} case={case} {n_b}x{n_a}");
        let expected = oracle(n_b, n_a, &edges);
        for mode in MODES {
            check(n_b, n_a, &edges, mode, 4, expected, &context);
        }
        check(
            n_b,
            n_a,
            &edges,
            ApproxMode::Hybrid,
            usize::MAX,
            expected,
            &context,
        );
        // Best chooses by weight within each component, not globally, and ties
        // retain score-first even if count-first has greater cardinality.
        let best = run(n_b, n_a, &edges, ApproxMode::Best, 1, 0);
        let mut expected_pairs = Vec::new();
        for part in components(&edges) {
            let score = run(n_b, n_a, &part, ApproxMode::ScoreFirst, 1, 0);
            let count = run(n_b, n_a, &part, ApproxMode::CountFirst, 1, 0);
            let chosen = if count.objective_units > score.objective_units {
                &count
            } else {
                &score
            };
            expected_pairs.extend(pairs(chosen, &part, &context));
        }
        expected_pairs.sort_unstable();
        assert_eq!(
            pairs(&best, &edges, &context),
            expected_pairs,
            "{context}: component-wise Best choice; graph={edges:?}"
        );
        if case % 100 == 0 {
            deterministic(n_b, n_a, &edges, &mut rng, &context);
        }
    }
}

#[test]
fn actual_pair_sets_pin_weight_cardinality_and_best_tie_rules() {
    for (name, weights, score_pairs, count_pairs, best_pairs) in [
        (
            "cardinality loses weight",
            [90, 40, 40],
            vec![(0, 0)],
            vec![(0, 1), (1, 0)],
            vec![(0, 0)],
        ),
        (
            "cardinality wins weight",
            [90, 60, 60],
            vec![(0, 0)],
            vec![(0, 1), (1, 0)],
            vec![(0, 1), (1, 0)],
        ),
        (
            "Best equal objective retains score-first",
            [90, 45, 45],
            vec![(0, 0)],
            vec![(0, 1), (1, 0)],
            vec![(0, 0)],
        ),
    ] {
        let edges = [
            edge(0, 0, weights[0], 99),
            edge(0, 1, weights[1], 3),
            edge(1, 0, weights[2], 17),
        ];
        for (mode, expected) in [
            (ApproxMode::ScoreFirst, score_pairs),
            (ApproxMode::CountFirst, count_pairs),
            (ApproxMode::Best, best_pairs),
        ] {
            let result = run(2, 2, &edges, mode, 1, 0);
            assert_eq!(pairs(&result, &edges, name), expected, "{name}: {mode:?}");
        }
        deterministic(2, 2, &edges, &mut Rng(0x7231_9acd_4271_ffe1), name);
    }
    // Equal edge weights prefer ascending canonical B then A, not payload.
    let edges = [
        edge(0, 0, 7, 999),
        edge(0, 1, 7, 1),
        edge(1, 0, 7, 2),
        edge(1, 1, 7, 0),
    ];
    for mode in MODES {
        assert_eq!(
            pairs(&run(2, 2, &edges, mode, 1, 0), &edges, "equal square"),
            vec![(0, 0), (1, 1)],
            "equal square: {mode:?}"
        );
    }
    deterministic(
        2,
        2,
        &edges,
        &mut Rng(0x7cef_49a1_0032_dabc),
        "equal square",
    );
}

#[test]
fn hk_retains_zero_edges_and_follows_long_augmenting_paths() {
    for n in [2, 3, 8, 64, 257] {
        // The last B can only take A0; all preceding B must shift by one.
        let mut edges = Vec::new();
        for b in 0..n - 1 {
            edges.push(edge(b, b, 0, edges.len()));
            edges.push(edge(b, b + 1, 0, edges.len()));
        }
        edges.push(edge(n - 1, 0, 0, edges.len()));
        let result = run(n, n, &edges, ApproxMode::CountFirst, 8, 0);
        let expected: Vec<_> = (0..n - 1).map(|b| (b, b + 1)).chain([(n - 1, 0)]).collect();
        assert_eq!(
            pairs(&result, &edges, "zero augmenting path"),
            expected,
            "n={n}"
        );
        assert_eq!(
            (result.objective_units, result.upper_units),
            (0, 0),
            "zero certificate n={n}"
        );
    }
}

#[test]
fn hybrid_mixes_components_at_inclusive_active_vertex_boundary() {
    // First component: exact improves 96 greedy to 189. Second: six active
    // vertices, Best remains 101 while exact would attain 102. Isolated IDs do
    // not count toward component sizes or the threshold.
    let edges = [
        edge(2, 3, 96, 1),
        edge(2, 7, 95, 2),
        edge(5, 3, 94, 3),
        edge(5, 7, 86, 4),
        edge(20, 30, 100, 5),
        edge(20, 31, 51, 6),
        edge(21, 30, 51, 7),
        edge(21, 31, 0, 8),
        edge(22, 31, 1, 9),
        edge(22, 32, 0, 10),
    ];
    let result = run(1000, 2000, &edges, ApproxMode::Hybrid, 8, 4);
    assert_eq!(
        (
            result.exact_components,
            result.approx_components,
            result.largest_component_vertices
        ),
        (1, 1, 6),
        "mixed stats"
    );
    let mut expected = vec![(2, 7), (5, 3)];
    let large: Vec<_> = edges.iter().copied().filter(|e| e.b >= 20).collect();
    expected.extend(pairs(
        &run(1000, 2000, &large, ApproxMode::Best, 1, 0),
        &large,
        "large component",
    ));
    assert_eq!(
        pairs(&result, &edges, "mixed hybrid"),
        expected,
        "component-local exact/Best dispatch"
    );
    for (limit, exact, approx) in [(0, 0, 2), (3, 0, 2), (4, 1, 1), (5, 1, 1), (6, 2, 0)] {
        let result = run(1000, 2000, &edges, ApproxMode::Hybrid, 2, limit);
        assert_eq!(
            (result.exact_components, result.approx_components),
            (exact, approx),
            "inclusive limit={limit}"
        );
    }
    deterministic(
        1000,
        2000,
        &edges,
        &mut Rng(0x11ab_cde2_8790_5421),
        "mixed hybrid",
    );
}

#[test]
fn disconnected_unequal_components_are_deterministic_across_workers() {
    let mut edges = Vec::new();
    for component in 0..24 {
        let base = component * 10;
        for b in 0..1 + component % 6 {
            for a in 0..1 + component % 5 {
                edges.push(edge(
                    base + b,
                    base + a,
                    ((b * 3 + a * 7) % 5) as i64,
                    edges.len(),
                ));
            }
        }
    }
    deterministic(
        250,
        250,
        &edges,
        &mut Rng(0x9913_ac87_4256_bdef),
        "24 unequal components",
    );
}

#[test]
fn rejects_malformed_graphs_and_limits_for_every_mode() {
    let valid = edge(0, 0, 1, 42);
    for mode in MODES {
        for broken in [
            edge(1, 0, 1, 42),
            edge(0, 1, 1, 42),
            edge(0, 0, -1, 42),
            edge(0, 0, SCALE + 1, 42),
        ] {
            assert!(
                matches!(
                    solve_approx(
                        1,
                        1,
                        &[broken],
                        Limits::default(),
                        mode,
                        NonZeroUsize::MIN,
                        4
                    ),
                    Err(SolverError::InvalidEdge { .. })
                ),
                "{mode:?}: invalid graph accepted: {broken:?}"
            );
        }
        assert!(
            matches!(
                solve_approx(
                    1,
                    1,
                    &[valid, edge(0, 0, 2, 43)],
                    Limits::default(),
                    mode,
                    NonZeroUsize::MIN,
                    4
                ),
                Err(SolverError::DuplicateEdge { b: 0, a: 0 })
            ),
            "{mode:?}: duplicate accepted"
        );
        for limits in [
            Limits {
                max_edges: 0,
                ..Limits::default()
            },
            Limits {
                max_active_vertices: 1,
                ..Limits::default()
            },
        ] {
            assert!(
                matches!(
                    solve_approx(1, 1, &[valid], limits, mode, NonZeroUsize::MIN, 4),
                    Err(SolverError::Limit(_))
                ),
                "{mode:?}: limit ignored: {limits:?}"
            );
        }
        // Huge declared domains must not allocate isolated endpoints.
        let sparse = [edge(usize::MAX - 1, usize::MAX - 2, SCALE, 777)];
        let result = run(usize::MAX, usize::MAX, &sparse, mode, 8, 2);
        assert_eq!(
            pairs(&result, &sparse, "huge sparse domains"),
            vec![(usize::MAX - 1, usize::MAX - 2)],
            "{mode:?}"
        );
        for (n_b, n_a) in [(0, 0), (0, 17), (17, 0), (usize::MAX, usize::MAX)] {
            check(n_b, n_a, &[], mode, 4, (0, 0), "empty graph");
        }
    }
}
