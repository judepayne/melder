//! Compare exact and approximate solvers on frozen integer-unit graphs or sparse-v1 inputs.
//!
//! Usage: compare GRAPH_OR_SHAPE:N:D:SEED MODE WORKERS LIMIT SHUFFLE_SEED PAIR_OUTPUT
//! Frozen text starts with `n_b n_a`, followed by `b a weight` (integer SCALE units).
//! LIMIT is the approximate solver's exact-component vertex limit, not a resource cap.
//! Pair output is sorted by (b, a); solve_s excludes loading, shuffling, validation and IO.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::num::NonZeroUsize;
use std::time::Instant;

use assignment_solver::{
    ApproxMode, Limits, Options, SCALE, WeightedEdge, solve_approx, solve_with_options,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// Keep the RNG draws and sorted-neighbor ordering identical to measure.rs sparse-v1.
fn generate(n: usize, degree: usize, shape: &str, seed: u64) -> Vec<WeightedEdge> {
    let mut rng = Rng(seed);
    let mut edges = Vec::new();
    let mut start = 0;
    let mut group = 0;
    while start < n {
        let width = if shape == "skewed" {
            [1, 4, 16, 64, 256, 1024][group % 6].min(n - start)
        } else {
            n
        };
        for local_b in 0..width {
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
            neighbors.sort_unstable();
            for local_a in neighbors {
                let weight = if shape == "chain" {
                    if local_a == local_b {
                        750_000_000
                    } else {
                        500_000_000
                    }
                } else {
                    (rng.next() % 1_000_000_000 + 1) as i64
                };
                edges.push(WeightedEdge {
                    b: start + local_b,
                    a: start + local_a,
                    weight,
                    payload: edges.len(),
                });
            }
        }
        start += width;
        group += 1;
    }
    edges
}

fn read_graph(reader: impl BufRead) -> Result<(usize, usize, Vec<WeightedEdge>), String> {
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or("missing n_b n_a header")?
        .map_err(|error| error.to_string())?;
    let dimensions: Vec<_> = header.split_whitespace().collect();
    if dimensions.len() != 2 {
        return Err("header must contain n_b n_a".into());
    }
    let n_b = dimensions[0].parse().map_err(|_| "invalid n_b")?;
    let n_a = dimensions[1].parse().map_err(|_| "invalid n_a")?;
    let mut edges = Vec::new();
    for (index, line) in lines.enumerate() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        let parse = || -> Option<WeightedEdge> {
            if fields.len() != 3 {
                return None;
            }
            Some(WeightedEdge {
                b: fields[0].parse().ok()?,
                a: fields[1].parse().ok()?,
                weight: fields[2].parse().ok()?,
                payload: edges.len(),
            })
        };
        let edge = parse().ok_or_else(|| format!("invalid b a weight at line {}", index + 2))?;
        if edge.b >= n_b || edge.a >= n_a || !(0..=SCALE).contains(&edge.weight) {
            return Err(format!(
                "edge outside dimensions or weight range at line {}",
                index + 2
            ));
        }
        edges.push(edge);
    }
    Ok((n_b, n_a, edges))
}

fn load_graph(source: &str) -> Result<(usize, usize, Vec<WeightedEdge>), String> {
    let fields: Vec<_> = source.split(':').collect();
    if matches!(fields[0], "giant" | "skewed" | "chain") {
        if fields.len() != 4 {
            return Err("synthetic source must be giant|skewed|chain:N:D:SEED".into());
        }
        let n = fields[1].parse().map_err(|_| "invalid synthetic N")?;
        let degree = fields[2].parse().map_err(|_| "invalid synthetic degree")?;
        let seed = fields[3].parse().map_err(|_| "invalid synthetic seed")?;
        if n == 0 || degree < 2 || SCALE != 1_000_000_000 {
            return Err("sparse-v1 requires N > 0, D >= 2 and SCALE = 1000000000".into());
        }
        Ok((n, n, generate(n, degree, fields[0], seed)))
    } else {
        read_graph(BufReader::new(
            File::open(source).map_err(|error| error.to_string())?,
        ))
    }
}

fn shuffle(edges: &mut [WeightedEdge], seed: u64) {
    let mut rng = Rng(seed);
    for i in (1..edges.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        edges.swap(i, j);
    }
}

fn selected_pairs(
    canonical: &[WeightedEdge],
    selected: &[usize],
    objective: i128,
) -> Result<Vec<(usize, usize)>, String> {
    let mut pairs = Vec::with_capacity(selected.len());
    let mut total = 0_i128;
    for &payload in selected {
        let edge = canonical
            .get(payload)
            .ok_or("selected payload outside graph")?;
        total += i128::from(edge.weight);
        pairs.push((edge.b, edge.a));
    }
    pairs.sort_unstable();
    if total != objective {
        return Err("selected objective differs from reported objective".into());
    }
    if pairs.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("selected matching repeats a B vertex".into());
    }
    let mut a_ids: Vec<_> = pairs.iter().map(|pair| pair.1).collect();
    a_ids.sort_unstable();
    if a_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("selected matching repeats an A vertex".into());
    }
    Ok(pairs)
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 7 {
        return Err("usage: compare GRAPH_OR_giant:N:D:SEED exact|score|count|best|hybrid WORKERS LIMIT SHUFFLE_SEED PAIR_OUTPUT (weights are integer units; LIMIT is exact-component vertices)".into());
    }
    let mode = match args[2].as_str() {
        "exact" => None,
        "score" => Some(ApproxMode::ScoreFirst),
        "count" => Some(ApproxMode::CountFirst),
        "best" => Some(ApproxMode::Best),
        "hybrid" => Some(ApproxMode::Hybrid),
        _ => return Err("mode must be exact|score|count|best|hybrid".into()),
    };
    let workers: NonZeroUsize = args[3].parse().map_err(|_| "WORKERS must be positive")?;
    let limit: usize = args[4].parse().map_err(|_| "invalid LIMIT")?;
    let shuffle_seed: u64 = args[5].parse().map_err(|_| "invalid SHUFFLE_SEED")?;
    let (n_b, n_a, canonical) = load_graph(&args[1])?;
    let mut edges = canonical.clone();
    shuffle(&mut edges, shuffle_seed);

    // Only the solver call is timed. Input construction, cloning and output are excluded.
    let start = Instant::now();
    let (selected, objective, upper, exact_components, approx_components, largest) =
        if let Some(mode) = mode {
            let result = solve_approx(n_b, n_a, &edges, Limits::default(), mode, workers, limit)
                .map_err(|error| format!("solve: {error}"))?;
            (
                result.selected_payloads,
                result.objective_units,
                result.upper_units,
                result.exact_components,
                result.approx_components,
                result.largest_component_vertices,
            )
        } else {
            let result = solve_with_options(
                n_b,
                n_a,
                &edges,
                Limits::default(),
                Options {
                    workers,
                    batch_admissible: true,
                },
            )
            .map_err(|error| format!("solve: {error}"))?;
            (
                result.selected_payloads,
                result.objective_units,
                result.objective_units,
                result.stats.components,
                0,
                result.stats.largest_component_vertices,
            )
        };
    let solve_s = start.elapsed().as_secs_f64();
    if upper < objective {
        return Err("upper bound is below objective".into());
    }
    let pairs = selected_pairs(&canonical, &selected, objective)?;
    let mut output = BufWriter::new(File::create(&args[6]).map_err(|error| error.to_string())?);
    for (b, a) in &pairs {
        writeln!(output, "{b} {a}").map_err(|error| error.to_string())?;
    }
    output.flush().map_err(|error| error.to_string())?;
    println!(
        "{{\"mode\":\"{}\",\"workers\":{workers},\"exact_component_limit\":{limit},\"shuffle_seed\":{shuffle_seed},\"n_b\":{n_b},\"n_a\":{n_a},\"edges\":{},\"solve_s\":{solve_s},\"objective_units\":\"{objective}\",\"upper_units\":\"{upper}\",\"selected\":{},\"components\":{},\"exact_components\":{exact_components},\"approx_components\":{approx_components},\"largest_component_vertices\":{largest}}}",
        args[2],
        edges.len(),
        pairs.len(),
        exact_components + approx_components,
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
    fn frozen_graph_uses_integer_units_and_stable_payloads() {
        let (n_b, n_a, canonical) =
            read_graph(&b"2 3\n1 2 750000000\n0 1 500000000\n"[..]).expect("valid frozen graph");
        assert_eq!((n_b, n_a), (2, 3), "rectangular dimensions survive");
        let mut shuffled = canonical.clone();
        shuffle(&mut shuffled, 17);
        for edge in shuffled {
            assert_eq!(
                edge, canonical[edge.payload],
                "payload identifies original edge"
            );
        }
        assert_eq!(
            selected_pairs(&canonical, &[0, 1], 1_250_000_000).expect("valid matching"),
            vec![(0, 1), (1, 2)],
            "output is canonical"
        );
    }

    #[test]
    fn rejects_malformed_frozen_graphs() {
        for input in [
            "2\n",
            "2 2\n0 2 1\n",
            "2 2\n0 0 .5\n",
            "2 2\n0 0 -1\n",
            "2 2\n0 0 1000000001\n",
        ] {
            assert!(
                read_graph(input.as_bytes()).is_err(),
                "must reject {input:?}"
            );
        }
    }

    #[test]
    fn chain_recipe_and_shuffle_are_reproducible() {
        let edges = generate(4, 2, "chain", 19);
        assert_eq!(edges.len(), 7, "chain has diagonal and successor edges");
        assert_eq!(edges[0].weight, 750_000_000, "diagonal weight");
        assert_eq!(edges[1].weight, 500_000_000, "successor weight");
        let mut first = edges.clone();
        let mut second = edges;
        shuffle(&mut first, 31);
        shuffle(&mut second, 31);
        assert_eq!(first, second, "shuffle is deterministic");
    }
}
