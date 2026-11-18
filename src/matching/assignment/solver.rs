//! Sparse optional-flow assignment solver with checked integer arithmetic.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use thiserror::Error;

mod admissible;
mod approx;
mod parallel;

use admissible::Admissible;
use parallel::map_components;

/// Integer units per score point (nine decimal places).
pub const SCALE: i64 = 1_000_000_000;

/// One retained B-to-A edge and its caller-owned metadata handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightedEdge {
    pub b: usize,
    pub a: usize,
    pub weight: i64,
    pub payload: usize,
}

/// Hard limits checked before solving; exceeding either returns an error.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_edges: usize,
    pub max_active_vertices: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_edges: usize::MAX,
            max_active_vertices: usize::MAX,
        }
    }
}

/// Exact-solver options; default uses the sequential reference solver.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub workers: NonZeroUsize,
    pub batch_admissible: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            workers: NonZeroUsize::MIN,
            batch_admissible: false,
        }
    }
}

/// Exact-solver work counters and component phase timings.
#[derive(Debug, Default)]
pub struct Stats {
    pub components: usize,
    pub largest_component_vertices: usize,
    pub largest_component_edges: usize,
    pub augmentations: usize,
    pub shortest_path_searches: usize,
    pub batched_augmentations: usize,
    // Phase durations sum over components, not elapsed wall time across workers.
    pub build_time: Duration,
    pub search_time: Duration,
    pub extract_time: Duration,
}

/// Certified exact assignment over the retained integer-weight graph.
#[derive(Debug, Default)]
pub struct Matching {
    /// Metadata handles ordered by (B, A), independent of their numeric values.
    pub selected_payloads: Vec<usize>,
    pub objective_units: i128,
    pub stats: Stats,
}

/// Invalid input, resource limits, or a failed checked solver operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SolverError {
    #[error("score must be finite and in [0, 1]")]
    InvalidScore,
    #[error("invalid edge ({b}, {a}) with weight {weight}")]
    InvalidEdge { b: usize, a: usize, weight: i64 },
    #[error("duplicate edge ({b}, {a})")]
    DuplicateEdge { b: usize, a: usize },
    #[error("{0} limit exceeded")]
    Limit(&'static str),
    #[error("solver allocation failed")]
    Allocation,
    #[error("solver arithmetic overflow")]
    Overflow,
    #[error("solver invariant: {0}")]
    Invariant(&'static str),
}

type Result<T> = std::result::Result<T, SolverError>;

/// Quantize a score that has already passed the caller's raw acceptance rules.
pub fn quantize(score: f64) -> Result<i64> {
    if !(0.0..=1.0).contains(&score) {
        return Err(SolverError::InvalidScore);
    }
    // Both the scaled and rounded values are in [0, SCALE], well within i64.
    Ok((score * SCALE as f64).round() as i64)
}

fn add(x: i128, y: i128) -> Result<i128> {
    x.checked_add(y).ok_or(SolverError::Overflow)
}

fn sub(x: i128, y: i128) -> Result<i128> {
    x.checked_sub(y).ok_or(SolverError::Overflow)
}

fn reserve<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| SolverError::Allocation)?;
    Ok(v)
}

fn filled<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut v = reserve(n)?;
    v.resize(n, value);
    Ok(v)
}

// --- Canonical graph preparation ---

fn endpoints(edges: &[WeightedEdge], key: fn(&WeightedEdge) -> usize) -> Result<Vec<usize>> {
    let mut ids = reserve(edges.len())?;
    ids.extend(edges.iter().map(key));
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

struct Components {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Components {
    fn new(n: usize) -> Result<Self> {
        let mut parent = reserve(n)?;
        parent.extend(0..n);
        Ok(Self {
            parent,
            size: filled(n, 1)?,
        })
    }

    fn root(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            self.parent[v] = self.parent[self.parent[v]];
            v = self.parent[v];
        }
        v
    }

    fn union(&mut self, x: usize, y: usize) -> Result<()> {
        let (mut x, mut y) = (self.root(x), self.root(y));
        if x != y {
            if self.size[x] < self.size[y] {
                std::mem::swap(&mut x, &mut y);
            }
            self.parent[y] = x;
            self.size[x] = self.size[x]
                .checked_add(self.size[y])
                .ok_or(SolverError::Overflow)?;
        }
        Ok(())
    }
}

struct Group {
    range: std::ops::Range<usize>,
    vertices: usize,
}

struct Prepared {
    sorted: Vec<WeightedEdge>,
    grouped: Vec<(usize, usize)>,
    groups: Vec<Group>,
    capacity: usize,
}

impl Prepared {
    fn edges(&self, job: usize) -> Result<Vec<WeightedEdge>> {
        let range = self.groups[job].range.clone();
        let mut part = reserve(range.len())?;
        part.extend(self.grouped[range].iter().map(|&(_, i)| self.sorted[i]));
        Ok(part)
    }
}

fn prepare(
    n_b: usize,
    n_a: usize,
    edges: &[WeightedEdge],
    limits: Limits,
    workers: NonZeroUsize,
) -> Result<Prepared> {
    if edges.len() > limits.max_edges {
        return Err(SolverError::Limit("edge"));
    }
    for e in edges {
        if e.b >= n_b || e.a >= n_a || !(0..=SCALE).contains(&e.weight) {
            return Err(SolverError::InvalidEdge {
                b: e.b,
                a: e.a,
                weight: e.weight,
            });
        }
    }
    let mut sorted = reserve(edges.len())?;
    sorted.extend_from_slice(edges);
    sorted.sort_unstable_by_key(|e| (e.b, e.a));
    for pair in sorted.windows(2) {
        if (pair[0].b, pair[0].a) == (pair[1].b, pair[1].a) {
            return Err(SolverError::DuplicateEdge {
                b: pair[0].b,
                a: pair[0].a,
            });
        }
    }
    let (bs, as_) = (endpoints(&sorted, |e| e.b)?, endpoints(&sorted, |e| e.a)?);
    let active = bs
        .len()
        .checked_add(as_.len())
        .ok_or(SolverError::Overflow)?;
    if active > limits.max_active_vertices {
        return Err(SolverError::Limit("active vertex"));
    }
    let mut cc = Components::new(active)?;
    let mut grouped = reserve(sorted.len())?;
    for (idx, e) in sorted.iter().enumerate() {
        let b = ordinal(&bs, e.b)?;
        let a = bs.len() + ordinal(&as_, e.a)?; // bounded by `active`
        cc.union(b, a)?;
        grouped.push((b, idx));
    }
    for (b, _) in &mut grouped {
        *b = cc.root(*b);
    }
    grouped.sort_unstable();
    let mut groups = reserve(bs.len())?;
    let mut start = 0;
    for group in grouped.chunk_by(|x, y| x.0 == y.0) {
        let end = start + group.len();
        groups.push(Group {
            range: start..end,
            vertices: cc.size[group[0].0],
        });
        start = end;
    }
    // Canonical minimum edge breaks ties even when large jobs run first.
    groups.sort_unstable_by_key(|g| {
        (
            Reverse(if workers.get() > 1 { g.range.len() } else { 0 }),
            grouped[g.range.start].1,
        )
    });
    Ok(Prepared {
        sorted,
        grouped,
        groups,
        capacity: bs.len().min(as_.len()),
    })
}

/// Maximize the sum of retained integer weights. Endpoints are side-specific
/// ordinals assigned by the caller in canonical ID order. Input edge order is
/// immaterial. Unmatched is allowed; tied zero-gain augmentations are skipped.
pub fn solve(n_b: usize, n_a: usize, edges: &[WeightedEdge], limits: Limits) -> Result<Matching> {
    solve_with_options(n_b, n_a, edges, limits, Options::default())
}

/// Solve with independent component workers and/or batches of equal-gain paths.
/// Worker count never changes pairs. Batching can change which tied optimum wins.
pub fn solve_with_options(
    n_b: usize,
    n_a: usize,
    edges: &[WeightedEdge],
    limits: Limits,
    options: Options,
) -> Result<Matching> {
    let started = Instant::now();
    let prepared = prepare(n_b, n_a, edges, limits, options.workers)?;
    let mut solution = Matching::default();
    solution.stats.build_time = started.elapsed();
    solution.stats.components = prepared.groups.len();
    let parts = map_components(prepared.groups.len(), options.workers, |job| {
        solve_component(&prepared.edges(job)?, options.batch_admissible)
    })?;
    let mut chosen = reserve(prepared.capacity)?;
    for part in parts {
        let now = Instant::now();
        solution.objective_units = add(solution.objective_units, part.objective)?;
        chosen.extend(part.chosen);
        let dst = &mut solution.stats;
        let src = part.stats;
        dst.largest_component_vertices = dst
            .largest_component_vertices
            .max(src.largest_component_vertices);
        dst.largest_component_edges = dst.largest_component_edges.max(src.largest_component_edges);
        dst.augmentations = dst
            .augmentations
            .checked_add(src.augmentations)
            .ok_or(SolverError::Overflow)?;
        dst.shortest_path_searches = dst
            .shortest_path_searches
            .checked_add(src.shortest_path_searches)
            .ok_or(SolverError::Overflow)?;
        dst.batched_augmentations = dst
            .batched_augmentations
            .checked_add(src.batched_augmentations)
            .ok_or(SolverError::Overflow)?;
        dst.build_time += src.build_time;
        dst.search_time += src.search_time;
        dst.extract_time += src.extract_time + now.elapsed();
    }
    let now = Instant::now();
    chosen.sort_unstable_by_key(|e| (e.b, e.a));
    solution.selected_payloads = reserve(chosen.len())?;
    solution
        .selected_payloads
        .extend(chosen.into_iter().map(|e| e.payload));
    solution.stats.extract_time += now.elapsed();
    Ok(solution)
}

/// Explicit approximate strategies; none changes the behavior of `solve`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApproxMode {
    ScoreFirst,
    CountFirst,
    Best,
    Hybrid,
}

/// Feasible assignment with a certified objective upper bound.
#[derive(Debug, Default)]
pub struct ApproxMatching {
    pub selected_payloads: Vec<usize>,
    pub objective_units: i128,
    pub upper_units: i128,
    pub exact_components: usize,
    pub approx_components: usize,
    pub largest_component_vertices: usize,
}

/// Compare deterministic feasible matchings on each component. Hybrid solves components
/// of at most `exact_component_limit` active endpoints exactly; its bound is tight there.
pub fn solve_approx(
    n_b: usize,
    n_a: usize,
    edges: &[WeightedEdge],
    limits: Limits,
    mode: ApproxMode,
    workers: NonZeroUsize,
    exact_component_limit: usize,
) -> Result<ApproxMatching> {
    let prepared = prepare(n_b, n_a, edges, limits, workers)?;
    let parts = map_components(prepared.groups.len(), workers, |job| {
        let edges = prepared.edges(job)?;
        let vertices = prepared.groups[job].vertices;
        let exact = mode == ApproxMode::Hybrid && vertices <= exact_component_limit;
        let (chosen, score, upper) = if exact {
            let part = solve_component(&edges, true)?;
            (part.chosen, part.objective, part.objective)
        } else {
            approx::component(&edges, mode)?
        };
        if score > upper {
            return Err(SolverError::Invariant("score exceeds certificate"));
        }
        Ok((chosen, score, upper, exact, vertices))
    })?;
    let mut chosen = reserve(prepared.capacity)?;
    let mut result = ApproxMatching::default();
    for (part, score, upper, exact, vertices) in parts {
        chosen.extend(part);
        result.objective_units = add(result.objective_units, score)?;
        result.upper_units = add(result.upper_units, upper)?;
        result.exact_components += usize::from(exact);
        result.approx_components += usize::from(!exact);
        result.largest_component_vertices = result.largest_component_vertices.max(vertices);
    }
    chosen.sort_unstable_by_key(|edge| (edge.b, edge.a));
    result.selected_payloads = reserve(chosen.len())?;
    result
        .selected_payloads
        .extend(chosen.into_iter().map(|edge| edge.payload));
    Ok(result)
}

struct Part {
    chosen: Vec<WeightedEdge>,
    objective: i128,
    stats: Stats,
}

fn solve_component(edges: &[WeightedEdge], batch: bool) -> Result<Part> {
    let now = Instant::now();
    let mut network = Network::build(edges)?;
    let mut stats = Stats {
        largest_component_vertices: network.sink - 1,
        largest_component_edges: edges.len(),
        build_time: now.elapsed(),
        ..Stats::default()
    };
    let now = Instant::now();
    let mut admissible = None;
    loop {
        stats.shortest_path_searches = stats
            .shortest_path_searches
            .checked_add(1)
            .ok_or(SolverError::Overflow)?;
        let Some(cost) = network.augment()? else {
            break;
        };
        if batch && network.has_tight_sink_arc()? {
            if admissible.is_none() {
                admissible = Some(Admissible::new(network.adj.len())?);
            }
            let scratch = admissible
                .as_mut()
                .ok_or(SolverError::Invariant("missing batch scratch"))?;
            let before = network.flow;
            scratch.augment(&mut network, cost)?;
            stats.batched_augmentations += network.flow - before;
        }
    }
    stats.search_time = now.elapsed();
    stats.augmentations = network.flow;
    let now = Instant::now();
    let mut chosen = reserve(network.flow)?;
    let objective = network.extract(edges, &mut chosen)?;
    stats.extract_time = now.elapsed();
    Ok(Part {
        chosen,
        objective,
        stats,
    })
}

fn ordinal(ids: &[usize], id: usize) -> Result<usize> {
    ids.binary_search(&id)
        .map_err(|_| SolverError::Invariant("missing compact endpoint"))
}

// --- Residual network and shortest paths ---

#[derive(Clone, Copy)]
struct Arc {
    to: usize,
    rev: usize,
    cost: i64,
    residual: u8,
}

type Handle = (usize, usize);
const SOURCE: usize = 0;

struct Network {
    adj: Vec<Vec<Arc>>,
    originals: Vec<Handle>,
    pi: Vec<i128>,
    dist: Vec<Option<i128>>,
    prev: Vec<Option<Handle>>,
    settled: Vec<bool>,
    heap: BinaryHeap<Reverse<(i128, usize)>>,
    sink: usize,
    objective: i128,
    flow: usize,
}

impl Network {
    fn build(edges: &[WeightedEdge]) -> Result<Self> {
        let (bs, as_) = (endpoints(edges, |e| e.b)?, endpoints(edges, |e| e.a)?);
        let vertices = bs
            .len()
            .checked_add(as_.len())
            .and_then(|n| n.checked_add(2))
            .ok_or(SolverError::Overflow)?;
        let sink = vertices - 1;
        let a_start = bs.len() + 1;
        let mut compact = reserve(edges.len())?;
        let mut degree = filled(vertices, 1_usize)?; // one incident source or sink arc per endpoint
        degree[SOURCE] = bs.len();
        degree[sink] = as_.len();
        for e in edges {
            let b = 1 + ordinal(&bs, e.b)?;
            let a = a_start + ordinal(&as_, e.a)?;
            for v in [b, a] {
                degree[v] = degree[v].checked_add(1).ok_or(SolverError::Overflow)?;
            }
            // The validated weight is nonnegative and at most SCALE.
            compact.push((b, a, -e.weight));
        }
        let mut net = Self {
            adj: reserve(vertices)?,
            originals: reserve(edges.len())?,
            pi: filled(vertices, 0)?,
            dist: filled(vertices, None)?,
            prev: filled(vertices, None)?,
            settled: filled(vertices, false)?,
            heap: BinaryHeap::new(),
            sink,
            objective: 0,
            flow: 0,
        };
        for count in degree {
            net.adj.push(reserve(count)?);
        }
        for b in 1..a_start {
            net.add_arc(SOURCE, b, 0);
        }
        for (b, a, cost) in compact {
            let i = net.add_arc(b, a, cost);
            net.originals.push((b, i));
            net.pi[a] = net.pi[a].min(i128::from(cost));
        }
        for a in a_start..sink {
            net.add_arc(a, sink, 0);
            net.pi[sink] = net.pi[sink].min(net.pi[a]);
        }
        #[cfg(debug_assertions)]
        net.check_residual()?;
        Ok(net)
    }

    fn add_arc(&mut self, u: usize, v: usize, cost: i64) -> usize {
        let (fwd, rev) = (self.adj[u].len(), self.adj[v].len());
        self.adj[u].push(Arc {
            to: v,
            rev,
            cost,
            residual: 1,
        });
        self.adj[v].push(Arc {
            to: u,
            rev: fwd,
            cost: -cost,
            residual: 0,
        });
        fwd
    }

    fn reduced(&self, u: usize, arc: Arc) -> Result<i128> {
        let cost = sub(add(i128::from(arc.cost), self.pi[u])?, self.pi[arc.to])?;
        if cost < 0 {
            return Err(SolverError::Invariant("negative reduced cost"));
        }
        Ok(cost)
    }

    fn shortest_path(&mut self) -> Result<Option<i128>> {
        self.dist.fill(None);
        self.prev.fill(None);
        self.settled.fill(false);
        self.heap.clear();
        self.heap
            .try_reserve(1)
            .map_err(|_| SolverError::Allocation)?;
        self.heap.push(Reverse((0, SOURCE)));
        self.dist[SOURCE] = Some(0);
        while let Some(Reverse((d, u))) = self.heap.pop() {
            if self.settled[u] || self.dist[u] != Some(d) {
                continue;
            }
            self.settled[u] = true;
            if u == self.sink {
                return Ok(Some(d));
            }
            for (i, arc) in self.adj[u].iter().copied().enumerate() {
                if arc.residual == 0 || self.settled[arc.to] {
                    continue;
                }
                let nd = add(d, self.reduced(u, arc)?)?;
                if self.dist[arc.to].is_none_or(|old| nd < old) {
                    self.dist[arc.to] = Some(nd);
                    self.prev[arc.to] = Some((u, i));
                    self.heap
                        .try_reserve(1)
                        .map_err(|_| SolverError::Allocation)?;
                    self.heap.push(Reverse((nd, arc.to)));
                }
            }
        }
        Ok(None)
    }

    fn augment(&mut self) -> Result<Option<i128>> {
        let Some(d) = self.shortest_path()? else {
            return Ok(None);
        };
        let original = add(sub(d, self.pi[SOURCE])?, self.pi[self.sink])?;
        if original >= 0 {
            return Ok(None);
        }
        for (v, pi) in self.pi.iter_mut().enumerate() {
            let delta = if self.settled[v] {
                self.dist[v].ok_or(SolverError::Invariant("settled vertex without distance"))?
            } else {
                d
            };
            *pi = add(*pi, delta)?;
        }
        let mut v = self.sink;
        for _ in 1..self.adj.len() {
            let (u, i) = self.prev[v].ok_or(SolverError::Invariant("missing path predecessor"))?;
            self.flip(u, i, v)?;
            v = u;
            if v == SOURCE {
                self.record_augmentation(original)?;
                return Ok(Some(original));
            }
        }
        Err(SolverError::Invariant(
            "augmentation failed to reach source",
        ))
    }

    fn flip(&mut self, u: usize, i: usize, v: usize) -> Result<()> {
        let (fwd, rev) = self.link(u, i)?;
        if fwd.to != v || fwd.residual != 1 || rev.residual != 0 || self.reduced(u, fwd)? != 0 {
            return Err(SolverError::Invariant("invalid augmentation arc"));
        }
        self.adj[u][i].residual = 0;
        self.adj[v][fwd.rev].residual = 1;
        Ok(())
    }

    fn record_augmentation(&mut self, cost: i128) -> Result<()> {
        self.objective = sub(self.objective, cost)?;
        self.flow = self.flow.checked_add(1).ok_or(SolverError::Overflow)?;
        #[cfg(debug_assertions)]
        self.check_residual()?;
        Ok(())
    }

    fn has_tight_sink_arc(&self) -> Result<bool> {
        for reverse in &self.adj[self.sink] {
            // These are the cost-zero reverses of unit-capacity A -> sink arcs.
            if reverse.residual == 0
                && self.reduced(reverse.to, self.adj[reverse.to][reverse.rev])? == 0
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn link(&self, u: usize, i: usize) -> Result<(Arc, Arc)> {
        let get = |v: usize, slot: usize| {
            self.adj
                .get(v)
                .and_then(|arcs| arcs.get(slot))
                .copied()
                .ok_or(SolverError::Invariant("bad arc index"))
        };
        let fwd = get(u, i)?;
        let rev = get(fwd.to, fwd.rev)?;
        if rev.to != u
            || rev.rev != i
            || fwd.cost.checked_neg() != Some(rev.cost)
            || !matches!((fwd.residual, rev.residual), (0, 1) | (1, 0))
        {
            return Err(SolverError::Invariant("invalid reverse arc"));
        }
        Ok((fwd, rev))
    }

    fn check_residual(&self) -> Result<()> {
        for (u, arcs) in self.adj.iter().enumerate() {
            for i in 0..arcs.len() {
                let (arc, _) = self.link(u, i)?;
                if arc.residual > 0 {
                    self.reduced(u, arc)?;
                }
            }
        }
        Ok(())
    }

    fn extract(&self, edges: &[WeightedEdge], chosen: &mut Vec<WeightedEdge>) -> Result<i128> {
        self.check_residual()?;
        let mut used = filled(self.adj.len(), false)?;
        let mut total = 0;
        let before = chosen.len();
        for (&(u, i), &edge) in self.originals.iter().zip(edges) {
            let (arc, _) = self.link(u, i)?;
            if arc.residual == 0 {
                if used[u] || used[arc.to] {
                    return Err(SolverError::Invariant("selected pairs are not bijective"));
                }
                used[u] = true;
                used[arc.to] = true;
                total = add(total, i128::from(edge.weight))?;
                chosen.push(edge);
            }
        }
        if total != self.objective || chosen.len() - before != self.flow {
            return Err(SolverError::Invariant("extraction disagrees with flow"));
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_input_and_limits_fail_instead_of_returning_partial_matches() {
        let edge = WeightedEdge {
            a: 0,
            b: 0,
            weight: 1,
            payload: 42,
        };
        let solve_edges =
            |edges: &[WeightedEdge]| solve(1, 1, edges, Limits::default()).unwrap_err();
        for broken in [
            WeightedEdge { a: 1, ..edge },
            WeightedEdge { b: 1, ..edge },
            WeightedEdge { weight: -1, ..edge },
            WeightedEdge {
                weight: SCALE + 1,
                ..edge
            },
        ] {
            assert!(matches!(
                solve_edges(&[broken]),
                SolverError::InvalidEdge { .. }
            ));
        }
        assert_eq!(
            solve_edges(&[edge, edge]),
            SolverError::DuplicateEdge { b: 0, a: 0 }
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
            assert!(matches!(
                solve(1, 1, &[edge], limits),
                Err(SolverError::Limit(_))
            ));
        }
        // Isolated endpoints don't require allocation proportional to declared domains.
        let out = solve(
            usize::MAX,
            usize::MAX,
            &[WeightedEdge {
                a: usize::MAX - 1,
                ..edge
            }],
            Limits::default(),
        )
        .unwrap();
        assert_eq!(out.selected_payloads, vec![42]);
    }

    #[test]
    fn checked_arithmetic_and_residual_faults_are_detected() {
        assert_eq!(add(i128::MAX, 1), Err(SolverError::Overflow));
        assert_eq!(sub(i128::MIN, 1), Err(SolverError::Overflow));
        assert!(matches!(
            reserve::<usize>(usize::MAX),
            Err(SolverError::Allocation)
        ));
        let mut net = Network::build(&[WeightedEdge {
            a: 0,
            b: 0,
            weight: 1,
            payload: 0,
        }])
        .unwrap();
        net.pi[2] = 1;
        assert_eq!(
            net.augment(),
            Err(SolverError::Invariant("negative reduced cost"))
        );
        net.adj[0][0].rev = usize::MAX;
        assert_eq!(
            net.check_residual(),
            Err(SolverError::Invariant("bad arc index"))
        );
    }
}
