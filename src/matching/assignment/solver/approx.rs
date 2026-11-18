//! Deterministic sparse component approximations with a certified objective bound.

use std::cmp::Reverse;

use super::{
    ApproxMode, Result, SolverError, WeightedEdge, add, endpoints, filled, ordinal, reserve,
};

struct Graph {
    compact: Vec<(usize, usize)>,
    order: Vec<usize>,
    offsets: Vec<usize>,
    n_a: usize,
}

impl Graph {
    fn build(edges: &[WeightedEdge]) -> Result<Self> {
        let bs = endpoints(edges, |e| e.b)?;
        let as_ = endpoints(edges, |e| e.a)?;
        let mut compact = reserve(edges.len())?;
        let mut offsets = filled(
            bs.len().checked_add(1).ok_or(SolverError::Overflow)?,
            0usize,
        )?;
        for e in edges {
            let b = ordinal(&bs, e.b)?;
            compact.push((b, ordinal(&as_, e.a)?));
            offsets[b + 1] = offsets[b + 1].checked_add(1).ok_or(SolverError::Overflow)?;
        }
        for b in 0..bs.len() {
            offsets[b + 1] = offsets[b + 1]
                .checked_add(offsets[b])
                .ok_or(SolverError::Overflow)?;
        }
        let mut order = reserve(edges.len())?;
        order.extend(0..edges.len());
        // A total key makes adjacency independent of sorting implementation and payload.
        order.sort_unstable_by_key(|&i| (edges[i].b, Reverse(edges[i].weight), edges[i].a));
        Ok(Self {
            compact,
            order,
            offsets,
            n_a: as_.len(),
        })
    }

    fn n_b(&self) -> usize {
        self.offsets.len() - 1
    }

    fn upper(&self, edges: &[WeightedEdge]) -> Result<i128> {
        let mut rows = filled(self.n_b(), 0i64)?;
        let mut cols = filled(self.n_a, 0i64)?;
        for (e, &(b, a)) in edges.iter().zip(&self.compact) {
            rows[b] = rows[b].max(e.weight);
            cols[a] = cols[a].max(e.weight);
        }
        let sum = |values: Vec<i64>| {
            values
                .into_iter()
                .try_fold(0, |total, w| add(total, i128::from(w)))
        };
        Ok(sum(rows)?.min(sum(cols)?))
    }

    fn score_first(&self, edges: &[WeightedEdge]) -> Result<Vec<usize>> {
        let mut order = reserve(edges.len())?;
        order.extend(0..edges.len());
        order.sort_unstable_by_key(|&i| (Reverse(edges[i].weight), edges[i].b, edges[i].a));
        let mut used_b = filled(self.n_b(), false)?;
        let mut used_a = filled(self.n_a, false)?;
        let mut chosen = reserve(self.n_b().min(self.n_a))?;
        for i in order {
            let (b, a) = self.compact[i];
            if !used_b[b] && !used_a[a] {
                used_b[b] = true;
                used_a[a] = true;
                chosen.push(i);
            }
        }
        Ok(chosen)
    }

    // Hopcroft–Karp: BFS shortest augmenting layers, then iterative DFS with
    // phase-local current arcs. Weighted adjacency is a heuristic, not a claim
    // of maximum weight among maximum-cardinality matchings.
    fn count_first(&self) -> Result<Vec<usize>> {
        let n_b = self.n_b();
        let mut matched_b = filled(n_b, None)?;
        let mut matched_a = filled(self.n_a, None)?;
        let mut dist = filled(n_b, usize::MAX)?;
        let mut next = filled(n_b, 0usize)?;
        let mut queue = reserve(n_b)?;
        let mut stack = reserve(n_b)?;
        let mut path = reserve(n_b)?;
        loop {
            queue.clear();
            dist.fill(usize::MAX);
            for b in 0..n_b {
                if matched_b[b].is_none() {
                    dist[b] = 0;
                    queue.push(b);
                }
                next[b] = self.offsets[b];
            }
            let mut shortest = usize::MAX;
            let mut head = 0;
            while head < queue.len() {
                let b = queue[head];
                head += 1;
                let depth = dist[b].checked_add(1).ok_or(SolverError::Overflow)?;
                if depth > shortest {
                    continue;
                }
                for &i in &self.order[self.offsets[b]..self.offsets[b + 1]] {
                    let a = self.compact[i].1;
                    if let Some(child) = matched_a[a] {
                        if depth < shortest && dist[child] == usize::MAX {
                            dist[child] = depth;
                            queue.push(child);
                        }
                    } else {
                        shortest = depth;
                    }
                }
            }
            if shortest == usize::MAX {
                break;
            }
            for root in 0..n_b {
                if matched_b[root].is_some() || dist[root] == usize::MAX {
                    continue;
                }
                stack.clear();
                path.clear();
                stack.push(root);
                while let Some(&b) = stack.last() {
                    if next[b] == self.offsets[b + 1] {
                        dist[b] = usize::MAX;
                        stack.pop();
                        if !stack.is_empty() {
                            path.pop();
                        }
                        continue;
                    }
                    let i = self.order[next[b]];
                    next[b] += 1; // bounded by the allocated adjacency length
                    let a = self.compact[i].1;
                    let depth = dist[b].checked_add(1).ok_or(SolverError::Overflow)?;
                    if let Some(child) = matched_a[a] {
                        if depth < shortest && dist[child] == depth {
                            path.push(i);
                            stack.push(child);
                        }
                    } else if depth == shortest {
                        path.push(i);
                        for &edge in &path {
                            let (b, a) = self.compact[edge];
                            matched_b[b] = Some(edge);
                            matched_a[a] = Some(b);
                        }
                        break;
                    }
                }
            }
        }
        let mut chosen = reserve(n_b.min(self.n_a))?;
        chosen.extend(matched_b.into_iter().flatten());
        Ok(chosen)
    }

    fn objective(&self, edges: &[WeightedEdge], chosen: &[usize]) -> Result<i128> {
        chosen
            .iter()
            .try_fold(0, |total, &i| add(total, i128::from(edges[i].weight)))
    }

    fn certify(
        &self,
        edges: &[WeightedEdge],
        mut chosen: Vec<usize>,
        upper: i128,
    ) -> Result<(Vec<WeightedEdge>, i128, i128)> {
        let mut used_b = filled(self.n_b(), false)?;
        let mut used_a = filled(self.n_a, false)?;
        let mut total = 0;
        chosen.sort_unstable(); // input is canonical (B, A), so output is too
        let mut selected = reserve(chosen.len())?;
        for i in chosen {
            let &(b, a) = self
                .compact
                .get(i)
                .ok_or(SolverError::Invariant("bad approximate edge"))?;
            if used_b[b] || used_a[a] {
                return Err(SolverError::Invariant(
                    "approximate pairs are not bijective",
                ));
            }
            used_b[b] = true;
            used_a[a] = true;
            total = add(total, i128::from(edges[i].weight))?;
            selected.push(edges[i]);
        }
        if total > upper {
            return Err(SolverError::Invariant(
                "approximate objective exceeds upper bound",
            ));
        }
        Ok((selected, total, upper))
    }
}

/// Approximate a nonempty, checked, canonical component in sparse global ordinals.
/// Returns canonical selected edges, their objective, and a valid weight upper bound.
/// Best (also Hybrid here) compares weight, retaining score-first on an exact tie.
pub(super) fn component(
    edges: &[WeightedEdge],
    mode: ApproxMode,
) -> Result<(Vec<WeightedEdge>, i128, i128)> {
    if edges.is_empty() {
        return Err(SolverError::Invariant("empty approximate component"));
    }
    let graph = Graph::build(edges)?;
    let upper = graph.upper(edges)?;
    let chosen = match mode {
        ApproxMode::ScoreFirst => graph.score_first(edges)?,
        ApproxMode::CountFirst => graph.count_first()?,
        ApproxMode::Best | ApproxMode::Hybrid => {
            let score = graph.score_first(edges)?;
            let count = graph.count_first()?;
            if graph.objective(edges, &count)? > graph.objective(edges, &score)? {
                count
            } else {
                score
            }
        }
    };
    graph.certify(edges, chosen, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edges(values: &[(usize, usize, i64)]) -> Vec<WeightedEdge> {
        values
            .iter()
            .enumerate()
            .map(|(payload, &(b, a, weight))| WeightedEdge {
                b,
                a,
                weight,
                payload,
            })
            .collect()
    }

    #[test]
    fn cardinality_weight_tradeoffs_and_fixed_tie() {
        let edges = make_edges(&[(0, 0, 10), (0, 1, 4), (1, 0, 4)]);
        let score = component(&edges, ApproxMode::ScoreFirst).unwrap();
        let count = component(&edges, ApproxMode::CountFirst).unwrap();
        assert_eq!(
            (score.0.len(), score.1, score.2),
            (1, 10, 14),
            "greedy and bound"
        );
        assert_eq!(
            (count.0.len(), count.1),
            (2, 8),
            "cardinality outranks weight"
        );
        assert_eq!(
            component(&edges, ApproxMode::Best).unwrap(),
            score,
            "best keeps higher score"
        );
        let edges = make_edges(&[(0, 0, 10), (0, 1, 5), (1, 0, 5)]);
        let score = component(&edges, ApproxMode::ScoreFirst).unwrap();
        for mode in [ApproxMode::Best, ApproxMode::Hybrid] {
            assert_eq!(
                component(&edges, mode).unwrap(),
                score,
                "ties retain score-first"
            );
        }
        let edges = make_edges(&[(0, 0, 10), (0, 1, 9), (1, 0, 9)]);
        assert_eq!(
            component(&edges, ApproxMode::Best).unwrap().1,
            18,
            "best can choose count-first"
        );
    }

    #[test]
    fn exhaustive_small_graphs_certify_cardinality_and_bounds() {
        for mask in 1usize..(1 << 9) {
            let values: Vec<_> = (0..9)
                .filter(|&i| mask & (1 << i) != 0)
                .map(|i| (i / 3, i % 3, (i % 4) as i64))
                .collect();
            let edges = make_edges(&values);
            let mut max_count = 0;
            let mut max_weight = 0;
            for subset in 0usize..(1 << edges.len()) {
                let (mut bs, mut as_, mut count, mut weight) = (0usize, 0usize, 0, 0i128);
                let mut valid = true;
                for (i, e) in edges.iter().enumerate() {
                    if subset & (1 << i) == 0 {
                        continue;
                    }
                    if bs & (1 << e.b) != 0 || as_ & (1 << e.a) != 0 {
                        valid = false;
                        break;
                    }
                    bs |= 1 << e.b;
                    as_ |= 1 << e.a;
                    count += 1;
                    weight += i128::from(e.weight);
                }
                if valid {
                    max_count = max_count.max(count);
                    max_weight = max_weight.max(weight);
                }
            }
            let count = component(&edges, ApproxMode::CountFirst).unwrap();
            assert_eq!(
                count.0.len(),
                max_count,
                "maximum cardinality for mask {mask}"
            );
            let score = component(&edges, ApproxMode::ScoreFirst).unwrap();
            for mode in [
                ApproxMode::ScoreFirst,
                ApproxMode::CountFirst,
                ApproxMode::Best,
                ApproxMode::Hybrid,
            ] {
                let out = component(&edges, mode).unwrap();
                assert!(
                    out.1 <= max_weight && max_weight <= out.2,
                    "certified bound for mask {mask}"
                );
                assert_eq!(
                    component(&edges, mode).unwrap(),
                    out,
                    "deterministic for mask {mask}"
                );
                if matches!(mode, ApproxMode::Best | ApproxMode::Hybrid) {
                    assert_eq!(
                        out.1,
                        score.1.max(count.1),
                        "best objective for mask {mask}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_ordinals_and_long_iterative_augmentation() {
        let n = 2048;
        let mut values = Vec::new();
        for b in 0..n - 1 {
            values.push((usize::MAX - n + b, usize::MAX - n + b, 2));
            values.push((usize::MAX - n + b, usize::MAX - n + b + 1, 1));
        }
        values.push((usize::MAX - 1, usize::MAX - n, 1));
        let out = component(&make_edges(&values), ApproxMode::CountFirst).unwrap();
        assert_eq!(
            out.0.len(),
            n,
            "long alternating path reaches a perfect matching"
        );
        assert_eq!(
            out.1, n as i128,
            "long path has only unit-weight selected edges"
        );
    }

    #[test]
    fn certification_rejects_duplicate_selection() {
        let edges = make_edges(&[(0, 0, 1)]);
        let graph = Graph::build(&edges).unwrap();
        assert!(
            matches!(
                graph.certify(&edges, vec![0, 0], 1),
                Err(SolverError::Invariant(_))
            ),
            "duplicate edges fail certification"
        );
        assert!(
            component(&[], ApproxMode::Best).is_err(),
            "empty component is a contract violation"
        );
    }
}
