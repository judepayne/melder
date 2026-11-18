//! Batch additional strictly improving paths at the current feasible potentials.

use super::{Handle, Network, Result, SOURCE, SolverError, filled, reserve, sub};

const UNREACHED: usize = usize::MAX;

pub(super) struct Admissible {
    level: Vec<usize>,
    cursor: Vec<usize>,
    queue: Vec<usize>,
    path: Vec<Handle>,
}

impl Admissible {
    pub(super) fn new(vertices: usize) -> Result<Self> {
        Ok(Self {
            level: filled(vertices, UNREACHED)?,
            cursor: filled(vertices, 0)?,
            queue: reserve(vertices)?,
            path: reserve(vertices)?,
        })
    }

    pub(super) fn augment(&mut self, net: &mut Network, cost: i128) -> Result<()> {
        // Every tight path telescopes to this same original (negative) cost.
        if cost >= 0 || cost != sub(net.pi[net.sink], net.pi[SOURCE])? {
            return Err(SolverError::Invariant("invalid batch path cost"));
        }
        while self.build_levels(net)? {
            self.blocking_flow(net, cost)?;
        }
        Ok(())
    }

    fn build_levels(&mut self, net: &Network) -> Result<bool> {
        self.level.fill(UNREACHED);
        self.queue.clear();
        self.level[SOURCE] = 0;
        self.queue.push(SOURCE);
        let mut next = 0;
        while next < self.queue.len() {
            let u = self.queue[next];
            next += 1;
            for &arc in &net.adj[u] {
                if arc.residual == 0 || self.level[arc.to] != UNREACHED || net.reduced(u, arc)? != 0
                {
                    continue;
                }
                self.level[arc.to] = self.level[u] + 1;
                // All vertices at the preceding level have already been discovered.
                if arc.to == net.sink {
                    return Ok(true);
                }
                self.queue.push(arc.to);
            }
        }
        Ok(false)
    }

    fn blocking_flow(&mut self, net: &mut Network, cost: i128) -> Result<()> {
        self.cursor.fill(0);
        self.path.clear();
        let mut u = SOURCE;
        loop {
            if u == net.sink {
                let mut v = net.sink;
                for &(from, slot) in self.path.iter().rev() {
                    net.flip(from, slot, v)?;
                    v = from;
                }
                net.record_augmentation(cost)?;
                // Unit capacities saturate every arc on the path. Each new
                // reverse arc points to an earlier level and isn't usable here.
                self.path.clear();
                u = SOURCE;
                continue;
            }
            let slot = self.cursor[u];
            if let Some(&arc) = net.adj[u].get(slot) {
                self.cursor[u] += 1;
                if arc.residual > 0
                    && self.level[arc.to] == self.level[u] + 1
                    && net.reduced(u, arc)? == 0
                {
                    self.path.push((u, slot));
                    u = arc.to;
                }
            } else if let Some((from, _)) = self.path.pop() {
                u = from;
            } else {
                return Ok(());
            }
        }
    }
}
