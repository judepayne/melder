//! Sparse maximum-weight bipartite assignment with exact and approximate modes.

mod solver;

pub use solver::{
    ApproxMatching, ApproxMode, Limits, Matching, Options, SCALE, SolverError, Stats, WeightedEdge,
    quantize, solve, solve_approx, solve_with_options,
};

#[cfg(test)]
#[path = "tests/approx.rs"]
mod approx_tests;
#[cfg(test)]
#[path = "tests/oracle.rs"]
mod oracle_tests;
