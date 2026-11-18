//! Standalone benchmark adapter for the production assignment solver.

#[path = "../../../../src/matching/assignment/mod.rs"]
mod assignment;

pub use assignment::{
    ApproxMatching, ApproxMode, Limits, Matching, Options, SCALE, SolverError, Stats, WeightedEdge,
    quantize, solve, solve_approx, solve_with_options,
};
