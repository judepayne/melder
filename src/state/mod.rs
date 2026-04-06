//! State management: loading datasets, building/loading caches, and managing
//! the composite `MatchState` that holds everything needed for matching.

pub mod live;
pub mod match_log;
#[allow(clippy::module_inception)]
pub mod state;

pub use live::LiveMatchState;
pub use match_log::{MatchLog, MatchLogEvent};
pub use state::{LoadOptions, MatchState};
