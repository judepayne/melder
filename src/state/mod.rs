//! State management: loading datasets, building/loading caches, and managing
//! the composite `MatchState` that holds everything needed for matching.

pub mod live;
#[allow(clippy::module_inception)]
pub mod state;
pub mod upsert_log;

pub use live::LiveMatchState;
pub use state::{LoadOptions, MatchState};
pub use upsert_log::{UpsertLog, WalEvent};
