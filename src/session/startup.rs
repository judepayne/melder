//! Startup graph solving over shared-pipeline candidates, before live requests begin.

use tracing::info;

use crate::batch::assignment::{ScoredQuery, solve};
use crate::config::BatchAssignmentMode;
use crate::error::SessionError;
use crate::models::Side;

use super::Session;

impl Session {
    pub(super) fn startup_query_vector(
        &self,
        id: &str,
        strict: bool,
    ) -> Result<Vec<f32>, SessionError> {
        let index = self.state.side(Side::B).combined_index.as_ref();
        let vector = index
            .map(|index| index.get(id))
            .transpose()
            .map_err(|e| SessionError::Assignment(e.to_string()))?
            .flatten();
        if strict && !self.emb_specs.is_empty() {
            let expected = index.map(|index| index.dim());
            if !matches!((&vector, expected), (Some(vector), Some(dim)) if dim > 0 && vector.len() == dim)
            {
                return Err(SessionError::Assignment(format!(
                    "startup B record {id} has a missing or invalid query vector"
                )));
            }
        }
        Ok(vector.unwrap_or_default())
    }

    pub(super) fn validate_startup_constraints(&self) -> Result<(), SessionError> {
        for (a, b) in self.state.crossmap.pairs() {
            if self.state.exclusions.contains(&a, &b) {
                return Err(SessionError::Assignment(format!(
                    "protected startup pair ({a}, {b}) is excluded"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn finish_startup_graph(
        &self,
        mode: BatchAssignmentMode,
        ids: &[String],
        scored: Vec<ScoredQuery>,
    ) -> Result<(), SessionError> {
        let ids: Vec<_> = ids.iter().map(String::as_str).collect();
        let (outcomes, summary) = solve(
            mode,
            self.state.store.as_ref(),
            self.state.crossmap.as_ref(),
            &ids,
            scored,
        )
        .map_err(|e| SessionError::Assignment(e.to_string()))?;
        self.publish_startup_graph(outcomes)?;
        info!(
            ?mode,
            retained_edges = summary.retained_edges,
            fixed_pairs = summary.fixed_pairs,
            exact_groups = summary.exact_groups,
            approximate_groups = summary.approximate_groups,
            largest_group = summary.largest_group,
            solve_s = summary.solve_secs,
            "startup graph assignment complete"
        );
        Ok(())
    }
}
