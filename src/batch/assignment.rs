//! Bridge between batch scoring and deterministic assignment. No scoring happens here.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::num::NonZeroUsize;

use crate::config::{BatchAssignmentMode, Config};
use crate::crossmap::CrossMapOps;
use crate::error::MelderError;
use crate::matching::assignment::{
    ApproxMatching, ApproxMode, Limits, Options, WeightedEdge, quantize, solve_approx,
    solve_with_options,
};
use crate::matching::exclusions::Exclusions;
use crate::models::{Classification, MatchResult, Record, Side};
use crate::state::match_log::{MatchLog, MatchLogEvent};
use crate::store::RecordStore;

use super::engine::RecordOutcome;

#[cfg(test)]
#[path = "assignment_fault_tests.rs"]
mod fault_tests;

const EXACT_COMPONENT_LIMIT: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum BatchAssignmentError {
    #[error(transparent)]
    Solver(#[from] crate::matching::assignment::SolverError),
    #[error("B-side vector lookup: {0}")]
    Vector(#[from] crate::vectordb::VectorDBError),
    #[error("assignment log: {0}")]
    Log(#[from] std::io::Error),
    #[error("missing {side:?} record {id}")]
    MissingRecord { side: Side, id: String },
    #[error("protected pair ({a}, {b}) is also excluded")]
    ExcludedProtected { a: String, b: String },
    #[error("{0}")]
    Invariant(&'static str),
    #[error("failed to reverse unpublished graph links after log failure ({log}): {rollback}")]
    Rollback { log: String, rollback: String },
}

#[derive(Debug)]
pub struct AssignmentSummary {
    pub mode: BatchAssignmentMode,
    pub retained_edges: usize,
    pub fixed_pairs: usize,
    pub solve_secs: f64,
    pub exact_groups: usize,
    pub approximate_groups: usize,
    pub largest_group: usize,
    pub objective_units: i128,
    pub upper_units: i128,
}

impl AssignmentSummary {
    /// Certified lower bound in tenths of one percent, rounded down.
    pub fn guaranteed_tenths(&self) -> Option<i128> {
        (self.upper_units > 0).then(|| 1000 * self.objective_units / self.upper_units)
    }
}

pub(super) fn record(store: &dyn RecordStore, side: Side, id: &str) -> Result<Record, MelderError> {
    store.get(side, id)?.ok_or_else(|| {
        BatchAssignmentError::MissingRecord {
            side,
            id: id.into(),
        }
        .into()
    })
}

pub(super) fn emit(
    log: Option<&MatchLog>,
    event: &MatchLogEvent,
) -> Result<(), BatchAssignmentError> {
    if let Some(log) = log {
        log.append(event)?;
    }
    Ok(())
}

pub(super) fn confirm(result: &MatchResult) -> MatchLogEvent {
    MatchLogEvent::CrossMapConfirm {
        a_id: result.matched_id.clone(),
        b_id: result.query_id.clone(),
        score: Some(result.score),
        rank: result.rank,
        reason: result.reason.clone(),
    }
}

pub(super) fn prematch(
    config: &Config,
    store: &dyn RecordStore,
    crossmap: &dyn CrossMapOps,
    exclusions: &Exclusions,
    b_ids: &[String],
    log: Option<&MatchLog>,
) -> Result<Vec<MatchResult>, MelderError> {
    let common = config
        .datasets
        .a
        .common_id_field
        .as_deref()
        .zip(config.datasets.b.common_id_field.as_deref());
    let exact = config.exact_prefilter.enabled && !config.exact_prefilter.fields.is_empty();
    if common.is_none() && !exact {
        return Ok(Vec::new());
    }
    let mut a_ids = store.ids(Side::A)?;
    a_ids.sort_unstable();
    let mut common_index: HashMap<String, Vec<&str>> = HashMap::new();
    let mut exact_index: HashMap<Vec<String>, Vec<&str>> = HashMap::new();
    for a in &a_ids {
        if crossmap.has_a(a) {
            continue;
        }
        let rec = record(store, Side::A, a)?;
        if let Some((field, _)) = common
            && let Some(value) = rec.get(field).map(|v| v.trim()).filter(|v| !v.is_empty())
        {
            common_index.entry(value.to_owned()).or_default().push(a);
        }
        if exact && let Some(key) = exact_key(config, &rec, Side::A) {
            exact_index.entry(key).or_default().push(a);
        }
    }
    let mut matched = Vec::new();
    // Common IDs take priority globally, not just within each B's choices.
    for reason in ["canonical", "exact"] {
        if (reason == "canonical" && common.is_none()) || (reason == "exact" && !exact) {
            continue;
        }
        for b in b_ids {
            if crossmap.has_b(b) {
                continue;
            }
            let rec = record(store, Side::B, b)?;
            let candidates = if reason == "canonical" {
                common
                    .and_then(|(_, field)| rec.get(field))
                    .and_then(|v| common_index.get(v.trim()))
            } else {
                exact_key(config, &rec, Side::B).and_then(|key| exact_index.get(&key))
            };
            let Some(a) = candidates.and_then(|ids| {
                ids.iter()
                    .copied()
                    .find(|a| !crossmap.has_a(a) && !exclusions.contains(a, b))
            }) else {
                continue;
            };
            let matched_record = Some(record(store, Side::A, a)?);
            if !crossmap.apply_assignment_delta(&[], &[(a, b)])? {
                return Err(
                    BatchAssignmentError::Invariant("backend refused protected pre-match").into(),
                );
            }
            let result = MatchResult {
                query_id: b.clone(),
                matched_id: a.into(),
                query_side: Side::B,
                score: 1.0,
                field_scores: vec![],
                classification: Classification::Auto,
                matched_record,
                from_crossmap: false,
                rank: None,
                reason: Some(reason.into()),
            };
            emit(log, &confirm(&result))?;
            matched.push(result);
        }
    }
    Ok(matched)
}

fn exact_key(config: &Config, rec: &Record, side: Side) -> Option<Vec<String>> {
    config
        .exact_prefilter
        .fields
        .iter()
        .map(|pair| {
            let field = if side == Side::A {
                &pair.field_a
            } else {
                &pair.field_b
            };
            let value = rec.get(field)?.trim();
            (!value.is_empty()).then(|| value.to_lowercase())
        })
        .collect()
}

pub(crate) struct ScoredQuery {
    pub best: Option<MatchResult>,
    pub eligible: Vec<MatchResult>,
}

pub(crate) fn retain(
    config: &Config,
    crossmap: &dyn CrossMapOps,
    mut results: Vec<MatchResult>,
) -> Result<ScoredQuery, BatchAssignmentError> {
    for candidate in &mut results {
        if !(0.0..=1.0).contains(&candidate.score) {
            return Err(crate::matching::assignment::SolverError::InvalidScore.into());
        }
        candidate.matched_record = None;
    }
    let best = results.first().cloned();
    let mut eligible = Vec::new();
    if best
        .as_ref()
        .is_some_and(|r| r.classification == Classification::Auto)
    {
        eligible.extend(
            results
                .into_iter()
                .take_while(|r| {
                    r.classification == Classification::Auto
                        && r.score >= config.thresholds.auto_match
                })
                .filter(|r| !crossmap.has_a(&r.matched_id)),
        );
    }
    Ok(ScoredQuery { best, eligible })
}

pub(super) fn finish(
    mode: BatchAssignmentMode,
    store: &dyn RecordStore,
    crossmap: &dyn CrossMapOps,
    work_ids: &[&str],
    scored: Vec<ScoredQuery>,
    log: Option<&MatchLog>,
) -> Result<(Vec<RecordOutcome>, AssignmentSummary), MelderError> {
    let (outcomes, summary) = solve(mode, store, crossmap, work_ids, scored)?;
    let pairs: Vec<_> = outcomes
        .iter()
        .filter_map(|o| match o {
            RecordOutcome::Auto(r) => Some((r.matched_id.as_str(), r.query_id.as_str())),
            _ => None,
        })
        .collect();
    publish_changes(
        &pairs,
        |expected, replacements| crossmap.apply_assignment_delta(expected, replacements),
        || match log {
            Some(log) => publish_log(log, &outcomes, &summary),
            None => Ok(()),
        },
    )?;
    Ok((outcomes, summary))
}

/// Resolve retained candidates without publishing; live startup owns its own WAL handoff.
pub(crate) fn solve(
    mode: BatchAssignmentMode,
    store: &dyn RecordStore,
    crossmap: &dyn CrossMapOps,
    work_ids: &[&str],
    scored: Vec<ScoredQuery>,
) -> Result<(Vec<RecordOutcome>, AssignmentSummary), MelderError> {
    if scored.len() != work_ids.len() {
        return Err(BatchAssignmentError::Invariant("missing graph query result").into());
    }
    let a_ids: BTreeSet<&str> = scored
        .iter()
        .flat_map(|q| q.eligible.iter().map(|r| r.matched_id.as_str()))
        .collect();
    let a_index: HashMap<_, _> = a_ids
        .into_iter()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    let mut edges = Vec::new();
    for (b, q) in scored.iter().enumerate() {
        for r in &q.eligible {
            edges.push(WeightedEdge {
                b,
                a: a_index[r.matched_id.as_str()],
                weight: quantize(r.score).map_err(BatchAssignmentError::from)?,
                payload: edges.len(),
            });
        }
    }
    let workers = NonZeroUsize::new(rayon::current_num_threads()).unwrap_or(NonZeroUsize::MIN);
    let solve_start = std::time::Instant::now();
    let solved = match mode {
        BatchAssignmentMode::Approximate => solve_approx(
            work_ids.len(),
            a_index.len(),
            &edges,
            Limits::default(),
            ApproxMode::Hybrid,
            workers,
            EXACT_COMPONENT_LIMIT,
        ),
        BatchAssignmentMode::Exact => solve_with_options(
            work_ids.len(),
            a_index.len(),
            &edges,
            Limits::default(),
            Options {
                workers,
                batch_admissible: true,
            },
        )
        .map(|result| ApproxMatching {
            selected_payloads: result.selected_payloads,
            objective_units: result.objective_units,
            upper_units: result.objective_units,
            exact_components: result.stats.components,
            approx_components: 0,
            largest_component_vertices: result.stats.largest_component_vertices,
        }),
    }
    .map_err(BatchAssignmentError::from)?;
    let summary = AssignmentSummary {
        mode,
        retained_edges: edges.len(),
        fixed_pairs: crossmap.len(),
        solve_secs: solve_start.elapsed().as_secs_f64(),
        exact_groups: solved.exact_components,
        approximate_groups: solved.approx_components,
        largest_group: solved.largest_component_vertices,
        objective_units: solved.objective_units,
        upper_units: solved.upper_units,
    };
    let mut selected = vec![None; work_ids.len()];
    let mut used_a = HashSet::new();
    let mut weight = 0i128;
    for payload in solved.selected_payloads {
        let edge = edges
            .get(payload)
            .ok_or(BatchAssignmentError::Invariant("invalid solver payload"))?;
        if selected[edge.b].replace(payload).is_some() || !used_a.insert(edge.a) {
            return Err(
                BatchAssignmentError::Invariant("solver selection violates bijection").into(),
            );
        }
        weight += i128::from(edge.weight);
    }
    if weight != summary.objective_units || weight > summary.upper_units {
        return Err(BatchAssignmentError::Invariant(
            "solver selection disagrees with score certificate",
        )
        .into());
    }
    drop(a_index);
    let mut outcomes = Vec::with_capacity(work_ids.len());
    let mut offset = 0;
    for (b, (query, selected)) in scored.into_iter().zip(selected).enumerate() {
        let id = work_ids[b];
        let width = query.eligible.len();
        if let Some(payload) = selected {
            let mut result = query.eligible.into_iter().nth(payload - offset).ok_or(
                BatchAssignmentError::Invariant("solver payload not in query"),
            )?;
            result.matched_record = Some(record(store, Side::A, &result.matched_id)?);
            result.reason = Some("graph_assignment".into());
            outcomes.push(RecordOutcome::Auto(result));
        } else if let Some(mut best) = query
            .best
            .as_ref()
            .filter(|r| r.classification != Classification::NoMatch)
            .cloned()
        {
            if best.classification == Classification::Auto {
                best.reason = Some("assignment_conflict".into());
            }
            best.classification = Classification::Review;
            outcomes.push(RecordOutcome::Review(best));
        } else {
            let best_score = query.best.map(|r| r.score);
            outcomes.push(RecordOutcome::NoMatch(
                id.to_owned(),
                record(store, Side::B, id)?,
                best_score,
            ));
        }
        offset += width;
    }
    Ok((outcomes, summary))
}

fn publish_changes(
    pairs: &[(&str, &str)],
    mut apply: impl FnMut(&[(&str, &str)], &[(&str, &str)]) -> Result<bool, crate::error::CrossMapError>,
    write_log: impl FnOnce() -> Result<(), BatchAssignmentError>,
) -> Result<(), MelderError> {
    if !pairs.is_empty() && !apply(&[], pairs)? {
        return Err(
            BatchAssignmentError::Invariant("backend refused final graph assignment").into(),
        );
    }
    if let Err(error) = write_log() {
        if !pairs.is_empty() {
            let rollback = match apply(pairs, &[]) {
                Ok(true) => None,
                Ok(false) => Some("backend refused rollback".to_owned()),
                Err(e) => Some(e.to_string()),
            };
            if let Some(rollback) = rollback {
                return Err(BatchAssignmentError::Rollback {
                    log: error.to_string(),
                    rollback,
                }
                .into());
            }
        }
        return Err(error.into());
    }
    Ok(())
}

fn publish_log(
    log: &MatchLog,
    outcomes: &[RecordOutcome],
    summary: &AssignmentSummary,
) -> Result<(), BatchAssignmentError> {
    for outcome in outcomes {
        log.append(&match outcome {
            RecordOutcome::Scored(_) => {
                return Err(BatchAssignmentError::Invariant(
                    "unresolved scoring outcome",
                ));
            }
            RecordOutcome::Auto(r) => confirm(r),
            RecordOutcome::Review(r) => MatchLogEvent::ReviewMatch {
                id: r.query_id.clone(),
                side: Side::B,
                candidate_id: r.matched_id.clone(),
                score: r.score,
                rank: r.rank,
                reason: r.reason.clone(),
            },
            RecordOutcome::NoMatch(id, _, score) => MatchLogEvent::NoMatchBelow {
                query_id: id.clone(),
                query_side: Side::B,
                best_candidate_id: None,
                best_score: *score,
            },
        })?;
    }
    log.complete_batch_assignment(&MatchLogEvent::BatchAssignmentCompleted {
        retained_edges: summary.retained_edges,
        fixed_pairs: summary.fixed_pairs,
        scale: crate::matching::assignment::SCALE,
        exact_groups: summary.exact_groups,
        approximate_groups: summary.approximate_groups,
        largest_group: summary.largest_group,
        objective_units: summary.objective_units,
        upper_units: summary.upper_units,
    })?;
    Ok(())
}
