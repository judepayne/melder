//! Bounded, scoped component execution with canonical result and error ordering.

use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;

use super::{Result, SolverError, reserve};

/// Map canonical component indices, independent of thread execution order.
/// Errors are selected in worker order; every spawned worker is joined, including
/// after a work error or spawn failure. Work must not depend on scheduling order.
pub(super) fn map_components<T: Send, F: Fn(usize) -> Result<T> + Sync>(
    jobs: usize,
    workers: NonZeroUsize,
    work: F,
) -> Result<Vec<T>> {
    let workers = workers.get().min(jobs);
    if workers <= 1 {
        return run_batch(0, 1, jobs, &work);
    }

    let mut output = reserve(jobs)?;
    // The caller orders large jobs first. Interleave that order across workers
    // so one worker doesn't get all the expensive components.
    let mut batches = reserve(workers)?;
    thread::scope(|scope| {
        let mut handles = reserve(workers)?;
        let mut spawn_error = None;
        for worker in 0..workers {
            let work = &work;
            match thread::Builder::new()
                .spawn_scoped(scope, move || run_batch(worker, workers, jobs, work))
            {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    spawn_error = Some(SolverError::Invariant("component worker spawn failed"));
                    break;
                }
            }
        }

        let mut first_error = None;
        for handle in handles {
            let batch = handle
                .join()
                .unwrap_or(Err(SolverError::Invariant("component worker panicked")));
            match batch {
                Ok(values) => batches.push(values.into_iter()),
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        if let Some(error) = spawn_error.or(first_error) {
            return Err(error);
        }
        for index in 0..jobs {
            output.push(
                batches[index % workers]
                    .next()
                    .ok_or(SolverError::Invariant("component result missing"))?,
            );
        }
        Ok(output)
    })
}

fn run_batch<T, F: Fn(usize) -> Result<T>>(
    start: usize,
    step: usize,
    end: usize,
    work: &F,
) -> Result<Vec<T>> {
    // Keep panic handling consistent between the serial and threaded paths.
    catch_unwind(AssertUnwindSafe(|| {
        let indices = (start..end).step_by(step);
        let mut values = reserve(indices.len())?;
        for index in indices {
            values.push(work(index)?);
        }
        Ok(values)
    }))
    .unwrap_or(Err(SolverError::Invariant("component worker panicked")))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    fn workers(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).expect("test worker count is positive")
    }

    #[test]
    fn empty_and_serial_paths() {
        let empty = map_components::<usize, _>(0, workers(8), |_| {
            panic!("empty input must not invoke work")
        });
        assert_eq!(empty, Ok(Vec::new()), "empty input must return no results");
        assert_eq!(
            map_components(5, workers(1), |index| Ok(index * 2)),
            Ok(vec![0, 2, 4, 6, 8]),
            "serial results must follow index order"
        );
    }

    #[test]
    fn canonical_order_and_exactly_once() {
        for count in [1, 2, 3, 8, 64] {
            let calls = AtomicUsize::new(0);
            let values = map_components(17, workers(count), |index| {
                calls.fetch_add(1, Ordering::Relaxed);
                if index == 0 {
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(Box::new(index))
            })
            .expect("mapping must succeed");
            assert_eq!(
                values.into_iter().map(|value| *value).collect::<Vec<_>>(),
                (0..17).collect::<Vec<_>>(),
                "results must not follow completion order"
            );
            assert_eq!(calls.load(Ordering::Relaxed), 17, "visit every job once");
        }
    }

    #[test]
    fn stable_error_and_all_workers_joined() {
        let completed = AtomicUsize::new(0);
        let result = map_components(6, workers(3), |index| {
            if index == 0 {
                thread::sleep(Duration::from_millis(10));
                return Err(SolverError::Overflow);
            }
            if index == 2 {
                return Err(SolverError::Allocation);
            }
            completed.fetch_add(1, Ordering::Relaxed);
            Ok(index)
        });
        assert_eq!(result, Err(SolverError::Overflow), "first index error wins");
        assert_eq!(
            completed.load(Ordering::Relaxed),
            2,
            "successful batch must finish before returning"
        );
    }

    #[test]
    fn panics_become_errors_in_both_paths() {
        for count in [1, 3] {
            let result = map_components::<usize, _>(3, workers(count), |_| panic!("test panic"));
            assert_eq!(
                result,
                Err(SolverError::Invariant("component worker panicked")),
                "work panics must not escape"
            );
        }
    }
}
