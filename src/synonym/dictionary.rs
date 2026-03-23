//! User-provided synonym dictionary.
//!
//! Loads equivalence groups from a CSV file where each row contains two or
//! more terms that should be treated as synonyms. All terms in a row map
//! bidirectionally to each other. Transitive groups are merged: if row 1
//! has `A,B` and row 2 has `B,C`, then A, B, and C are all equivalent.

use std::collections::HashMap;
use std::path::Path;

use tracing::warn;

use crate::error::DataError;

/// A synonym dictionary loaded from a CSV file.
///
/// Maps each normalised term to its full equivalence set (excluding self).
pub struct SynonymDictionary {
    equivalences: HashMap<String, Vec<String>>,
    group_count: usize,
}

impl std::fmt::Debug for SynonymDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynonymDictionary")
            .field("groups", &self.group_count)
            .field("terms", &self.equivalences.len())
            .finish()
    }
}

impl SynonymDictionary {
    /// Load a synonym dictionary from a CSV file.
    ///
    /// Each row is an equivalence group: all non-empty terms in the row are
    /// synonyms of each other. Rows with fewer than 2 terms are skipped.
    /// Terms are normalised to uppercase with leading/trailing whitespace
    /// trimmed. Transitive groups are merged automatically.
    pub fn load(path: &Path) -> Result<Self, DataError> {
        if !path.exists() {
            return Err(DataError::NotFound {
                path: path.display().to_string(),
            });
        }

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(path)?;

        // Collect raw groups from CSV rows.
        let mut raw_groups: Vec<Vec<String>> = Vec::new();
        for (row_idx, result) in reader.records().enumerate() {
            let record = result?;
            let terms: Vec<String> = record
                .iter()
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty())
                .collect();

            if terms.len() < 2 {
                if terms.len() == 1 {
                    warn!(row = row_idx + 1, term = %terms[0], "skipping single-term row");
                }
                continue;
            }
            raw_groups.push(terms);
        }

        // Merge transitive groups using union-find.
        // Map each term to a group index. If a term appears in multiple rows,
        // merge those groups.
        let mut term_to_group: HashMap<String, usize> = HashMap::new();
        let mut groups: Vec<Vec<String>> = Vec::new();
        // parent[i] = canonical group index for group i
        let mut parent: Vec<usize> = Vec::new();

        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]]; // path compression
                i = parent[i];
            }
            i
        }

        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }

        for terms in &raw_groups {
            // Find if any term already belongs to an existing group.
            let mut existing_group: Option<usize> = None;
            for term in terms {
                if let Some(&gid) = term_to_group.get(term) {
                    let root = find(&mut parent, gid);
                    match existing_group {
                        None => existing_group = Some(root),
                        Some(eg) => union(&mut parent, eg, root),
                    }
                }
            }

            let gid = match existing_group {
                Some(eg) => find(&mut parent, eg),
                None => {
                    let new_id = groups.len();
                    groups.push(Vec::new());
                    parent.push(new_id);
                    new_id
                }
            };

            // Add all terms to this group and update the mapping.
            for term in terms {
                term_to_group.insert(term.clone(), gid);
            }
        }

        // Collect final merged groups.
        let mut merged: HashMap<usize, Vec<String>> = HashMap::new();
        for (term, gid) in &term_to_group {
            let root = find(&mut parent, *gid);
            merged.entry(root).or_default().push(term.clone());
        }

        // Build the equivalences map.
        let mut equivalences: HashMap<String, Vec<String>> = HashMap::new();
        let group_count = merged.len();

        for (_root, members) in &merged {
            for term in members {
                let others: Vec<String> = members.iter().filter(|m| *m != term).cloned().collect();
                equivalences.insert(term.clone(), others);
            }
        }

        Ok(Self {
            equivalences,
            group_count,
        })
    }

    /// Return all equivalent terms for the given input (excluding itself).
    ///
    /// Returns an empty slice if the term is not in the dictionary.
    pub fn expand(&self, term: &str) -> &[String] {
        let key = term.trim().to_uppercase();
        self.equivalences
            .get(&key)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Check if two terms are in the same equivalence group.
    pub fn is_equivalent(&self, a: &str, b: &str) -> bool {
        let a_key = a.trim().to_uppercase();
        let b_key = b.trim().to_uppercase();
        if a_key == b_key {
            return false; // same term is not a "synonym match"
        }
        self.equivalences
            .get(&a_key)
            .map(|equivs| equivs.iter().any(|e| *e == b_key))
            .unwrap_or(false)
    }

    /// Number of equivalence groups in the dictionary.
    pub fn len(&self) -> usize {
        self.group_count
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.group_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn load_basic() {
        let f = write_csv(
            "HSBC,Hongkong and Shanghai Banking Corporation\nJPM,JP Morgan,JPMorgan Chase\n",
        );
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert_eq!(dict.len(), 2, "two groups");
        assert!(!dict.is_empty());

        // HSBC expands to the full name
        let expanded = dict.expand("HSBC");
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0], "HONGKONG AND SHANGHAI BANKING CORPORATION");

        // JPM expands to two terms
        let expanded = dict.expand("JPM");
        assert_eq!(expanded.len(), 2, "JPM has 2 equivalents: {:?}", expanded);
    }

    #[test]
    fn case_insensitive() {
        let f = write_csv("hsbc,HONGKONG AND SHANGHAI BANKING CORPORATION\n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert!(dict.is_equivalent("HSBC", "Hongkong and Shanghai Banking Corporation"));
        assert!(dict.is_equivalent("hsbc", "HONGKONG AND SHANGHAI BANKING CORPORATION"));
    }

    #[test]
    fn bidirectional() {
        let f = write_csv("IBM,International Business Machines\n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert!(dict.is_equivalent("IBM", "International Business Machines"));
        assert!(dict.is_equivalent("International Business Machines", "IBM"));
    }

    #[test]
    fn transitive_merge() {
        let f = write_csv("A,B\nB,C\n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert_eq!(dict.len(), 1, "should merge into one group");
        assert!(dict.is_equivalent("A", "B"));
        assert!(dict.is_equivalent("A", "C"));
        assert!(dict.is_equivalent("B", "C"));
        assert!(dict.is_equivalent("C", "A"));
    }

    #[test]
    fn single_term_row_skipped() {
        let f = write_csv("lonely\nIBM,International Business Machines\n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert_eq!(dict.len(), 1, "single-term row should be skipped");
        assert!(dict.expand("LONELY").is_empty());
    }

    #[test]
    fn empty_cells_ignored() {
        let f = write_csv("IBM,,International Business Machines,,\n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert_eq!(dict.expand("IBM").len(), 1);
    }

    #[test]
    fn same_term_not_equivalent() {
        let f = write_csv("IBM,International Business Machines\n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert!(!dict.is_equivalent("IBM", "IBM"));
    }

    #[test]
    fn file_not_found() {
        let result = SynonymDictionary::load(Path::new("/nonexistent/synonyms.csv"));
        assert!(result.is_err());
    }

    #[test]
    fn whitespace_trimmed() {
        let f = write_csv("  HSBC  ,  Hongkong and Shanghai Banking Corporation  \n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert!(dict.is_equivalent("HSBC", "Hongkong and Shanghai Banking Corporation"));
    }

    #[test]
    fn variable_column_count() {
        let f = write_csv("A,B\nC,D,E,F\nG,H,I\n");
        let dict = SynonymDictionary::load(f.path()).unwrap();

        assert_eq!(dict.len(), 3, "three independent groups");
        assert_eq!(dict.expand("C").len(), 3, "C has 3 equivalents");
    }
}
