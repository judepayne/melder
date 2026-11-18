//! Strict CSV parsing shared by graph-mode protections and exclusions.

use std::collections::HashSet;
use std::path::Path;

use crate::error::CrossMapError;

/// Validate the entire constraint CSV before returning its trimmed pairs.
/// Only crossmaps allow a missing file to represent a fresh mapping.
pub(crate) fn load_pairs(
    path: &Path,
    a_field: &str,
    b_field: &str,
    allow_missing: bool,
) -> Result<Vec<(String, String)>, CrossMapError> {
    let invalid = |message: String| {
        CrossMapError::InvalidConstraint(format!("{}: {message}", path.display()))
    };
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if allow_missing && e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Vec::new());
        }
        Err(e) => return Err(e.into()),
    };
    if a_field.is_empty() || b_field.is_empty() || a_field == b_field {
        return Err(invalid(
            "ID column names must be nonempty and distinct".into(),
        ));
    }
    let mut reader = csv::Reader::from_reader(file);
    let headers = reader.headers()?.clone();
    let mut seen = HashSet::new();
    for header in &headers {
        let header = header.trim();
        if header.is_empty() || !seen.insert(header) {
            return Err(invalid(format!("empty or duplicate header {header:?}")));
        }
    }
    let a_idx = headers.iter().position(|h| h.trim() == a_field);
    let b_idx = headers.iter().position(|h| h.trim() == b_field);
    let (Some(a_idx), Some(b_idx)) = (a_idx, b_idx) else {
        return Err(invalid(format!(
            "required columns {a_field:?} and {b_field:?} not found"
        )));
    };
    let mut pairs = Vec::new();
    for result in reader.records() {
        let row = result?;
        let a_id = row.get(a_idx).unwrap_or("").trim();
        let b_id = row.get(b_idx).unwrap_or("").trim();
        if a_id.is_empty() || b_id.is_empty() {
            let line = row.position().map(|p| p.line()).unwrap_or(0);
            return Err(invalid(format!("empty ID at line {line}")));
        }
        pairs.push((a_id.to_string(), b_id.to_string()));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_headers_and_rows() {
        let dir = tempfile::tempdir().expect("temp directory");
        let path = dir.path().join("constraints.csv");
        for contents in [
            "",
            "wrong,columns\n",
            "a,wrong\n",
            "a,b,a\n",
            "a,b, b \n",
            "a,b,\n",
            "a,b\n,b1\n",
            "a,b\na1, \n",
            "a,b\na1\n",
            "a,b\na1,b1,extra\n",
            "a,b\na1,b1\na2,\n",
        ] {
            std::fs::write(&path, contents).expect("write CSV");
            assert!(
                load_pairs(&path, "a", "b", false).is_err(),
                "must reject {contents:?}",
            );
        }
        std::fs::write(&path, " a , b ,note\n a1 , b1 ,extra\n").expect("write CSV");
        assert_eq!(
            load_pairs(&path, "a", "b", false).expect("valid CSV"),
            vec![("a1".into(), "b1".into())],
            "trim IDs and accept additional named columns",
        );
        std::fs::write(&path, "a,b\n").expect("write CSV");
        assert!(
            load_pairs(&path, "a", "b", false)
                .expect("valid header")
                .is_empty(),
            "a valid header-only CSV is empty",
        );
        assert!(
            load_pairs(&path, "a", "a", false).is_err(),
            "distinct columns required"
        );
    }
}
