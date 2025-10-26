//! Parallel CSV checksum utilities.
//!
//! This crate exposes sequential and Rayon-backed parallel readers that compute
//! deterministic SHA-256 based digests for CSV datasets. The parallel
//! implementation aggregates chunk-level digests via XOR reduction to avoid
//! contention while remaining reproducible.

use anyhow::Result;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use thiserror::Error;

const DEFAULT_CHUNK_SIZE: usize = 10_000;

/// Errors emitted by the CSV checksum routines.
#[derive(Debug, Error)]
pub enum CsvChecksumError {
    /// Raised when the provided path cannot be opened for reading.
    #[error("failed to open CSV at {path}: {source}")]
    IoOpen {
        /// Path that failed to open.
        path: PathBuf,
        /// Source IO error.
        #[source]
        source: io::Error,
    },
    /// Raised when line iteration fails due to an underlying IO error.
    #[error("failed to read CSV rows from {path}: {source}")]
    IoRead {
        /// Path that failed to stream.
        path: PathBuf,
        /// Source IO error.
        #[source]
        source: io::Error,
    },
}

/// Compute a sequential checksum for the CSV file located at `path`.
///
/// The function streams the CSV line-by-line, hashing each entry and
/// returning the total row count alongside a hexadecimal digest string.
pub fn sequential_csv_checksum<P: AsRef<Path>>(path: P) -> Result<(usize, String)> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref).map_err(|source| CsvChecksumError::IoOpen {
        path: path_ref.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut total = 0usize;
    let mut digest = [0u8; 32];
    let mut chunk: Vec<String> = Vec::with_capacity(DEFAULT_CHUNK_SIZE);

    for line in reader.lines() {
        let line = line.map_err(|source| CsvChecksumError::IoRead {
            path: path_ref.to_path_buf(),
            source,
        })?;
        total += 1;
        chunk.push(line);
        if chunk.len() == DEFAULT_CHUNK_SIZE {
            xor_in_place(&mut digest, digest_for_lines(&chunk));
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        xor_in_place(&mut digest, digest_for_lines(&chunk));
    }

    Ok((total, hex::encode(digest)))
}

/// Compute a Rayon-backed parallel checksum for the CSV file located at `path`.
///
/// The implementation loads the CSV into memory once to enable predictable
/// chunking and then performs a parallel reduction over SHA-256 digests. It
/// returns the total row count and the final digest string.
pub fn parallel_csv_checksum<P: AsRef<Path>>(path: P) -> Result<(usize, String)> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref).map_err(|source| CsvChecksumError::IoOpen {
        path: path_ref.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let lines: Vec<String> =
        reader
            .lines()
            .collect::<Result<_, _>>()
            .map_err(|source| CsvChecksumError::IoRead {
                path: path_ref.to_path_buf(),
                source,
            })?;

    let total = lines.len();
    let digest = lines
        .par_chunks(DEFAULT_CHUNK_SIZE.max(1))
        .map(digest_for_lines)
        .reduce(|| [0u8; 32], xor_digests);

    Ok((total, hex::encode(digest)))
}

fn finalize_to_array(hasher: Sha256) -> [u8; 32] {
    let bytes = hasher.finalize();
    let mut buffer = [0u8; 32];
    buffer.copy_from_slice(&bytes);
    buffer
}

fn digest_for_lines(lines: &[String]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for line in lines {
        hasher.update(line.as_bytes());
    }
    finalize_to_array(hasher)
}

fn xor_in_place(target: &mut [u8; 32], rhs: [u8; 32]) {
    for (lhs, r) in target.iter_mut().zip(rhs) {
        *lhs ^= r;
    }
}

fn xor_digests(lhs: [u8; 32], rhs: [u8; 32]) -> [u8; 32] {
    let mut out = lhs;
    xor_in_place(&mut out, rhs);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn write_csv(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("tempfile");
        use std::io::Write;
        file.write_all(contents.as_bytes()).expect("write");
        file
    }

    #[test]
    fn sequential_and_parallel_match() {
        let file = write_csv("a,b,c\n1,2,3\n4,5,6\n");
        let seq = sequential_csv_checksum(file.path()).expect("sequential");
        let par = parallel_csv_checksum(file.path()).expect("parallel");
        assert_eq!(seq.0, 3);
        assert_eq!(seq, par);
    }

    #[test]
    fn handles_large_dataset_consistently() {
        let mut rows = String::from("id,value\n");
        for idx in 0..25_000 {
            rows.push_str(&format!("{idx},{}\n", idx * 3));
        }
        let file = write_csv(&rows);
        let seq = sequential_csv_checksum(file.path()).expect("sequential");
        let par = parallel_csv_checksum(file.path()).expect("parallel");
        assert_eq!(seq.0, 25_001);
        assert_eq!(seq, par);
    }

    #[test]
    fn reports_open_error() {
        let err = sequential_csv_checksum("/non/existent/path.csv").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("failed to open CSV"));
    }
}
