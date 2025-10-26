use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use once_cell::sync::Lazy;
use parallel_csv_reader::{parallel_csv_checksum, sequential_csv_checksum};
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::TempDir;

static DATASET: Lazy<Dataset> = Lazy::new(|| Dataset::new(200_000));

struct Dataset {
    _tempdir: TempDir,
    path: PathBuf,
    rows: usize,
}

impl Dataset {
    fn new(rows: usize) -> Self {
        let tempdir = TempDir::new().expect("tempdir");
        let path = tempdir.path().join("parallel.csv");
        let mut file = File::create(&path).expect("create dataset");
        writeln!(file, "id,value").expect("header");
        for idx in 0..rows {
            writeln!(file, "{idx},{}", idx * 17).expect("row");
        }
        Dataset {
            _tempdir: tempdir,
            path,
            rows: rows + 1, // header + rows
        }
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkSnapshot {
    dataset_rows: usize,
    sequential_microseconds: f64,
    parallel_microseconds: f64,
    speedup: f64,
}

pub fn csv_benchmark(c: &mut Criterion) {
    let dataset_rows = DATASET.rows;

    let mut group = c.benchmark_group("parallel_csv_checksum");
    group.throughput(Throughput::Elements(dataset_rows as u64));

    group.bench_function(BenchmarkId::new("sequential", dataset_rows), |b| {
        b.iter_batched(
            || DATASET.path.clone(),
            |path| {
                let (rows, _) = sequential_csv_checksum(&path).expect("checksum");
                assert_eq!(rows, dataset_rows);
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function(BenchmarkId::new("parallel", dataset_rows), |b| {
        b.iter_batched(
            || DATASET.path.clone(),
            |path| {
                let (rows, _) = parallel_csv_checksum(&path).expect("checksum");
                assert_eq!(rows, dataset_rows);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();

    let sequential_time = {
        let start = Instant::now();
        let (rows, _) = sequential_csv_checksum(DATASET.path.as_path()).expect("checksum");
        assert_eq!(rows, dataset_rows);
        start.elapsed().as_secs_f64() * 1_000_000.0
    };
    let parallel_time = {
        let start = Instant::now();
        let (rows, _) = parallel_csv_checksum(DATASET.path.as_path()).expect("checksum");
        assert_eq!(rows, dataset_rows);
        start.elapsed().as_secs_f64() * 1_000_000.0
    };

    persist_snapshot(BenchmarkSnapshot {
        dataset_rows,
        sequential_microseconds: sequential_time,
        parallel_microseconds: parallel_time,
        speedup: sequential_time / parallel_time,
    });
}

fn persist_snapshot(snapshot: BenchmarkSnapshot) {
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("benchmarks");
    fs::create_dir_all(&output_dir).expect("create benchmarks dir");
    let output = output_dir.join("parallel_csv.json");
    let payload = serde_json::to_string_pretty(&snapshot).expect("serialize snapshot");
    fs::write(output, payload).expect("write snapshot");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(15);
    targets = csv_benchmark
);
criterion_main!(benches);
