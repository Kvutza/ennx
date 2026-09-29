use super::*;
use futures::TryStreamExt;
use opendal::{Operator, layers, services};
use parquet::arrow::arrow_reader::{ArrowPredicateFn, RowFilter};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::errors::ParquetError;
use parquet_opendal::AsyncReader;

pub(super) async fn collect(job: Job) -> Result<Selection> {
    if job.targets.is_empty()
        || job.targets.iter().any(|(s, b)| {
            !SPLITS.contains(&s.as_str())
                || b.len() != BUCKETS.len()
                || BUCKETS.iter().any(|name| !b.contains_key(*name))
        })
    {
        return Err("invalid corpus split/bucket targets".into());
    }
    let operator = if job.root.starts_with("https://") || job.root.starts_with("http://") {
        Operator::new(services::Http::default().endpoint(&job.root))?.finish()
    } else {
        Operator::new(services::Fs::default().root(&job.root))?.finish()
    }
    .layer(layers::TimeoutLayer::new().with_timeout(Duration::from_secs(60)))
    .layer(layers::RetryLayer::new().with_max_times(3));
    let mut selected = Selection::new(&job.targets);
    let active = Arc::new(AtomicU16::new(selected.active(&job.targets)));
    let started = Instant::now();
    let mut reported = Instant::now();
    for (index, path) in job.paths.iter().enumerate() {
        if active.load(Ordering::Relaxed) == 0 {
            break;
        }
        eprintln!(
            "read | shard {}/{} | {}",
            index + 1,
            job.paths.len(),
            path.rsplit('/').next().unwrap_or(path)
        );
        read_shard(
            &operator,
            path,
            &job,
            &mut selected,
            &active,
            started,
            &mut reported,
        )
        .await?;
    }
    selected.report(&job, started);
    if selected.active(&job.targets) != 0 {
        return Err("source exhausted before corpus quotas were met".into());
    }
    Ok(selected)
}

async fn read_shard(
    operator: &Operator,
    path: &str,
    job: &Job,
    selected: &mut Selection,
    active: &Arc<AtomicU16>,
    started: Instant,
    reported: &mut Instant,
) -> Result<()> {
    let length = operator.stat(path).await?.content_length();
    let reader = operator
        .reader_with(path)
        .gap(512 * 1024)
        .chunk(8 * 1024 * 1024)
        .concurrent(4)
        .await?;
    let builder = ParquetRecordBatchStreamBuilder::new(AsyncReader::new(reader, length)).await?;
    let schema = builder.parquet_schema();
    let projection = projection(schema);
    let filter_mask = ProjectionMask::leaves(
        schema,
        schema.columns().iter().enumerate().filter_map(|(i, c)| {
            (matches!(c.path().parts()[0].as_str(), "repo_path" | "commit_id")
                || (c.path().parts()[0] == "github_metadata" && c.name() == "is_fork"))
                .then_some(i)
        }),
    );
    let mask = active.clone();
    let file_mask = ProjectionMask::leaves(
        schema,
        schema.columns().iter().enumerate().filter_map(|(i, c)| {
            (matches!(c.path().parts()[0].as_str(), "repo_path" | "commit_id")
                || (c.path().parts()[0] == "files"
                    && ["file_path", "language", "license_type", "is_vendor"].contains(&c.name())))
            .then_some(i)
        }),
    );
    let predicate = ArrowPredicateFn::new(filter_mask, move |batch| {
        Ok(eligible(&batch, mask.load(Ordering::Relaxed))
            .map_err(|e| ParquetError::General(e.to_string()))?)
    });
    let mask = active.clone();
    let files = ArrowPredicateFn::new(file_mask, move |batch| {
        Ok(eligible_files(&batch, mask.load(Ordering::Relaxed))
            .map_err(|e| ParquetError::General(e.to_string()))?)
    });
    let mut stream = builder
        .with_batch_size(64)
        .with_projection(projection)
        .with_row_filter(RowFilter::new(vec![Box::new(predicate), Box::new(files)]))
        .build()?;
    while let Some(batch) = stream.try_next().await? {
        selected.accept(&batch, &job)?;
        active.store(selected.active(&job.targets), Ordering::Relaxed);
        if reported.elapsed() >= Duration::from_secs(5) {
            selected.report(&job, started);
            *reported = Instant::now();
        }
        if active.load(Ordering::Relaxed) == 0 {
            break;
        }
    }
    Ok(())
}

fn projection(schema: &parquet::schema::types::SchemaDescriptor) -> ProjectionMask {
    ProjectionMask::leaves(
        schema,
        schema.columns().iter().enumerate().filter_map(|(i, c)| {
            let parts = c.path().parts();
            let keep = match parts[0].as_str() {
                "repo_path" | "commit_id" => true,
                "github_metadata" => c.name() == "is_fork",
                "files" => [
                    "file_path",
                    "language",
                    "license_type",
                    "is_vendor",
                    "content_id",
                    "content",
                ]
                .contains(&c.name()),
                _ => false,
            };
            keep.then_some(i)
        }),
    )
}
