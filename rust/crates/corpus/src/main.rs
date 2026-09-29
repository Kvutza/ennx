use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufReader, BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::{Duration, Instant};

use arrow_array::{Array, BooleanArray, ListArray, RecordBatch, StringArray, StructArray};
use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

mod reader;
use reader::collect;
type Counts = BTreeMap<String, BTreeMap<String, usize>>;
const BUCKETS: [&str; 4] = ["implementation", "tests", "documentation", "configuration"];
const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Deserialize)]
struct Job {
    root: String,
    paths: Vec<String>,
    targets: Counts,
    max_file: usize,
    max_repository: usize,
}

#[derive(Serialize)]
struct Document {
    repository: String,
    commit: String,
    content_id: String,
    path: String,
    bucket: String,
    text: String,
}

#[derive(Serialize)]
struct Selection {
    documents: BTreeMap<String, BTreeMap<String, Vec<Document>>>,
    characters: Counts,
    #[deser(skip)]
    seen: HashSet<String>,
}

fn split(repository: &str) -> usize {
    let mut hash = Sha256::new();
    hash.update(repository);
    match u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap()) % 20 {
        18 => 1,
        19 => 2,
        _ => 0,
    }
}

fn classify(path: &str, language: &str) -> Option<&'static str> {
    let path = format!("/{}", path.to_lowercase().trim_start_matches('/'));
    let name = path.rsplit('/').next().unwrap_or("");
    if language == "Python" {
        return Some(
            if path.contains("/test/")
                || path.contains("/tests/")
                || name.starts_with("test_")
                || name.ends_with("_test.py")
            {
                "tests"
            } else {
                "implementation"
            },
        );
    }
    if matches!(language, "Markdown" | "reStructuredText")
        || ["readme", "contributing", "architecture"]
            .iter()
            .any(|s| name.starts_with(s))
    {
        return Some("documentation");
    }
    matches!(
        language,
        "Dockerfile" | "INI" | "JSON" | "Makefile" | "Shell" | "TOML" | "YAML"
    )
    .then_some("configuration")
}

fn typed<T: 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| format!("unexpected Arrow type: {:?}", array.data_type()).into())
}

fn field<'a, T: 'static>(array: &'a StructArray, name: &str) -> Result<&'a T> {
    typed(
        array
            .column_by_name(name)
            .ok_or_else(|| format!("missing field {name}"))?
            .as_ref(),
    )
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    typed(
        batch
            .column_by_name(name)
            .ok_or_else(|| format!("missing column {name}"))?
            .as_ref(),
    )
}

fn eligible(batch: &RecordBatch, active: u16) -> Result<BooleanArray> {
    let repos: &StringArray = column(batch, "repo_path")?;
    let commits: &StringArray = column(batch, "commit_id")?;
    let metadata: &StructArray = column(batch, "github_metadata")?;
    let forks: &BooleanArray = field(metadata, "is_fork")?;
    Ok((0..batch.num_rows())
        .map(|row| {
            Some(
                !(metadata.is_valid(row) && forks.is_valid(row) && forks.value(row))
                    && repos.is_valid(row)
                    && commits.is_valid(row)
                    && active & (15 << (4 * split(repos.value(row)))) != 0,
            )
        })
        .collect())
}

fn eligible_files(batch: &RecordBatch, active: u16) -> Result<BooleanArray> {
    let repos: &StringArray = column(batch, "repo_path")?;
    let lists: &ListArray = column(batch, "files")?;
    let files: &StructArray = typed(lists.values().as_ref())?;
    let paths: &StringArray = field(files, "file_path")?;
    let languages: &StringArray = field(files, "language")?;
    let licenses: &StringArray = field(files, "license_type")?;
    let vendors: &BooleanArray = field(files, "is_vendor")?;
    let offsets = lists.value_offsets();
    Ok((0..batch.num_rows())
        .map(|row| {
            let needed = active >> (4 * split(repos.value(row)));
            Some(
                lists.is_valid(row)
                    && (offsets[row] as usize..offsets[row + 1] as usize).any(|i| {
                        files.is_valid(i)
                            && !(vendors.is_valid(i) && vendors.value(i))
                            && licenses.is_valid(i)
                            && licenses.value(i) == "permissive"
                            && paths.is_valid(i)
                            && languages.is_valid(i)
                            && requested_bucket(paths.value(i), languages.value(i), needed)
                    }),
            )
        })
        .collect())
}

fn requested_bucket(path: &str, language: &str, needed: u16) -> bool {
    let Some(bucket) = classify(path, language) else {
        return false;
    };
    let bit = BUCKETS.iter().position(|name| *name == bucket).unwrap();
    needed & (1 << bit) != 0
}

impl Selection {
    fn new(targets: &Counts) -> Self {
        Self {
            documents: targets
                .keys()
                .map(|s| {
                    (
                        s.clone(),
                        BUCKETS
                            .into_iter()
                            .map(|b| (b.into(), Vec::new()))
                            .collect(),
                    )
                })
                .collect(),
            characters: targets
                .keys()
                .map(|s| {
                    (
                        s.clone(),
                        BUCKETS.into_iter().map(|b| (b.into(), 0)).collect(),
                    )
                })
                .collect(),
            seen: HashSet::new(),
        }
    }

    fn active(&self, targets: &Counts) -> u16 {
        let mut mask = 0;
        for (index, split) in SPLITS.iter().enumerate() {
            let Some(buckets) = targets.get(*split) else {
                continue;
            };
            for (bucket, name) in BUCKETS.iter().enumerate() {
                if self.characters[*split][*name] < buckets[*name] {
                    mask |= 1 << (4 * index + bucket);
                }
            }
        }
        mask
    }

    fn accept(&mut self, batch: &RecordBatch, job: &Job) -> Result<()> {
        let repos: &StringArray = column(batch, "repo_path")?;
        let commits: &StringArray = column(batch, "commit_id")?;
        let lists: &ListArray = column(batch, "files")?;
        let files: &StructArray = typed(lists.values().as_ref())?;
        let paths: &StringArray = field(files, "file_path")?;
        let languages: &StringArray = field(files, "language")?;
        let licenses: &StringArray = field(files, "license_type")?;
        let vendors: &BooleanArray = field(files, "is_vendor")?;
        let ids: &StringArray = field(files, "content_id")?;
        let texts: &StringArray = field(files, "content")?;
        for row in 0..batch.num_rows() {
            let repository = repos.value(row);
            let commit = commits.value(row);
            let index = split(repository);
            let split = SPLITS[index];
            if self.active(&job.targets) & (15 << (4 * index)) == 0 || lists.is_null(row) {
                continue;
            }
            let mut used = 0;
            let offsets = lists.value_offsets();
            for index in offsets[row] as usize..offsets[row + 1] as usize {
                if used >= job.max_repository {
                    break;
                }
                if files.is_null(index)
                    || (vendors.is_valid(index) && vendors.value(index))
                    || licenses.is_null(index)
                    || licenses.value(index) != "permissive"
                    || paths.is_null(index)
                    || languages.is_null(index)
                    || ids.is_null(index)
                    || texts.is_null(index)
                {
                    continue;
                }
                let Some(bucket) = classify(paths.value(index), languages.value(index)) else {
                    continue;
                };
                if self.characters[split][bucket] >= job.targets[split][bucket]
                    || self.seen.contains(ids.value(index))
                {
                    continue;
                }
                let text = texts.value(index);
                // Match Python len(str), not UTF-8 byte length.
                let count = text.chars().count();
                if !(64..=job.max_file).contains(&count)
                    || count > job.max_repository - used
                    || text.contains('\0')
                {
                    continue;
                }
                self.seen.insert(ids.value(index).into());
                self.documents
                    .get_mut(split)
                    .unwrap()
                    .get_mut(bucket)
                    .unwrap()
                    .push(Document {
                        repository: repository.into(),
                        commit: commit.into(),
                        content_id: ids.value(index).into(),
                        path: paths.value(index).into(),
                        bucket: bucket.into(),
                        text: text.into(),
                    });
                *self
                    .characters
                    .get_mut(split)
                    .unwrap()
                    .get_mut(bucket)
                    .unwrap() += count;
                used += count;
            }
            if self.active(&job.targets) == 0 {
                break;
            }
        }
        Ok(())
    }

    fn report(&self, job: &Job, started: Instant) {
        let remaining: Vec<_> = job
            .targets
            .iter()
            .flat_map(|(split, buckets)| {
                buckets.iter().filter_map(move |(bucket, target)| {
                    let left = target.saturating_sub(self.characters[split][bucket]);
                    if left > 0 {
                        Some(format!("{split}/{bucket}={left}"))
                    } else {
                        None
                    }
                })
            })
            .collect();
        eprintln!(
            "collect | kept {} documents | {:.1}s | characters remaining: {}",
            self.seen.len(),
            started.elapsed().as_secs_f64(),
            if remaining.is_empty() {
                "none".into()
            } else {
                remaining.join(", ")
            }
        );
    }
}

#[tokio::main(worker_threads = 4)]
async fn main() -> Result<()> {
    let job = ennx_wire::json::from_reader(BufReader::new(io::stdin().lock()))?;
    let selected = collect(job).await?;
    let mut output = BufWriter::new(io::stdout().lock());
    ennx_wire::json::to_writer(&mut output, &selected)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        assert_eq!(classify("Tests/TEST_model.py", "Python"), Some("tests"));
        assert_eq!(classify("/src/model.py", "Python"), Some("implementation"));
        assert_eq!(classify("Readme.custom", "Other"), Some("documentation"));
        assert_eq!(classify("pyproject.toml", "TOML"), Some("configuration"));
        assert_eq!(classify("main.go", "Go"), None);
    }

    #[test]
    fn completion() {
        let targets = BTreeMap::from([(
            "train".into(),
            BUCKETS.into_iter().map(|b| (b.into(), 80)).collect(),
        )]);
        let mut selected = Selection::new(&targets);
        assert_eq!(selected.active(&targets), 15);
        selected.characters = targets.clone();
        assert_eq!(selected.active(&targets), 0);
    }
}
