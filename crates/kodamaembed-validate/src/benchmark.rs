//! Retrieval scores against public benchmarks' relevance judgments.

use crate::{ranking, vectors::Matrix};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize as _;
use std::{fs, path::Path};

/// A benchmark the harness reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Benchmark {
    /// A BEIR dataset as MTEB publishes it: corpus.jsonl, queries.jsonl and
    /// qrels test.tsv in one directory.
    Beir,
    /// LoCoMo's locomo10.json: each conversation's turns, searched by its
    /// questions, judged by their evidence turns.
    Locomo,
}

/// Passages, queries and each query's relevant passage indices.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub name: String,
    pub passages: Vec<String>,
    pub queries: Vec<String>,
    pub relevant: Vec<Vec<usize>>,
    /// Unmatched positive judgments (BEIR) or queries without matched
    /// evidence (LoCoMo).
    pub unjudged: usize,
}

pub fn load(benchmark: Benchmark, path: &Path) -> crate::Result<Vec<Task>> {
    match benchmark {
        Benchmark::Beir => Ok(vec![beir(
            &fs::read_to_string(path.join("corpus.jsonl"))?,
            &fs::read_to_string(path.join("queries.jsonl"))?,
            &fs::read_to_string(path.join("test.tsv"))?,
        )?]),
        Benchmark::Locomo => locomo(&fs::read(path)?),
    }
}

#[derive(serde::Deserialize)]
struct Document {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(serde::Deserialize)]
struct Query {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

fn json_lines<T: serde::de::DeserializeOwned>(text: &str) -> crate::Result<Vec<T>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

/// Title and text join with a space, as MTEB does.
pub fn beir(corpus: &str, queries: &str, judgments: &str) -> crate::Result<Task> {
    let documents: Vec<Document> = json_lines(corpus)?;
    let queries: Vec<Query> = json_lines(queries)?;
    let mut index = FxHashMap::default();
    for (i, document) in documents.iter().enumerate() {
        if index.insert(document.id.as_str(), i).is_some() {
            return Err(format!("duplicate corpus document ID {:?}", document.id).into());
        }
    }
    let mut texts = FxHashMap::default();
    for query in &queries {
        if texts
            .insert(query.id.as_str(), query.text.as_str())
            .is_some()
        {
            return Err(format!("duplicate query ID {:?}", query.id).into());
        }
    }
    let mut order: Vec<&str> = Vec::new();
    let mut relevant: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    let mut unjudged = 0;
    for (number, line) in judgments.lines().enumerate().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        let [query, document, score] = fields[..] else {
            return Err(format!("judgment line {} has {} fields", number + 1, fields.len()).into());
        };
        if score.trim().parse::<i32>()? <= 0 {
            continue;
        }
        let Some(&passage) = index.get(document) else {
            unjudged += 1;
            continue;
        };
        if !texts.contains_key(query) {
            return Err(format!("judged query {query} is not among the queries").into());
        }
        let wanted = relevant.entry(query).or_insert_with(|| {
            order.push(query);
            Vec::new()
        });
        if wanted.contains(&passage) {
            return Err(format!("duplicate positive judgment for {query:?}, {document:?}").into());
        }
        wanted.push(passage);
    }
    Ok(Task {
        name: "beir".into(),
        passages: documents
            .iter()
            .map(|document| {
                if document.title.is_empty() {
                    document.text.clone()
                } else {
                    format!("{} {}", document.title, document.text)
                }
            })
            .collect(),
        queries: order.iter().map(|query| texts[query].to_string()).collect(),
        relevant: order.iter().map(|query| relevant[query].clone()).collect(),
        unjudged,
    })
}

#[derive(serde::Deserialize)]
struct Conversation {
    sample_id: String,
    conversation: serde_json::Map<String, serde_json::Value>,
    qa: Vec<Question>,
}

#[derive(serde::Deserialize)]
struct Turn {
    speaker: String,
    dia_id: String,
    text: String,
    #[serde(default)]
    blip_caption: Option<String>,
}

#[derive(serde::Deserialize)]
struct Question {
    question: String,
    #[serde(default)]
    evidence: Vec<String>,
}

/// Repeated evidence counts once: binary relevance judges passages, not
/// citations. Task IDs become file names, so they must be portable names.
pub fn locomo(bytes: &[u8]) -> crate::Result<Vec<Task>> {
    let conversations: Vec<Conversation> = serde_json::from_slice(bytes)?;
    let mut names = FxHashSet::default();
    for conversation in &conversations {
        let name = &conversation.sample_id;
        let valid = (1..=128).contains(&name.len())
            && name.as_bytes()[0].is_ascii_alphanumeric()
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
            && !name.ends_with('.');
        if !valid {
            return Err(format!("invalid portable LoCoMo task ID {name:?}").into());
        }
        let folded = name.to_ascii_lowercase();
        let stem = folded.split('.').next().expect("nonempty task ID");
        let device = matches!(stem, "con" | "prn" | "aux" | "nul")
            || (stem.len() == 4
                && (stem.starts_with("com") || stem.starts_with("lpt"))
                && (b'1'..=b'9').contains(&stem.as_bytes()[3]));
        if device || !names.insert(folded) {
            return Err(format!("reserved or colliding LoCoMo task ID {name:?}").into());
        }
    }
    conversations
        .into_iter()
        .map(|conversation| {
            let mut sessions: Vec<(u32, &serde_json::Value)> = conversation
                .conversation
                .iter()
                .filter_map(|(key, value)| {
                    let number = key.strip_prefix("session_")?.parse().ok()?;
                    value.is_array().then_some((number, value))
                })
                .collect();
            sessions.sort_by_key(|&(number, _)| number);
            let mut passages = Vec::new();
            let mut index: FxHashMap<String, usize> = FxHashMap::default();
            for (_, turns) in sessions {
                for turn in Vec::<Turn>::deserialize(turns)? {
                    if index.insert(turn.dia_id.clone(), passages.len()).is_some() {
                        return Err(format!(
                            "{}: duplicate turn ID {:?}",
                            conversation.sample_id, turn.dia_id
                        )
                        .into());
                    }
                    passages.push(match turn.blip_caption {
                        Some(caption) => {
                            format!(
                                "{}: {} (shares a photo: {caption})",
                                turn.speaker, turn.text
                            )
                        }
                        None => format!("{}: {}", turn.speaker, turn.text),
                    });
                }
            }
            let (mut queries, mut relevant, mut unjudged) = (Vec::new(), Vec::new(), 0);
            for question in conversation.qa {
                let mut unique = FxHashSet::default();
                let turns: Vec<usize> = question
                    .evidence
                    .iter()
                    .filter_map(|id| index.get(id).copied())
                    .filter(|&turn| unique.insert(turn))
                    .collect();
                if turns.is_empty() {
                    unjudged += 1;
                } else {
                    queries.push(question.question);
                    relevant.push(turns);
                }
            }
            Ok(Task {
                name: conversation.sample_id,
                passages,
                queries,
                relevant,
                unjudged,
            })
        })
        .collect()
}

/// Mean scores over queries, with binary relevance.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Scores {
    pub queries: usize,
    pub ndcg_at_10: f64,
    pub recall_at_10: f64,
    pub hit_at_10: f64,
}

const K: usize = 10;

/// Each relevant passage counts once in recall and ideal DCG.
pub const RETRIEVAL_METRIC_CONVENTION: &str = "binary-distinct-relevance-v1";

// Fidelity reports non-finite values itself; retrieval rejects them.
fn validate_vectors(passages: &Matrix, queries: &Matrix) -> crate::Result<()> {
    if !passages.has_valid_shape() || !queries.has_valid_shape() {
        return Err("invalid retrieval matrix shape".into());
    }
    if queries.rows > 0 && passages.dim != queries.dim {
        return Err("retrieval passage/query dimensions differ".into());
    }
    if passages
        .data
        .iter()
        .chain(&queries.data)
        .any(|value| !value.is_finite())
    {
        return Err("retrieval matrices contain nonfinite values".into());
    }
    Ok(())
}

/// nDCG, recall and hit rate at 10; every query needs a relevant passage.
pub fn score(
    passages: &Matrix,
    queries: &Matrix,
    relevant: &[Vec<usize>],
) -> crate::Result<Scores> {
    validate_vectors(passages, queries)?;
    validate_judgments(passages.rows, queries.rows, relevant)?;
    let mut totals = Scores {
        queries: relevant.len(),
        ..Scores::default()
    };
    for (query, wanted) in relevant.iter().enumerate() {
        let order = ranking::ranked(&ranking::scores(passages, queries.row(query)));
        add_query_score(&mut totals, &order, wanted);
    }
    finish_scores(&mut totals);
    Ok(totals)
}

fn validate_judgments(
    passages: usize,
    queries: usize,
    relevant: &[Vec<usize>],
) -> crate::Result<()> {
    if queries != relevant.len() {
        return Err("retrieval query/judgment counts differ".into());
    }
    for (query, wanted) in relevant.iter().enumerate() {
        if wanted.is_empty() {
            return Err(format!("query {query} has no relevant passage").into());
        }
        let mut unique = FxHashSet::default();
        for &passage in wanted {
            if passage >= passages || !unique.insert(passage) {
                return Err(
                    format!("query {query} has an invalid or duplicate relevant passage").into(),
                );
            }
        }
    }
    Ok(())
}

// A fixed reduction order keeps scores reproducible across callers.
fn add_query_score(totals: &mut Scores, order: &[usize], wanted: &[usize]) {
    let gain = |rank: usize| 1.0 / ((rank + 2) as f64).log2();
    let found: Vec<usize> = (0..K.min(order.len()))
        .filter(|&rank| wanted.contains(&order[rank]))
        .collect();
    let ideal: f64 = (0..wanted.len().min(K)).map(gain).sum();
    totals.ndcg_at_10 += found.iter().map(|&rank| gain(rank)).sum::<f64>() / ideal;
    totals.recall_at_10 += found.len() as f64 / wanted.len() as f64;
    if !found.is_empty() {
        totals.hit_at_10 += 1.0;
    }
}

fn finish_scores(totals: &mut Scores) {
    let n = totals.queries.max(1) as f64;
    totals.ndcg_at_10 /= n;
    totals.recall_at_10 /= n;
    totals.hit_at_10 /= n;
}

/// Query-weighted mean of per-task scores.
pub fn combine(scores: &[Scores]) -> Scores {
    let queries: usize = scores.iter().map(|s| s.queries).sum();
    let weighted = |value: fn(&Scores) -> f64| {
        scores
            .iter()
            .map(|s| value(s) * s.queries as f64)
            .sum::<f64>()
            / queries.max(1) as f64
    };
    Scores {
        queries,
        ndcg_at_10: weighted(|s| s.ndcg_at_10),
        recall_at_10: weighted(|s| s.recall_at_10),
        hit_at_10: weighted(|s| s.hit_at_10),
    }
}

/// How many queries rank the same top 10 passages (as a set) in both runs.
pub fn identical_top_10(
    reference: (&Matrix, &Matrix),
    candidate: (&Matrix, &Matrix),
) -> crate::Result<usize> {
    validate_comparison(reference, candidate)?;
    Ok((0..reference.1.rows)
        .filter(|&query| {
            ranking::compare_top_k(
                &ranking::scores(reference.0, reference.1.row(query)),
                &ranking::scores(candidate.0, candidate.1.row(query)),
                K,
            )
            .identical_set
        })
        .count())
}

fn validate_comparison(
    reference: (&Matrix, &Matrix),
    candidate: (&Matrix, &Matrix),
) -> crate::Result<()> {
    validate_vectors(reference.0, reference.1)?;
    validate_vectors(candidate.0, candidate.1)?;
    if reference.0.rows != candidate.0.rows
        || reference.0.dim != candidate.0.dim
        || reference.1.rows != candidate.1.rows
        || reference.1.dim != candidate.1.dim
    {
        return Err("reference/candidate retrieval shapes differ".into());
    }
    Ok(())
}

/// Overlap is |intersection| / min(10, passages), averaged over queries; an
/// empty comparison cannot pass the gates.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Comparison {
    pub reference: Scores,
    pub candidate: Scores,
    pub identical_top10_sets: usize,
    pub mean_top10_overlap: f64,
    /// Worst absolute deviation from unit L2 norm across both roles/sides.
    pub max_unit_norm_error: f64,
}

pub fn compare(
    reference: (&Matrix, &Matrix),
    candidate: (&Matrix, &Matrix),
    relevant: &[Vec<usize>],
) -> crate::Result<Comparison> {
    validate_comparison(reference, candidate)?;
    validate_judgments(reference.0.rows, reference.1.rows, relevant)?;
    let mut result = Comparison {
        reference: Scores {
            queries: relevant.len(),
            ..Scores::default()
        },
        candidate: Scores {
            queries: relevant.len(),
            ..Scores::default()
        },
        ..Comparison::default()
    };
    let k = K.min(reference.0.rows);
    for (query, wanted) in relevant.iter().enumerate() {
        let r = ranking::ranked(&ranking::scores(reference.0, reference.1.row(query)));
        let c = ranking::ranked(&ranking::scores(candidate.0, candidate.1.row(query)));
        add_query_score(&mut result.reference, &r, wanted);
        add_query_score(&mut result.candidate, &c, wanted);
        let overlap = c[..k].iter().filter(|index| r[..k].contains(index)).count();
        result.identical_top10_sets += usize::from(overlap == k);
        result.mean_top10_overlap += overlap as f64 / k as f64;
    }
    finish_scores(&mut result.reference);
    finish_scores(&mut result.candidate);
    result.mean_top10_overlap /= relevant.len().max(1) as f64;
    for matrix in [reference.0, reference.1, candidate.0, candidate.1] {
        for row in 0..matrix.rows {
            let norm = matrix
                .row(row)
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            result.max_unit_norm_error = result.max_unit_norm_error.max((norm - 1.0).abs());
        }
    }
    Ok(result)
}

/// Query-weighted comparison totals in task order.
pub fn combine_comparisons(comparisons: &[Comparison]) -> Comparison {
    let reference: Vec<_> = comparisons.iter().map(|c| c.reference.clone()).collect();
    let candidate: Vec<_> = comparisons.iter().map(|c| c.candidate.clone()).collect();
    let queries = reference.iter().map(|s| s.queries).sum::<usize>();
    Comparison {
        reference: combine(&reference),
        candidate: combine(&candidate),
        identical_top10_sets: comparisons.iter().map(|c| c.identical_top10_sets).sum(),
        mean_top10_overlap: comparisons
            .iter()
            .map(|c| c.mean_top10_overlap * c.reference.queries as f64)
            .sum::<f64>()
            / queries.max(1) as f64,
        max_unit_norm_error: comparisons
            .iter()
            .map(|c| c.max_unit_norm_error)
            .fold(0.0, f64::max),
    }
}

/// Passages and queries are never averaged together, so neither can hide the other.
#[derive(Debug, Clone, Copy, Default)]
pub struct QualityFidelity {
    pub mean_passage_cosine_distance: f64,
    pub mean_query_cosine_distance: f64,
    pub max_l2: f64,
    pub non_finite: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GateCheck {
    pub value: f64,
    pub relation: &'static str,
    pub threshold: f64,
    pub pass: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct QualityGates {
    pub policy: &'static str,
    pub evaluated: bool,
    pub fidelity_pass: bool,
    pub retrieval_pass: bool,
    pub pass: bool,
    pub mean_passage_cosine_distance: GateCheck,
    pub mean_query_cosine_distance: GateCheck,
    pub max_l2: GateCheck,
    pub max_unit_norm_error: GateCheck,
    pub non_finite: GateCheck,
    pub mean_top10_overlap: GateCheck,
    pub ndcg_loss: GateCheck,
    pub recall_loss: GateCheck,
    pub hit_loss: GateCheck,
}

/// A failed gate is a reported result, not an error.
pub fn quality_gates(comparison: &Comparison, fidelity: QualityFidelity) -> QualityGates {
    let upper = |value: f64, threshold| GateCheck {
        value,
        relation: "<=",
        threshold,
        pass: value.is_finite() && value <= threshold,
    };
    let mean_passage_cosine_distance = upper(fidelity.mean_passage_cosine_distance, 1e-3);
    let mean_query_cosine_distance = upper(fidelity.mean_query_cosine_distance, 1e-3);
    let max_l2 = upper(fidelity.max_l2, 0.1);
    let max_unit_norm_error = upper(comparison.max_unit_norm_error, 1e-5);
    let non_finite = upper(fidelity.non_finite as f64, 0.0);
    let mean_top10_overlap = GateCheck {
        value: comparison.mean_top10_overlap,
        relation: ">=",
        threshold: 0.9,
        pass: comparison.mean_top10_overlap.is_finite() && comparison.mean_top10_overlap >= 0.9,
    };
    let ndcg_loss = upper(
        comparison.reference.ndcg_at_10 - comparison.candidate.ndcg_at_10,
        0.005,
    );
    let recall_loss = upper(
        comparison.reference.recall_at_10 - comparison.candidate.recall_at_10,
        0.01,
    );
    let hit_loss = upper(
        comparison.reference.hit_at_10 - comparison.candidate.hit_at_10,
        0.01,
    );
    let evaluated = comparison.reference.queries > 0
        && comparison.reference.queries == comparison.candidate.queries;
    let fidelity_pass = evaluated
        && mean_passage_cosine_distance.pass
        && mean_query_cosine_distance.pass
        && max_l2.pass
        && max_unit_norm_error.pass
        && non_finite.pass;
    let retrieval_pass =
        evaluated && mean_top10_overlap.pass && ndcg_loss.pass && recall_loss.pass && hit_loss.pass;
    QualityGates {
        policy: "saved-retrieval-quality-v1",
        evaluated,
        fidelity_pass,
        retrieval_pass,
        pass: fidelity_pass && retrieval_pass,
        mean_passage_cosine_distance,
        mean_query_cosine_distance,
        max_l2,
        max_unit_norm_error,
        non_finite,
        mean_top10_overlap,
        ndcg_loss,
        recall_loss,
        hit_loss,
    }
}

#[cfg(test)]
#[path = "benchmark_tests.rs"]
mod tests;
