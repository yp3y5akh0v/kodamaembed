//! Tokenization-only comparison on JSONL rows with `text`, optional `title` and `query`.
//! Usage: bench_tokenize TOKENIZER_JSON DATA_JSONL MAX_LENGTH THREADS REPEATS [PREFIX]

#[path = "../src/bert_wordpiece.rs"]
mod bert_wordpiece;
#[path = "../src/tokenizing.rs"]
mod tokenizing;

use sha2::{Digest, Sha256};
use std::{error::Error, fs, hint::black_box, time::Instant};
use tokenizers::{EncodeInput, Encoding, Tokenizer, TruncationParams};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, PartialEq, Eq)]
struct OwnedInput {
    ids: Vec<u32>,
    types: Vec<u32>,
}

impl From<Encoding> for OwnedInput {
    fn from(encoding: Encoding) -> Self {
        Self {
            ids: encoding.get_ids().to_vec(),
            types: encoding.get_type_ids().to_vec(),
        }
    }
}

impl From<tokenizing::InferenceInput> for OwnedInput {
    fn from(input: tokenizing::InferenceInput) -> Self {
        match input {
            tokenizing::InferenceInput::Original(encoding) => encoding.into(),
            tokenizing::InferenceInput::Compact { ids, types } => Self { ids, types },
        }
    }
}

fn inputs(chunk: &[(Option<String>, String)]) -> Vec<EncodeInput<'_>> {
    chunk
        .iter()
        .map(|(query, text)| match query {
            Some(query) => (query.as_str(), text.as_str()).into(),
            None => text.as_str().into(),
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=6).contains(&args.len()) {
        return Err(
            "usage: bench_tokenize TOKENIZER_JSON DATA_JSONL MAX_LENGTH THREADS REPEATS [PREFIX]"
                .into(),
        );
    }
    let max_length = args[2].parse::<usize>()?;
    let threads = args[3].parse::<usize>()?;
    let repeats = args[4].parse::<usize>()?;
    if max_length == 0 || threads == 0 || repeats == 0 {
        return Err("length, threads and repeats must be positive".into());
    }
    let tokenizer_bytes = fs::read(&args[0])?;
    let data = fs::read(&args[1])?;
    let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
    tokenizing::sentencepiece_pipeline(&mut tokenizer);
    tokenizer.with_padding(None);
    tokenizer.with_truncation(Some(TruncationParams {
        max_length,
        ..Default::default()
    }))?;
    let prefix = args.get(5).map_or("", String::as_str);
    let rows = std::str::from_utf8(&data)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| -> Result<(Option<String>, String)> {
            let row: serde_json::Value = serde_json::from_str(line)?;
            let text = row["text"].as_str().ok_or("row requires text")?;
            let title = row["title"].as_str().unwrap_or("");
            let text = if title.is_empty() {
                format!("{prefix}{text}")
            } else {
                format!("{prefix}{title} {text}")
            };
            Ok((row["query"].as_str().map(str::to_owned), text))
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Err("no input rows".into());
    }
    let production = tokenizing::InferenceTokenizer::new(tokenizer.clone());
    let pool = kodamaembed_cpu::worker_pool("tokenizer-benchmark", threads)?;
    let encode = |chunk: &[(Option<String>, String)],
                  method: usize|
     -> tokenizers::Result<Vec<OwnedInput>> {
        pool.install(|| match method {
            0 => Ok(tokenizer
                .encode_batch(inputs(chunk), true)?
                .into_iter()
                .map(OwnedInput::from)
                .collect::<Vec<_>>()),
            // Preserve the ordinary pair path when a postprocessor is absent.
            1 => Ok(tokenizing::encode_batch(&tokenizer, inputs(chunk))?
                .into_iter()
                .map(OwnedInput::from)
                .collect()),
            2 => Ok(production
                .encode_batch(inputs(chunk))?
                .into_iter()
                .map(OwnedInput::from)
                .collect()),
            _ => unreachable!(),
        })
    };
    let mut tokens = 0;
    let mut compact_rows = 0;
    let mut expected = Vec::with_capacity(rows.len());
    let mut input_digest = Sha256::new();
    // Compare every final input and reference mask outside the timed runs.
    for chunk in rows.chunks(64) {
        let ordinary = pool.install(|| tokenizer.encode_batch(inputs(chunk), true))?;
        let fast = pool.install(|| tokenizing::encode_batch(&tokenizer, inputs(chunk)))?;
        let actual = production.encode_batch_in(&pool, inputs(chunk))?;
        assert_eq!(ordinary.len(), fast.len());
        assert_eq!(ordinary.len(), actual.len());
        for ((ordinary, fast), actual) in ordinary.iter().zip(&fast).zip(&actual) {
            assert_eq!(ordinary.get_ids(), fast.get_ids());
            assert_eq!(ordinary.get_type_ids(), fast.get_type_ids());
            assert_eq!(ordinary.get_attention_mask(), fast.get_attention_mask());
            assert_eq!(ordinary.get_ids(), actual.get_ids());
            assert_eq!(ordinary.get_type_ids(), actual.get_type_ids());
            match actual {
                tokenizing::InferenceInput::Original(encoding) => {
                    assert_eq!(ordinary.get_attention_mask(), encoding.get_attention_mask());
                }
                tokenizing::InferenceInput::Compact { .. } => {
                    // The guarded compact path has no padding and implicit ones.
                    assert!(ordinary.get_attention_mask().iter().all(|&v| v == 1));
                    compact_rows += 1;
                }
            }
            tokens += ordinary.len();
            input_digest.update((ordinary.len() as u64).to_le_bytes());
            for id in ordinary.get_ids().iter().chain(ordinary.get_type_ids()) {
                input_digest.update(id.to_le_bytes());
            }
        }
        expected.extend(ordinary.into_iter().map(OwnedInput::from));
    }
    let methods = ["offsets", "no_offsets", "production"];
    let mut runs = Vec::new();
    let mut expected_checksum = None;
    // Rotating the order balances each method's position every three rounds.
    for round in 0..repeats {
        for offset in 0..3 {
            let method = (round + offset) % 3;
            let mut retained = Vec::with_capacity(rows.len());
            let start = Instant::now();
            let mut checksum = 0_u64;
            for chunk in rows.chunks(64) {
                for input in encode(chunk, method)? {
                    for &id in input.ids.iter().chain(&input.types) {
                        checksum = checksum.wrapping_mul(31).wrapping_add(u64::from(id));
                    }
                    retained.push(input);
                }
            }
            black_box(&retained);
            black_box(checksum);
            let seconds = start.elapsed().as_secs_f64();
            assert_eq!(retained, expected, "{} round {round}", methods[method]);
            match expected_checksum {
                Some(expected) => assert_eq!(checksum, expected),
                None => expected_checksum = Some(checksum),
            }
            runs.push(serde_json::json!({
                "round":round, "method":methods[method],
                "seconds":seconds, "checksum":checksum,
            }));
        }
    }
    let median = |method: &str| {
        let mut values: Vec<_> = runs
            .iter()
            .filter(|r| r["method"] == method)
            .map(|r| r["seconds"].as_f64().unwrap())
            .collect();
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope":"Warmed tokenization and owned final IDs/types with common checksum and retention inside timer. File I/O, tokenizer loading, parity checks, inference and external output storage excluded. All legs retained; complete arrays checked after timing.",
            "consumer":"Original encodings copy IDs/types to owned arrays; production compact arrays move without copying. Compact masks are implicit ones under the guard; original masks are compared directly outside timing.",
            "tokenizers_version":"0.22.2", "tokenizer_sha256":format!("{:x}",Sha256::digest(&tokenizer_bytes)),
            "data_sha256":format!("{:x}",Sha256::digest(&data)), "rows":rows.len(),"tokens":tokens,
            "max_length":max_length,"threads":threads,"chunk":64,"repeats":repeats,"prefix":prefix,
            "production_compact_rows":compact_rows,"input_ids_types_sha256":format!("{:x}",input_digest.finalize()),
            "source_sha256":format!("{:x}",Sha256::digest(include_bytes!("bench_tokenize.rs"))),
            "bert_wordpiece_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/bert_wordpiece.rs"))),
            "tokenizing_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/tokenizing.rs"))),
            "all_ids_types_masks_identical":true,"offsets_median_seconds":median("offsets"),
            "no_offsets_median_seconds":median("no_offsets"),
            "production_median_seconds":median("production"),
            "speedup":median("offsets")/median("no_offsets"),"runs":runs,
            "production_speedup_vs_offsets":median("offsets")/median("production"),
            "production_speedup_vs_no_offsets":median("no_offsets")/median("production"),
        }))?
    );
    Ok(())
}
