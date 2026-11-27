use clap::{Parser, Subcommand, ValueEnum};
use kodamaembed_validate::{
    Result,
    benchmark::{self, Benchmark},
    engine::{self, Engine, Family, Kodamaembed},
    fidelity, power, vectors,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(
    name = "kodamaembed-validate",
    about = "Retrieval-quality evaluation for KodamaEmbed"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum EngineArg {
    /// KodamaEmbed on CUDA: FP16 weights, FP32 accumulation.
    Kodamaembed,
    /// Opt-in E5 centered INT8 on CUDA: the CPU preset's bits.
    KodamaembedCentered,
    /// KodamaEmbed on Vulkan (the first discrete GPU unless --device picks another).
    KodamaembedVulkan,
    /// KodamaEmbed on the CPU: FP32, every core.
    KodamaembedCpu,
    /// Opt-in E5 centered INT8 on an AVX-VNNI CPU.
    KodamaembedCpuCentered,
}

fn precision(engine: EngineArg) -> kodamaembed::Precision {
    use kodamaembed::Precision;
    match engine {
        EngineArg::KodamaembedCentered | EngineArg::KodamaembedCpuCentered => Precision::Quantized,
        EngineArg::Kodamaembed | EngineArg::KodamaembedVulkan | EngineArg::KodamaembedCpu => {
            Precision::Float
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Embed a public benchmark's passages and queries and score each ranking
    /// against the benchmark's judgments.
    Retrieval {
        #[arg(long, value_enum)]
        engine: EngineArg,
        #[arg(long, value_enum)]
        benchmark: Benchmark,
        /// BEIR: the directory with corpus.jsonl, queries.jsonl and
        /// test.tsv; LoCoMo: locomo10.json.
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_enum, default_value = "e5")]
        family: Family,
        /// Tokens per KodamaEmbed batch.
        #[arg(long, default_value_t = 16384)]
        batch_tokens: usize,
        /// CUDA ordinal or Vulkan adapter index for KodamaEmbed.
        #[arg(long)]
        device: Option<usize>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Compare a candidate retrieval run with a reference run of the same
    /// benchmark: fidelity, top-10 overlap, scores and fixed quality gates.
    CompareRetrieval {
        #[arg(long, value_enum)]
        benchmark: Benchmark,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Return an error after saving the report if quality gates fail or
        /// cannot be evaluated. By default failed gates are reported as data.
        #[arg(long)]
        require_quality: bool,
    },
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn kodamaembed_device(engine: EngineArg, device: Option<usize>) -> Result<kodamaembed::Device> {
    if device.is_some()
        && !matches!(
            engine,
            EngineArg::Kodamaembed | EngineArg::KodamaembedCentered | EngineArg::KodamaembedVulkan
        )
    {
        return Err("--device applies only to KodamaEmbed on CUDA or Vulkan".into());
    }
    Ok(match engine {
        EngineArg::Kodamaembed | EngineArg::KodamaembedCentered => {
            kodamaembed::Device::Cuda(device.unwrap_or(0))
        }
        EngineArg::KodamaembedVulkan => match device {
            Some(index) => kodamaembed::Device::Vulkan(index),
            None => kodamaembed::Device::first_discrete_vulkan()?,
        },
        EngineArg::KodamaembedCpu | EngineArg::KodamaembedCpuCentered => kodamaembed::Device::Cpu,
    })
}

fn load_engine(
    engine: EngineArg,
    model: &Path,
    family: Family,
    batch_tokens: usize,
    device: Option<usize>,
) -> Result<Box<dyn Engine>> {
    let embedder = kodamaembed::Embedder::builder()
        .device(kodamaembed_device(engine, device)?)
        .precision(precision(engine))
        .batch_tokens(batch_tokens)
        .build_from_dir(model, family.spec())?;
    Ok(Box::new(Kodamaembed(embedder)))
}

struct RetrievalArgs {
    engine: EngineArg,
    benchmark: Benchmark,
    data: PathBuf,
    model: PathBuf,
    family: Family,
    batch_tokens: usize,
    device: Option<usize>,
    out: PathBuf,
}

fn write_retrieval_task(
    engine: &mut dyn Engine,
    task: &benchmark::Task,
    out: &Path,
) -> Result<benchmark::Scores> {
    let (passages, queries) = engine::embed_task(engine, task)?;
    let scores = benchmark::score(&passages, &queries, &task.relevant)?;
    vectors::write(&out.join(format!("{}.passages.kodv", task.name)), &passages)?;
    vectors::write(&out.join(format!("{}.queries.kodv", task.name)), &queries)?;
    Ok(scores)
}

fn retrieval(args: &RetrievalArgs) -> Result<()> {
    let out = args.out.as_path();
    fs::create_dir(out)?;
    let tasks = benchmark::load(args.benchmark, &args.data)?;
    let mut engine = load_engine(
        args.engine,
        &args.model,
        args.family,
        args.batch_tokens,
        args.device,
    )?;
    let started = Instant::now();
    let (mut reports, mut all) = (
        Vec::with_capacity(tasks.len()),
        Vec::with_capacity(tasks.len()),
    );
    for task in &tasks {
        let scores = write_retrieval_task(engine.as_mut(), task, out)?;
        println!(
            "{} {} passages, {} queries: nDCG@10 {:.4}, recall@10 {:.4}",
            task.name,
            task.passages.len(),
            scores.queries,
            scores.ndcg_at_10,
            scores.recall_at_10
        );
        reports.push(serde_json::json!({
            "task": task.name,
            "passages": task.passages.len(),
            "unjudged_queries": task.unjudged,
            "scores": scores,
        }));
        all.push(scores);
    }
    let totals = benchmark::combine(&all);
    let mut run = serde_json::json!({
        "engine": args.engine,
        "benchmark": args.benchmark,
        "metric_convention": benchmark::RETRIEVAL_METRIC_CONVENTION,
        "model": args.model,
        "seconds": started.elapsed().as_secs_f64(),
        "totals": totals,
        "tasks": reports,
    });
    if let Some(report) = engine.report() {
        run["engine_report"] = report;
    }
    write_json(&out.join("scores.json"), &run)?;
    println!("{}", serde_json::to_string_pretty(&totals)?);
    Ok(())
}

fn compare_retrieval(
    benchmark: Benchmark,
    data: &Path,
    reference: &Path,
    candidate: &Path,
    out: &Path,
    require_quality: bool,
) -> Result<()> {
    let tasks = benchmark::load(benchmark, data)?;
    let mut reports = Vec::with_capacity(tasks.len());
    let mut comparisons = Vec::with_capacity(tasks.len());
    let (mut min_cosine, mut max_l2, mut non_finite) = (f64::INFINITY, 0.0f64, 0);
    let (mut passage_rows, mut query_rows) = (0usize, 0usize);
    let (mut passage_cosine_distance, mut query_cosine_distance) = (0.0, 0.0);
    for task in &tasks {
        let load =
            |dir: &Path, kind: &str| vectors::read(&dir.join(format!("{}.{kind}.kodv", task.name)));
        let (rp, rq) = (load(reference, "passages")?, load(reference, "queries")?);
        let (cp, cq) = (load(candidate, "passages")?, load(candidate, "queries")?);
        if rp.rows != task.passages.len() || rq.rows != task.queries.len() {
            return Err(format!(
                "{}: {} reference passages and {} queries for {} and {}",
                task.name,
                rp.rows,
                rq.rows,
                task.passages.len(),
                task.queries.len()
            )
            .into());
        }
        let comparison = benchmark::compare((&rp, &rq), (&cp, &cq), &task.relevant)?;
        let passages = fidelity::compare(&rp, &cp)?;
        let queries = fidelity::compare(&rq, &cq)?;
        let quality = benchmark::quality_gates(
            &comparison,
            benchmark::QualityFidelity {
                mean_passage_cosine_distance: 1.0 - passages.mean_cosine,
                mean_query_cosine_distance: 1.0 - queries.mean_cosine,
                max_l2: passages.max_l2.max(queries.max_l2),
                non_finite: passages.non_finite + queries.non_finite,
            },
        );
        passage_rows += passages.rows;
        query_rows += queries.rows;
        if passages.rows > 0 {
            passage_cosine_distance += (1.0 - passages.mean_cosine) * passages.rows as f64;
        }
        if queries.rows > 0 {
            query_cosine_distance += (1.0 - queries.mean_cosine) * queries.rows as f64;
        }
        min_cosine = min_cosine.min(passages.min_cosine).min(queries.min_cosine);
        max_l2 = max_l2.max(passages.max_l2).max(queries.max_l2);
        non_finite += passages.non_finite + queries.non_finite;
        reports.push(serde_json::json!({
            "task": task.name,
            "identical_top10_sets": comparison.identical_top10_sets,
            "mean_top10_overlap": comparison.mean_top10_overlap,
            "max_unit_norm_error": comparison.max_unit_norm_error,
            "queries": task.queries.len(),
            "passages": passages,
            "query_vectors": queries,
            "reference": comparison.reference,
            "candidate": comparison.candidate,
            "quality_gates": quality,
        }));
        comparisons.push(comparison);
    }
    let comparison = benchmark::combine_comparisons(&comparisons);
    let mean_passage_cosine_distance = if passage_rows > 0 {
        passage_cosine_distance / passage_rows as f64
    } else {
        f64::NAN
    };
    let mean_query_cosine_distance = if query_rows > 0 {
        query_cosine_distance / query_rows as f64
    } else {
        f64::NAN
    };
    let quality = benchmark::quality_gates(
        &comparison,
        benchmark::QualityFidelity {
            mean_passage_cosine_distance,
            mean_query_cosine_distance,
            max_l2,
            non_finite,
        },
    );
    let totals = serde_json::json!({
        "queries": tasks.iter().map(|task| task.queries.len()).sum::<usize>(),
        "identical_top10_sets": comparison.identical_top10_sets,
        "mean_top10_overlap": comparison.mean_top10_overlap,
        "mean_passage_cosine_distance": mean_passage_cosine_distance,
        "mean_query_cosine_distance": mean_query_cosine_distance,
        "max_unit_norm_error": comparison.max_unit_norm_error,
        "min_cosine": min_cosine,
        "max_l2": max_l2,
        "non_finite": non_finite,
        "reference": comparison.reference,
        "candidate": comparison.candidate,
        "quality_gates": quality,
    });
    write_json(
        out,
        &serde_json::json!({
            "metric_convention": benchmark::RETRIEVAL_METRIC_CONVENTION,
            "totals": totals,
            "tasks": reports,
        }),
    )?;
    println!("{}", serde_json::to_string_pretty(&totals)?);
    if require_quality && !quality.pass {
        return Err(format!(
            "retrieval quality gates failed or were unevaluated; report saved to {}",
            out.display()
        )
        .into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let command = Cli::parse().command;
    power::run_unthrottled()?;
    match command {
        Command::Retrieval {
            engine,
            benchmark,
            data,
            model,
            family,
            batch_tokens,
            device,
            out,
        } => retrieval(&RetrievalArgs {
            engine,
            benchmark,
            data,
            model,
            family,
            batch_tokens,
            device,
            out,
        }),
        Command::CompareRetrieval {
            benchmark,
            data,
            reference,
            candidate,
            out,
            require_quality,
        } => compare_retrieval(
            benchmark,
            &data,
            &reference,
            &candidate,
            &out,
            require_quality,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedRetrievalEngine {
        passages: Vec<Vec<f32>>,
        queries: Vec<Vec<f32>>,
    }

    impl Engine for FixedRetrievalEngine {
        fn embed_passages(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(self.passages.clone())
        }

        fn embed_queries(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(self.queries.clone())
        }

        fn report(&self) -> Option<serde_json::Value> {
            None
        }
    }

    struct RetrievalDirectory(PathBuf);

    impl Drop for RetrievalDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn saved_retrieval_comparison_emits_overlap_and_separate_quality_axes() {
        let args = [
            "kodamaembed-validate",
            "compare-retrieval",
            "--benchmark",
            "beir",
            "--data",
            "data",
            "--reference",
            "reference",
            "--candidate",
            "candidate",
            "--out",
            "report.json",
        ];
        assert!(matches!(
            Cli::try_parse_from(args).unwrap().command,
            Command::CompareRetrieval {
                require_quality: false,
                ..
            }
        ));
        assert!(matches!(
            Cli::try_parse_from(args.into_iter().chain(["--require-quality"]))
                .unwrap()
                .command,
            Command::CompareRetrieval {
                require_quality: true,
                ..
            }
        ));
        let path = std::env::temp_dir().join(format!(
            "kodamaembed-saved-comparison-{}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        let directory = RetrievalDirectory(path);
        let data = directory.0.join("data");
        let reference = directory.0.join("reference");
        let candidate = directory.0.join("candidate");
        for path in [&data, &reference, &candidate] {
            fs::create_dir(path).unwrap();
        }
        fs::write(
            data.join("corpus.jsonl"),
            "{\"_id\":\"d1\",\"text\":\"one\"}\n{\"_id\":\"d2\",\"text\":\"two\"}\n",
        )
        .unwrap();
        fs::write(
            data.join("queries.jsonl"),
            "{\"_id\":\"q1\",\"text\":\"which\"}\n",
        )
        .unwrap();
        fs::write(
            data.join("test.tsv"),
            "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        )
        .unwrap();
        let passages = vectors::Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let queries = vectors::Matrix::from_rows(&[vec![1.0, 0.0]]).unwrap();
        for dir in [&reference, &candidate] {
            vectors::write(&dir.join("beir.passages.kodv"), &passages).unwrap();
            vectors::write(&dir.join("beir.queries.kodv"), &queries).unwrap();
        }
        let out = directory.0.join("comparison.json");
        compare_retrieval(Benchmark::Beir, &data, &reference, &candidate, &out, true).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(report["metric_convention"], "binary-distinct-relevance-v1");
        assert_eq!(report["totals"]["mean_top10_overlap"], 1.0);
        assert_eq!(report["totals"]["reference"]["ndcg_at_10"], 1.0);
        assert_eq!(report["totals"]["quality_gates"]["pass"], true);
        assert_eq!(report["tasks"][0]["quality_gates"]["pass"], true);
        // Scaled vectors keep the rankings and fail the fidelity gate.
        let scaled = vectors::Matrix::from_rows(&[vec![2.0, 0.0], vec![0.0, 2.0]]).unwrap();
        vectors::write(&candidate.join("beir.passages.kodv"), &scaled).unwrap();
        compare_retrieval(Benchmark::Beir, &data, &reference, &candidate, &out, false).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(report["totals"]["quality_gates"]["retrieval_pass"], true);
        assert_eq!(report["totals"]["quality_gates"]["fidelity_pass"], false);
        assert_eq!(report["totals"]["quality_gates"]["pass"], false);
        assert_eq!(report["totals"]["max_unit_norm_error"], 1.0);
        fs::remove_file(&out).unwrap();
        assert!(
            compare_retrieval(Benchmark::Beir, &data, &reference, &candidate, &out, true).is_err()
        );
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(report["totals"]["quality_gates"]["pass"], false);
        fs::remove_file(&out).unwrap();
        let invalid = vectors::Matrix::from_rows(&[vec![f32::NAN, 0.0], vec![0.0, 1.0]]).unwrap();
        vectors::write(&candidate.join("beir.passages.kodv"), &invalid).unwrap();
        assert!(
            compare_retrieval(Benchmark::Beir, &data, &reference, &candidate, &out, true).is_err()
        );
        assert!(!out.exists());
        // A zero-query task keeps zero metrics and fails the gates.
        vectors::write(&candidate.join("beir.passages.kodv"), &passages).unwrap();
        let empty = vectors::Matrix::from_rows(&[]).unwrap();
        for dir in [&reference, &candidate] {
            vectors::write(&dir.join("beir.queries.kodv"), &empty).unwrap();
        }
        fs::write(data.join("test.tsv"), "query-id\tcorpus-id\tscore\n").unwrap();
        assert!(
            compare_retrieval(Benchmark::Beir, &data, &reference, &candidate, &out, true).is_err()
        );
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(report["totals"]["quality_gates"]["evaluated"], false);
        assert_eq!(report["totals"]["quality_gates"]["pass"], false);
    }

    #[test]
    fn retrieval_rejects_invalid_backend_results_before_writing_and_keeps_valid_order() {
        let path = std::env::temp_dir().join(format!(
            "kodamaembed-retrieval-write-check-{}",
            std::process::id(),
        ));
        fs::create_dir(&path).unwrap();
        let out = RetrievalDirectory(path);
        let task = benchmark::Task {
            name: "task-1".into(),
            passages: vec!["first".into(), "second".into()],
            queries: vec!["question".into()],
            relevant: vec![vec![1]],
            unjudged: 0,
        };
        let good_passages = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let good_queries = vec![vec![1.0, 0.0]];
        for mut engine in [
            FixedRetrievalEngine {
                passages: vec![vec![1.0, 0.0]],
                queries: good_queries.clone(),
            },
            FixedRetrievalEngine {
                passages: good_passages.clone(),
                queries: vec![vec![1.0, 0.0]; 2],
            },
            FixedRetrievalEngine {
                passages: vec![vec![f32::NAN, 0.0], vec![1.0, 0.0]],
                queries: good_queries.clone(),
            },
        ] {
            assert!(write_retrieval_task(&mut engine, &task, &out.0).is_err());
            assert_eq!(fs::read_dir(&out.0).unwrap().count(), 0);
        }
        let mut engine = FixedRetrievalEngine {
            passages: good_passages,
            queries: good_queries,
        };
        let scores = write_retrieval_task(&mut engine, &task, &out.0).unwrap();
        assert_eq!(scores.ndcg_at_10, 1.0);
        let passages = vectors::read(&out.0.join("task-1.passages.kodv")).unwrap();
        assert_eq!(passages.row(0), &[0.0, 1.0]);
        assert_eq!(passages.row(1), &[1.0, 0.0]);
        assert_eq!(
            vectors::read(&out.0.join("task-1.queries.kodv"))
                .unwrap()
                .rows,
            1
        );
    }

    #[test]
    fn cpu_centered_cli_choice_and_receipt_name() {
        let cli = Cli::try_parse_from([
            "kodamaembed-validate",
            "retrieval",
            "--engine",
            "kodamaembed-cpu-centered",
            "--benchmark",
            "beir",
            "--data",
            "unused-data",
            "--model",
            "unused-model",
            "--family",
            "e5",
            "--out",
            "unused-output",
        ])
        .unwrap();
        let Command::Retrieval {
            engine,
            family,
            batch_tokens,
            device,
            ..
        } = cli.command
        else {
            panic!("expected retrieval command");
        };
        assert!(matches!(engine, EngineArg::KodamaembedCpuCentered));
        assert_eq!(family, Family::E5);
        assert_eq!(batch_tokens, 16384);
        assert!(device.is_none());
        assert_eq!(
            serde_json::to_value(engine).unwrap(),
            serde_json::json!("kodamaembed-cpu-centered")
        );
    }

    #[test]
    fn cpu_engines_run_on_the_cpu_and_reject_device_indices() {
        for engine in [EngineArg::KodamaembedCpu, EngineArg::KodamaembedCpuCentered] {
            assert!(matches!(
                kodamaembed_device(engine, None).unwrap(),
                kodamaembed::Device::Cpu
            ));
            assert!(
                kodamaembed_device(engine, Some(0))
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("--device applies only")
            );
        }
    }

    #[test]
    fn each_engine_selects_its_precision() {
        use kodamaembed::Precision;
        for (engine, expected) in [
            (EngineArg::KodamaembedCpuCentered, Precision::Quantized),
            (EngineArg::KodamaembedCentered, Precision::Quantized),
            (EngineArg::Kodamaembed, Precision::Float),
            (EngineArg::KodamaembedVulkan, Precision::Float),
            (EngineArg::KodamaembedCpu, Precision::Float),
        ] {
            assert_eq!(precision(engine), expected, "{engine:?}");
        }
    }

    #[test]
    fn the_cuda_centered_engine_runs_on_a_cuda_ordinal() {
        let engine = EngineArg::KodamaembedCentered;
        assert!(matches!(
            kodamaembed_device(engine, None).unwrap(),
            kodamaembed::Device::Cuda(0)
        ));
        assert!(matches!(
            kodamaembed_device(engine, Some(1)).unwrap(),
            kodamaembed::Device::Cuda(1)
        ));
    }
}
