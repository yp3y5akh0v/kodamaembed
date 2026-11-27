# kodamaembed-validate

Retrieval-quality evaluation for KodamaEmbed: it embeds public benchmarks
with a chosen engine and compares saved vectors against a reference run. It
is a workspace tool (`publish = false`); models and datasets are separate
downloads.

```text
cargo run --release -p kodamaembed-validate --locked -- --help
cargo test -p kodamaembed-validate --locked
```

## Engines

| Engine | Numerical mode |
|---|---|
| `kodamaembed` | FP16 CUDA |
| `kodamaembed-centered` | Centered CUDA preset with CPU-identical vector bits |
| `kodamaembed-vulkan` | Vulkan |
| `kodamaembed-cpu` | FP32 CPU reference |
| `kodamaembed-cpu-centered` | Evaluated centered CPU INT8 preset |

`--family minilm`, `--family e5` or `--family modernbert-embed` selects the
model's text settings; the centered engines run `e5` only.
`--device` picks the CUDA ordinal or Vulkan adapter.
[Integration](../../docs/integration.md) lists the exact artifact and host
requirements.

## Saved-vector quality

`retrieval` embeds a benchmark's passages and queries and saves the vectors.
`compare-retrieval` evaluates two saved runs without loading a model or
repeating inference. Reference and candidate rows must identify the same
items in the same order.

```text
cargo run --release -p kodamaembed-validate --locked -- compare-retrieval --benchmark beir --data <dataset-dir> --reference <fp32-vectors> --candidate <candidate-vectors> --out <comparison.json>
```

Reports include `quality_gates` for each task and the dataset total.
Retrieval totals are query-weighted; fidelity means use each role's vector
count. nDCG, recall and hit use binary relevance: an item judged relevant
counts once, whatever its grade. The `saved-retrieval-quality-v1` policy
requires:

| Metric | Requirement |
|---|---:|
| Passage and query mean cosine distance | <= 0.001 |
| Maximum L2 deviation | <= 0.1 |
| Unit-norm error | <= 0.00001 |
| Finite vectors | Required |
| Mean top-10 overlap | >= 0.9 |
| Maximum nDCG / recall / hit loss | <= 0.005 / 0.01 / 0.01 |

The command succeeds once scoring completes. With `--require-quality` it
writes the report and then returns an error if the aggregate total fails or
cannot be evaluated. Reports list individual task failures alongside the
aggregate. Empty-query comparisons fail.

[Benchmarks](../../docs/benchmarks.md#retrieval-quality) lists the results
and reproduction commands.

[Apache-2.0](LICENSE).
