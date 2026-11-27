# Benchmarks and validation

The measurements ran on Linux 6.8 with an Intel Core i5-13500 and an NVIDIA
RTX 4000 SFF Ada Generation (driver 595.112.07) under its 70 W power limit.
`docs/benchmarks/release.json` and `docs/benchmarks/changing-lengths.json`,
the receipts, hold every sample behind the tables.

- [Embedding latency](#embedding-latency)
- [Correctness](#correctness)
- [Retrieval quality](#retrieval-quality)
- [Limits](#limits)
- [Reproducing the measurements](#reproducing-the-measurements)

## Embedding latency

Each call takes one query text and returns its normalized FP32 embedding on
the host; the time covers tokenization, transfer, inference, pooling and
normalization. Every engine receives the same text, prefix, truncation,
token IDs and, where the model takes them, segment IDs. Lengths include
special tokens and the query prefix, `query: ` for E5 and `search_query: `
for modernbert-embed; at the models' maximum lengths the texts are
truncated. Values are milliseconds per call, each the median of three
process medians. The last two columns divide the latency of the fastest
engine other than KodamaEmbed by KodamaEmbed's, using unrounded values: above
1.000x, KodamaEmbed is faster. KodamaEmbed runs E5-large-v1 in quantized
precision, centered INT8, and in float precision, and all-MiniLM-L6-v2 and
modernbert-embed-base in float precision.

On the GPU the other engines run FP16 graphs. In every row KodamaEmbed is
the fastest engine and TensorRT the second fastest. KodamaEmbed's float
precision is behind TensorRT in one case, E5-large at 512 tokens, by 0.6%;
its quantized precision is ahead there:

| Model | Tokens | KodamaEmbed quantized | KodamaEmbed float | ONNX Runtime CUDA | TensorRT | PyTorch SDPA | Fastest other / quantized | Fastest other / float |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E5-large-v1 | 16 | 1.3254 | 2.7084 | 2.9098 | 2.8463 | 3.1888 | 2.148x | 1.051x |
| E5-large-v1 | 33 | 1.5944 | 2.8568 | 3.0346 | 2.9316 | 3.3946 | 1.839x | 1.026x |
| E5-large-v1 | 64 | 1.8143 | 2.9723 | 3.1695 | 3.0883 | 3.6406 | 1.702x | 1.039x |
| E5-large-v1 | 65 | 2.0438 | 3.1569 | 3.3232 | 3.1653 | 3.7468 | 1.549x | 1.003x |
| E5-large-v1 | 128 | 2.5951 | 3.4091 | 3.8505 | 3.4755 | 4.0225 | 1.339x | 1.020x |
| E5-large-v1 | 256 | 4.6435 | 4.9202 | 5.1731 | 4.9675 | 5.5993 | 1.070x | 1.010x |
| E5-large-v1 | 512 | 7.7767 | 8.4626 | 9.6348 | 8.4153 | 9.8967 | 1.082x | 0.994x |
| all-MiniLM-L6-v2 | 16 | | 0.1087 | 0.2458 | 0.2185 | 0.3398 | | 2.010x |
| all-MiniLM-L6-v2 | 33 | | 0.1346 | 0.3014 | 0.2756 | 0.3833 | | 2.047x |
| all-MiniLM-L6-v2 | 64 | | 0.1606 | 0.3469 | 0.3187 | 0.4402 | | 1.984x |
| all-MiniLM-L6-v2 | 65 | | 0.1937 | 0.3561 | 0.3367 | 0.4457 | | 1.738x |
| all-MiniLM-L6-v2 | 128 | | 0.2221 | 0.5061 | 0.4827 | 0.6008 | | 2.174x |
| all-MiniLM-L6-v2 | 256 | | 0.3599 | 0.9156 | 0.8788 | 1.0193 | | 2.442x |
| modernbert-embed-base | 16 | | 1.1864 | 1.9160 | 1.3428 | 2.0362 | | 1.132x |
| modernbert-embed-base | 33 | | 1.3505 | 2.0356 | 1.4351 | 2.1626 | | 1.063x |
| modernbert-embed-base | 64 | | 1.4412 | 2.2030 | 1.5208 | 2.2736 | | 1.055x |
| modernbert-embed-base | 65 | | 1.5347 | 2.2437 | 1.5970 | 2.4264 | | 1.041x |
| modernbert-embed-base | 128 | | 1.6869 | 2.5310 | 1.8315 | 2.7486 | | 1.086x |
| modernbert-embed-base | 256 | | 2.5537 | 3.6519 | 2.5660 | 3.5454 | | 1.005x |
| modernbert-embed-base | 512 | | 4.1299 | 8.2806 | 4.3765 | 6.3367 | | 1.060x |
| modernbert-embed-base | 1,024 | | 8.2801 | 18.9388 | 8.4507 | 11.3773 | | 1.021x |
| modernbert-embed-base | 2,048 | | 17.3796 | 77.6550 | 19.9825 | 28.1677 | | 1.150x |
| modernbert-embed-base | 4,096 | | 41.5683 | 292.9960 | 59.3790 | 82.3007 | | 1.428x |
| modernbert-embed-base | 8,192 | | 110.8162 | 1079.7138 | 194.1871 | 293.4731 | | 1.752x |

Quantized precision's slowest run is faster than every other engine's
fastest run in all 7 cases, and float precision's in 22 of its 24. On
E5-large at 256 tokens, float precision's run medians of 4.88 to 5.00 ms
overlap TensorRT's 4.92 to 4.98 ms; at 512 tokens TensorRT's 8.41 to
8.42 ms lie below float precision's 8.43 to 8.47 ms.

On the processor every engine runs six threads on six efficiency cores, the
other engines in FP32. In every row KodamaEmbed is the fastest engine, in
both precisions. The second fastest is ONNX Runtime, or PyTorch for
modernbert-embed-base from 512 tokens:

| Model | Tokens | KodamaEmbed quantized | KodamaEmbed float | ONNX Runtime | PyTorch | Fastest other / quantized | Fastest other / float |
|---|---:|---:|---:|---:|---:|---:|---:|
| E5-large-v1 | 16 | 23.7055 | 47.6572 | 56.3975 | 87.2169 | 2.379x | 1.183x |
| E5-large-v1 | 33 | 45.9287 | 84.9343 | 97.4087 | 125.8677 | 2.121x | 1.147x |
| E5-large-v1 | 64 | 85.4281 | 155.6136 | 173.1461 | 293.0763 | 2.027x | 1.113x |
| E5-large-v1 | 65 | 87.0625 | 159.3438 | 176.5161 | 294.2110 | 2.027x | 1.108x |
| E5-large-v1 | 128 | 183.7337 | 308.8859 | 337.5869 | 446.3820 | 1.837x | 1.093x |
| E5-large-v1 | 256 | 376.8930 | 633.6124 | 678.4614 | 783.6900 | 1.800x | 1.071x |
| E5-large-v1 | 512 | 771.8245 | 1251.8569 | 1466.2245 | 1474.6151 | 1.900x | 1.171x |
| all-MiniLM-L6-v2 | 16 | | 1.8915 | 2.9156 | 4.9680 | | 1.541x |
| all-MiniLM-L6-v2 | 33 | | 3.3997 | 4.8148 | 6.6756 | | 1.416x |
| all-MiniLM-L6-v2 | 64 | | 6.1035 | 7.9966 | 10.4646 | | 1.310x |
| all-MiniLM-L6-v2 | 65 | | 6.2212 | 8.1546 | 10.4864 | | 1.311x |
| all-MiniLM-L6-v2 | 128 | | 12.0078 | 15.2591 | 17.6519 | | 1.271x |
| all-MiniLM-L6-v2 | 256 | | 24.8671 | 30.4595 | 33.6795 | | 1.225x |
| modernbert-embed-base | 16 | | 18.4693 | 23.3642 | 35.8909 | | 1.265x |
| modernbert-embed-base | 33 | | 32.5234 | 40.2784 | 52.1068 | | 1.238x |
| modernbert-embed-base | 64 | | 58.6426 | 70.4821 | 90.3641 | | 1.202x |
| modernbert-embed-base | 65 | | 60.4441 | 72.0025 | 89.0518 | | 1.191x |
| modernbert-embed-base | 128 | | 115.9100 | 133.3067 | 145.6864 | | 1.150x |
| modernbert-embed-base | 256 | | 231.7786 | 268.4905 | 279.6120 | | 1.158x |
| modernbert-embed-base | 512 | | 468.3507 | 589.9099 | 585.7621 | | 1.251x |
| modernbert-embed-base | 1,024 | | 979.3402 | 1413.4170 | 1384.7768 | | 1.414x |
| modernbert-embed-base | 2,048 | | 2194.8955 | 3723.6383 | 3511.3765 | | 1.600x |

There, KodamaEmbed's slowest run is faster than every other engine's fastest
run in all 7 quantized and all 22 float cases.
[Retrieval quality](#retrieval-quality) compares quantized precision with
FP32.

| Engine | Configuration |
|---|---|
| KodamaEmbed quantized | Public API with `Precision::Quantized` on `Device::Cuda(0)` or `Device::Cpu` (`bench_query_api --mode cuda-centered` or `cpu-centered`) |
| KodamaEmbed float | Public API with `Precision::Float` on `Device::Cuda(0)` or `Device::Cpu` (`bench_query_api --mode cuda-fp16` or `cpu-fp32`) |
| ONNX Runtime CUDA | ONNX Runtime GPU 1.30.0, FP16 graph from `export_bert.py`, TF32 disabled, CUDA graph replay |
| TensorRT | TensorRT 10.16.1.11 through ONNX Runtime's TensorRT provider, FP16, builder optimization level 5, one fixed shape per case, CUDA graph replay |
| PyTorch SDPA | PyTorch 2.14.1+cu130 and Transformers 4.57.6, the model in FP16 with scaled-dot-product attention, CUDA graphs, FP32 masked mean pooling |
| ONNX Runtime | ONNX Runtime GPU 1.30.0's CPU provider, FP32 graph from `export_bert.py` |
| PyTorch | PyTorch 2.14.1+cu130 and Transformers 4.57.6 on the processor, the model in FP32 with scaled-dot-product attention, FP32 masked mean pooling |

modernbert-embed-base adds settings of its own. ONNX Runtime CUDA's FP16
graph keeps the rotary positions and angles in FP32, because FP16 holds every
whole number only up to 2,048. PyTorch makes the model's attention masks once
per case, before its CUDA graph is captured or its timed calls begin, and with
one unpadded text per call passes no mask to the layers that attend to the
whole text; the model's own forward makes the masks at every call. On the GPU,
Transformers compiles the model's embeddings and feed-forward blocks, its
default where Triton is installed; on the processor it does not.

### Protocol

The runner executes one benchmark process at a time in three rounds, which
together make one rotation. Each round runs every case and engine, in the
reverse of the previous round's order. A process loads its model, settles
for 30 seconds, warms up for at least 2 seconds and 20 calls, and records 51
samples of three calls. All three returned outputs stay live until the
sample clock stops. Loading, export, compilation, calibration and graph
capture are outside timing.

Processes ran at nice 19 with affinity to logical processors 12 to 17, six
efficiency cores; the Python engines verify these settings before and after
timing. Before each GPU process, three consecutive GPU samples five seconds
apart must show at most 86 C, an enforced power limit of at least 70 W, the
GPU's default, and no software power cap or thermal slowdown. The GPU also
held one idle compute context of another workload, admitted by its process
ID, so every gate sample must also show at most 5% utilization; all showed
0%. A 100 ms telemetry monitor must bracket the timed window with no gap
above 350 ms, and a low enforced limit or thermal slowdown during timing
rejects the run. All 528 runs passed without retries.

`docs/benchmarks/release.json` holds three rotations: one of E5-large and
all-MiniLM-L6-v2 on the GPU and the processor, and one each of
modernbert-embed-base on the GPU and on the processor. Each records every
sample, interval and telemetry reading, the package versions, and the hashes
of the executable, models, graphs, fixtures and benchmark sources. Verify the
file without a GPU:

```text
python scripts/benchmarks/test_published_receipts.py
```

The check recomputes every aggregate from the samples and confirms that the
rotations together hold every case, engine and quality run of the release
once and that the recorded sources are byte-identical to this checkout: every
file under `crates` but its Markdown, the Cargo manifest and lockfile, the
toolchain file and the benchmark scripts. Between measurements the recorded
source hashes follow the repository: a change that cannot alter a measured
value updates them without a new measurement, and any other change requires
one.

### Changing lengths

The tables above time one warmed shape per case. A second measurement changes
the length on every call. E5-large takes one text of every length from 16 to
512 tokens, 497 texts, in the order `(83 i + 27) mod 497` over the texts in
ascending length, so no two consecutive calls share a length. A process loads
its model, settles for 30 seconds and runs the sequence three times with no
warm-up: the first pass meets every length on a fresh session, and the later
two repeat it. Each call is timed alone, and a pass's outputs stay live until
its clock stops.

For each fresh process, the receipt summarizes the 497 individually timed
calls of the first pass and the 994 calls of passes two and three pooled
together. It reports the mean, median, 99th percentile and maximum, plus the
median for each of three token ranges on the later passes. Each table entry
is the median of the corresponding statistic across three processes. The
first call was the slowest in every GPU process.

| Engine | Configuration that takes any length |
|---|---|
| KodamaEmbed | One `Embedder` reused across all calls in a process (`bench_query_api --sequence`) |
| ONNX Runtime CUDA | The FP16 graph with a symbolic sequence length, TF32 disabled |
| TensorRT | One FP16 engine whose profile spans 16, 264 and 512 tokens, builder optimization level 5 |
| PyTorch SDPA | FP16 BERT with scaled-dot-product attention, run eagerly |

The changing-length benchmark disables CUDA graph replay for ONNX Runtime
CUDA and TensorRT and runs PyTorch eagerly. KodamaEmbed keeps
recorded graphs for its most recent shapes, 64 in quantized precision and 32
in float, and this walk never returns to a kept one: quantized precision
records each call's launches while it issues them, and float precision
launches them directly. Padding every input to a few fixed lengths and
engines split by length range are not part of this measurement; the
fixed-length tables show each engine on a single shape. The centered CUDA
vectors of all 497 texts must equal the centered CPU run's byte for byte.
The rotation, gates and telemetry are those of the protocol above, and the
measurement is written to `docs/benchmarks/changing-lengths.json`, checked
by the same command.

Milliseconds per call; the last three columns are the later passes' medians
by token range:

| Engine | First pass mean | Later passes mean | 16-128 | 129-256 | 257-512 |
|---|---:|---:|---:|---:|---:|
| KodamaEmbed quantized | 4.863 | 4.881 | 2.182 | 4.010 | 6.525 |
| KodamaEmbed float | 5.492 | 5.548 | 3.587 | 4.620 | 7.182 |
| ONNX Runtime CUDA | 6.419 | 6.190 | 3.899 | 4.929 | 7.907 |
| TensorRT | 6.422 | 6.407 | 4.374 | 4.814 | 8.287 |
| PyTorch SDPA | 11.612 | 10.985 | 10.621 | 10.959 | 11.250 |

KodamaEmbed is the fastest engine in every column, in both precisions. Over
the later passes, the mean of the second-fastest engine, ONNX Runtime CUDA,
is 1.268 times quantized precision's and 1.116 times float precision's. By
range, the second-fastest engine's median is 1.787, 1.200 and 1.212 times
quantized precision's and 1.087, 1.042 and 1.101 times float precision's;
that engine is ONNX Runtime CUDA at 16 to 128 and at 257 to 512 tokens and
TensorRT at 129 to 256. All 18 runs passed without retries.

## Correctness

The centered CPU and CUDA modes return the same vector bits. The CUDA test
suite compares them at every length from 1 to 512 tokens, on random batches
and across the routing thresholds, and runs exhaustive checks: GELU on all
2^32 floats, the attention division over every mantissa pair, and the
quantization steps and weight indices on every float.

```text
cargo test --release -p kodamaembed-cuda --locked -- --test-threads=1 --include-ignored
```

The persistent plan's test needs a GPU with at least 40 streaming
multiprocessors; on a smaller GPU, add `--skip e5_stream_engine`.

In the release measurement, CUDA vectors for 26 real-data files, 27,541 rows
from SciFact, NFCorpus, FiQA and LoCoMo, equal the CPU's byte for byte, and
both runs report the same fingerprint. The release receipt records each
file's hash.

## Retrieval quality

Centered INT8 approximates FP32, and so does float precision on CUDA, whose
projection operands are FP16. The `saved-retrieval-quality-v1` gates, listed
in the [validator's guide](../crates/kodamaembed-validate/README.md#saved-vector-quality),
compare each model's scored run with FP32 CPU vectors of the same data:
centered INT8 for E5-large and float precision on CUDA for MiniLM and
modernbert-embed-base.

| Model | Dataset | Queries | FP32 nDCG@10 | Scored nDCG@10 | Query cosine distance | Top-10 overlap |
|---|---|---:|---:|---:|---:|---:|
| E5-large-v1 | SciFact | 300 | 0.7256 | 0.7251 | 2.4e-5 | 0.989 |
| E5-large-v1 | NFCorpus | 323 | 0.3616 | 0.3616 | 2.1e-5 | 0.985 |
| E5-large-v1 | FiQA | 243 | 0.5096 | 0.5076 | 2.1e-5 | 0.988 |
| E5-large-v1 | LoCoMo | 1,977 | 0.5506 | 0.5503 | 2.3e-5 | 0.989 |
| all-MiniLM-L6-v2 | SciFact | 300 | 0.6451 | 0.6451 | 1.9e-7 | 0.999 |
| all-MiniLM-L6-v2 | NFCorpus | 323 | 0.3184 | 0.3184 | 1.7e-7 | 1.000 |
| all-MiniLM-L6-v2 | FiQA | 243 | 0.4734 | 0.4734 | 1.6e-7 | 1.000 |
| all-MiniLM-L6-v2 | LoCoMo | 1,977 | 0.2756 | 0.2755 | 1.4e-7 | 0.999 |
| modernbert-embed-base | SciFact | 300 | 0.6989 | 0.6974 | 4.4e-7 | 0.999 |
| modernbert-embed-base | NFCorpus | 323 | 0.3333 | 0.3335 | 5.4e-7 | 0.998 |
| modernbert-embed-base | FiQA | 243 | 0.5155 | 0.5156 | 4.1e-7 | 0.999 |
| modernbert-embed-base | LoCoMo | 1,977 | 0.4876 | 0.4877 | 4.0e-7 | 0.999 |

nDCG@10 uses binary relevance. Every passage judged relevant counts once,
whatever its grade. All three models passed the gates on all four datasets
and all 13 tasks per model. The largest dataset-average nDCG@10 loss was
0.00194 for E5-large on the FiQA subset. The largest individual-task loss was
0.00401 for E5-large on LoCoMo conversation `conv-50`, within the 0.005 limit.
modernbert-embed-base's largest loss, by dataset and by task, was 0.00150 on
SciFact. SciFact, NFCorpus and LoCoMo were used during development.

`run_quality.py` embeds each dataset with FP32 on the CPU, the reference, and
with each model's scored run: centered INT8 on the CPU for E5-large, whose
CUDA vectors must equal the CPU's, and float precision on CUDA for MiniLM and
modernbert-embed-base.
The validator's `compare-retrieval` scores that run against the reference and
fails when an aggregate gate fails, and every vector file's hash is recorded.
A fourth dataset, the first 10,000 passages of FiQA with the test queries
judged against them, was used in no design or calibration step.
The models and datasets are separate downloads under their own licenses;
[Reproducing the measurements](#reproducing-the-measurements) lists them.

## Limits

- The receipts cover one processor and one GPU, an NVIDIA RTX 4000 SFF Ada
  Generation with 48 streaming multiprocessors under its 70 W power limit.
- The server ran other workloads during the measurement: the CPU engines had
  six efficiency cores at nice 19, and the GPU held another workload's idle
  compute context.
- Every call embeds one text; bulk throughput is not measured.
- The processor comparison ends at 2,048 tokens: modernbert-embed-base's
  4,096- and 8,192-token cases ran on the GPU only.
- The fixed-length comparison used CUDA graph replay for the three other GPU
  engines. The changing-length comparison used dynamic shapes in ONNX Runtime
  CUDA and TensorRT and eager execution in PyTorch. Both comparisons used
  FP16 and TensorRT builder optimization level 5.
- In a separate development run on an RTX 3080 Ti Laptop GPU, E5-large in
  float precision was slower than TensorRT FP16 at 64, 128, 256 and 512
  tokens. The published receipts cover the RTX 4000 SFF Ada Generation.

## Reproducing the measurements

The harness runs on Linux, Windows and macOS with Python 3.11 or newer. GPU
runs need one supported NVIDIA GPU with `CUDA_VISIBLE_DEVICES` unset. The
quantized runs also need an x86-64 host with AVX-VNNI, AVX2, FMA and F16C
to prepare the model. The commands use Linux and macOS paths. On Windows,
a virtual environment's interpreter is `.venv\Scripts\python.exe`.

Download all-MiniLM-L6-v2 and modernbert-embed-base at their revisions under
[Models by name](integration.md#models-by-name) into `models/minilm` and
`models/modernbert-embed-base`, the E5-large checkpoint under
[Exact INT8 checkpoints](integration.md#exact-int8-checkpoints) into
`models/e5-large`, and the datasets into `data`. Each dataset directory holds
`corpus.jsonl`, `queries.jsonl` and the `test.tsv` from its `qrels` directory:

| File | Source | SHA-256 |
|---|---|---|
| `data/scifact/corpus.jsonl` | [mteb/scifact](https://huggingface.co/datasets/mteb/scifact) | `f0d32db0d156b526d75921ed7a76f2cb912902631c87248c5c97c617bad0b60c` |
| `data/scifact/queries.jsonl` | [mteb/scifact](https://huggingface.co/datasets/mteb/scifact) | `9db7df096f7414435d52bafcbacf814c30cea50eb565c1e8fa6d11440759bba8` |
| `data/scifact/test.tsv` | [mteb/scifact](https://huggingface.co/datasets/mteb/scifact) | `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c` |
| `data/nfcorpus/corpus.jsonl` | [BEIR NFCorpus](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) | `10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f` |
| `data/nfcorpus/queries.jsonl` | [BEIR NFCorpus](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) | `d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a` |
| `data/nfcorpus/test.tsv` | [BEIR NFCorpus](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) | `f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a` |
| `data/fiqa/corpus.jsonl` | [BEIR FiQA](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip) | `ff593e4df9933955dc3af83be0c3fa28ac7465f627e08c2e53593e734d506517` |
| `data/fiqa/queries.jsonl` | [BEIR FiQA](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip) | `eede1e61d4a0188940239b53ebc2da91f577a6a34679c812d1eb9090c29877bc` |
| `data/fiqa/test.tsv` | [BEIR FiQA](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip) | `6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44` |
| `data/locomo10.json` | [LoCoMo](https://github.com/snap-research/locomo/blob/main/data/locomo10.json) | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |

The quality run reads FiQA's first 10,000 passages, which bound its FP32
reference run on a processor:

```text
python scripts/benchmarks/prepare_quality_subset.py --source data/fiqa --out data/fiqa-10k --passages 10000
```

Build the benchmark executables and create the environment for graph export,
fixtures and the CPU engines:

```text
cargo build --release --locked -p kodamaembed --example bench_query_api
cargo build --release --locked -p kodamaembed-validate
python -m venv .venv-export
.venv-export/bin/python -m pip install -r scripts/benchmarks/requirements-export.txt
```

Export the graphs, select the fixtures and copy the cases:

```text
.venv-export/bin/python scripts/benchmarks/export_bert.py --model-dir models/minilm --out-dir target/minilm-onnx --pooled
.venv-export/bin/python scripts/benchmarks/export_bert.py --model-dir models/e5-large --out-dir target/e5-onnx --pooled
.venv-export/bin/python scripts/benchmarks/export_bert.py --model-dir models/modernbert-embed-base --out-dir target/modernbert-embed-onnx --pooled
.venv-export/bin/python scripts/benchmarks/prepare_benchmark_fixtures.py --corpus data/scifact/corpus.jsonl --model-dir models/minilm --family minilm --out-dir target/matched/fixtures/minilm
.venv-export/bin/python scripts/benchmarks/prepare_benchmark_fixtures.py --corpus data/scifact/corpus.jsonl --model-dir models/e5-large --family e5 --out-dir target/matched/fixtures/e5
.venv-export/bin/python scripts/benchmarks/prepare_benchmark_fixtures.py --corpus data/scifact/corpus.jsonl --model-dir models/modernbert-embed-base --family modernbert-embed --out-dir target/matched/fixtures/modernbert-embed --running-text
cp scripts/benchmarks/cases.example.json target/matched/cases.json
```

`--running-text` continues each document with those after it: no SciFact
document reaches modernbert-embed-base's longer cases.

The GPU engines have their own environments; the PyTorch wheel comes from
PyTorch's CUDA index:

```text
python -m venv .venv
.venv/bin/python -m pip install -r scripts/benchmarks/requirements.txt
python -m venv .venv-torch
.venv-torch/bin/python -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv-torch/bin/python -m pip install -r scripts/benchmarks/requirements-torch.txt
```

Before the first rotation, prepare each case once per GPU engine, one process
at a time. Preparation checks the outputs, builds the TensorRT engine and
writes E5-large's centered calibration cache:

```text
target/release/examples/bench_query_api --fixture target/matched/fixtures/minilm/minilm-b1-l16.json --model-dir models/minilm --mode cuda-fp16 --out target/matched/prepare/minilm-b1-l16-kodamaembed.json --prepare-only
target/release/examples/bench_query_api --fixture target/matched/fixtures/e5/e5-b1-l16.json --model-dir models/e5-large --mode cuda-centered --out target/matched/prepare/e5-b1-l16-kodamaembed-centered.json --prepare-only
.venv/bin/python scripts/benchmarks/bench_ort_query.py --fixture target/matched/fixtures/minilm/minilm-b1-l16.json --model-dir models/minilm --export-receipt target/minilm-onnx/export.json --onnx target/minilm-onnx/model-gpu-fp16-pooled.onnx --provider cuda --out target/matched/prepare/minilm-b1-l16-cuda.json --prepare-only
.venv/bin/python scripts/benchmarks/bench_ort_query.py --fixture target/matched/fixtures/minilm/minilm-b1-l16.json --model-dir models/minilm --export-receipt target/minilm-onnx/export.json --onnx target/minilm-onnx/model-pooled.onnx --provider tensorrt --cache target/matched/cache/minilm-b1-l16 --level 5 --out target/matched/prepare/minilm-b1-l16-trt.json --prepare-only
.venv-torch/bin/python scripts/benchmarks/bench_torch_query.py --fixture target/matched/fixtures/minilm/minilm-b1-l16.json --model-dir models/minilm --out target/matched/prepare/minilm-b1-l16-torch.json --prepare-only --profile
```

Then run the rotations. A rotation takes the cases that `--families` and
`--max-tokens` select, so a model is measured by rotations of its own and no
other model runs again:

```text
python scripts/benchmarks/run_query_comparison.py --cases target/matched/cases.json --families e5,minilm --engines gpu,cpu --python .venv/bin/python --torch-python .venv-torch/bin/python --threads 6 --cpu-threads 6 --cpus 12-17 --trt-level 5 --warm 2 --max-run-minutes 60 --output target/matched/rotation
python scripts/benchmarks/run_query_comparison.py --cases target/matched/cases.json --families modernbert-embed --engines gpu --python .venv/bin/python --torch-python .venv-torch/bin/python --threads 6 --cpus 12-17 --trt-level 5 --warm 2 --max-run-minutes 60 --output target/matched/rotation-modernbert-embed-gpu
python scripts/benchmarks/run_query_comparison.py --cases target/matched/cases.json --families modernbert-embed --max-tokens 2048 --engines cpu --python .venv/bin/python --torch-python .venv-torch/bin/python --threads 6 --cpu-threads 6 --cpus 12-17 --warm 2 --max-run-minutes 60 --output target/matched/rotation-modernbert-embed-cpu
```

`--engines` accepts groups or a comma-separated list of engines. For an
FP32-only CPU comparison, use `--engines ort-cpu,torch-cpu,kodamaembed-cpu-fp32`
with `.venv-export/bin/python` as both interpreters. On Linux and Windows,
set `--cpus` to processors available on your machine. Omit it on macOS.
`--max-run-minutes 60` replaces the default limit of 15 minutes on one run:
the longest, modernbert-embed-base at 2,048 tokens on the processor, takes
about 11 minutes. With GPU engines, the published runs also passed
`--max-wait-minutes 720` and one `--allowed-idle-pid`. The Windows default is
`--priority high`.
Linux and macOS inherit the caller's scheduling priority by default.
Each GPU run first waits for three
consecutive passing readings, by default at most 86 C and an enforced power
limit of at least the GPU's default; `--allowed-idle-pid <pid>` admits one
identified idle GPU context and then requires at most 5% GPU utilization in
every reading. The output directory must be new or empty, and a rejected run
stops the rotation with its evidence retained.

Run each model's quality comparison and export your fixed-length receipt under
`target/`: the first rotation into a new file, each later one appended to it.

```text
python scripts/benchmarks/run_quality.py --minilm models/minilm --e5 models/e5-large --scifact data/scifact --nfcorpus data/nfcorpus --fiqa data/fiqa-10k --locomo data/locomo10.json --cuda --output target/quality
python scripts/benchmarks/export_receipt.py --rotation target/matched/rotation --quality target/quality --disclosures disclosures.json --out target/matched/release.json --partial
python scripts/benchmarks/run_quality.py --modernbert-embed models/modernbert-embed-base --scifact data/scifact --nfcorpus data/nfcorpus --fiqa data/fiqa-10k --locomo data/locomo10.json --cuda --output target/quality-modernbert-embed
python scripts/benchmarks/export_receipt.py --rotation target/matched/rotation-modernbert-embed-gpu --quality target/quality-modernbert-embed --disclosures disclosures.json --out target/matched/release.json --append --partial
python scripts/benchmarks/export_receipt.py --rotation target/matched/rotation-modernbert-embed-cpu --disclosures disclosures.json --out target/matched/release.json --append
```

`disclosures.json` holds the limitations of the measurement and every stopped
rotation with its reason; `export_receipt.py --help` shows its format.
`--append` adds a rotation to the receipt that `--out` holds. `--partial`
accepts a receipt that does not yet hold the whole release, so the last
export, without it, requires every case, engine and quality run. A model
added to `FAMILIES` in `_matched.py` joins the receipt the same way: its
cases, its own rotations, an append. `--check` writes nothing: it repeats a
rotation's export and requires the receipt to hold exactly that rotation.

The changing-lengths measurement has its own fixture, cases and receipt.
`--sequence` adds `e5-sequence.json` to the fixtures; write it to a new
directory, build TensorRT's engine for it, then run the rotation and export:

```text
.venv-export/bin/python scripts/benchmarks/prepare_benchmark_fixtures.py --corpus data/scifact/corpus.jsonl --model-dir models/e5-large --family e5 --out-dir target/matched/fixtures/e5-sequence --sequence
cp scripts/benchmarks/sequence.example.json target/matched/sequence.json
.venv/bin/python scripts/benchmarks/bench_ort_query.py --sequence target/matched/fixtures/e5-sequence/e5-sequence.json --model-dir models/e5-large --export-receipt target/e5-onnx/export.json --onnx target/e5-onnx/model-pooled.onnx --provider tensorrt --cache target/matched/cache/e5-sequence --level 5 --out target/matched/prepare/e5-sequence-trt.json --prepare-only
python scripts/benchmarks/run_length_sequence.py --cases target/matched/sequence.json --engines gpu,kodamaembed-cpu-centered --python .venv/bin/python --torch-python .venv-torch/bin/python --threads 6 --cpu-threads 6 --cpus 12-17 --trt-level 5 --max-run-minutes 60 --output target/matched/sequence
python scripts/benchmarks/sequence_receipt.py --rotation target/matched/sequence --disclosures disclosures.json --out target/matched/changing-lengths.json
```

`run_length_sequence.py` takes the options of `run_query_comparison.py` but
`--warm`. The centered CPU engine is in the rotation so that the receipt
holds the vectors the centered CUDA runs must equal; its runs take minutes
each, hence the longer run limit.
