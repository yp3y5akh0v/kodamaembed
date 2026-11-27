# Contributing

Use the Rust toolchain in `rust-toolchain.toml`, keep `Cargo.lock`, and run
commands from the workspace root. Native builds need the platform's linker
and C toolchain for tokenizer dependencies. Standard builds use the shipped
PTX and SPIR-V.

## Host checks

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test -p kodamaembed --locked
cargo test -p kodamaembed-model --locked
cargo test -p kodamaembed-cpu --locked
cargo test -p kodamaembed-cpu --examples --locked
cargo test -p kodamaembed-validate --locked
cargo test -p kodamaembed-cuda --lib --locked
cargo test -p kodamaembed-cuda --doc --locked
cargo test -p kodamaembed-vulkan --lib --locked
cargo test -p kodamaembed-vulkan --doc --locked
```

Set `RUSTDOCFLAGS` to `-D warnings` when building documentation:

```text
cargo doc --workspace --no-deps --locked
```

Check portable targets separately:

```text
rustup target add wasm32-unknown-unknown
cargo check -p kodamaembed-cpu --lib --examples --target wasm32-unknown-unknown --locked
cargo check -p kodamaembed --lib --no-default-features --target wasm32-unknown-unknown --locked
rustup target add aarch64-unknown-linux-gnu
cargo check -p kodamaembed-cuda --all-targets --target aarch64-unknown-linux-gnu --locked
```

The wasm build's own tests run inside Node, through the test runner of the
`wasm-bindgen` version in `Cargo.lock`; set
`CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER` to `wasm-bindgen-test-runner`:

```text
cargo install wasm-bindgen-cli --version 0.2.130 --locked
cargo test -p kodamaembed --lib --no-default-features --target wasm32-unknown-unknown --locked
```

CI runs host checks on Windows, Linux and macOS on Apple Silicon,
cross-checks CUDA development targets for ARM Linux and runs the wasm tests.

## GPU and numerical checks

Run one GPU test process at a time:

```text
cargo test --release -p kodamaembed-cuda --locked -- --test-threads=1
cargo test --release -p kodamaembed-vulkan --locked -- --test-threads=1
```

The Vulkan tests open every Vulkan adapter on the machine, and nine need an
adapter with cooperative matrices; `--skip where_supported` skips those.

The extended CUDA run includes every-length and random-batch comparisons,
persistent dispatch and storage reuse, pinned host-buffer alignment, and
exhaustive GELU and attention-division checks:

```text
cargo test --release -p kodamaembed-cuda --locked -- --test-threads=1 --include-ignored
```

The persistent plan's test needs a GPU with at least 40 streaming
multiprocessors; on a smaller GPU, add `--skip e5_stream_engine`.

Changes to centered GELU or attention division need the exhaustive checks,
and the centered CPU and CUDA modes must keep identical vector bits. A new
numerical source file joins its crate's `NUMERICAL_SOURCES`, which
fingerprints hash. Numerical changes need real-data quality checks with
aggregate and individual task results and recorded vector comparisons;
[retrieval quality](docs/benchmarks.md#retrieval-quality) lists the current
results.

A bug fix needs a regression test that fails with the defect reapplied and
passes with the fix.

## Regenerate GPU artifacts

The shipped PTX comes from nvcc 12.9.86 on x86-64 Linux, the build CI
checks: NVIDIA's Windows build of the same nvcc emits different PTX from the
same source. `scripts/install-nvcc.sh` installs it from NVIDIA's
redistributable archives; it needs GCC 14 or older, and on Windows it runs in
WSL. The PowerShell 7 scripts rebuild every kernel, then rebuild them in two
checkout directories and compare the PTX:

```text
bash scripts/install-nvcc.sh ~/nvcc-12.9.86
export PATH=~/nvcc-12.9.86/bin:$PATH
pwsh scripts/build-kernels.ps1
pwsh scripts/check-kernel-repro.ps1
```

The shipped SPIR-V comes from the glslc of Vulkan SDK 1.4.328.1, which the
script finds on `PATH`. LunarG's installer puts it there on Windows, and
`scripts/install-glslc.sh` installs it on x86-64 Linux, the build CI checks:

```text
bash scripts/install-glslc.sh ~/vulkan-1.4.328.1
export PATH=~/vulkan-1.4.328.1/bin:$PATH
pwsh scripts/build-shaders.ps1
```

Kernel source changes and their regenerated PTX or SPIR-V belong in the same
commit, and CI requires the rebuilt PTX and SPIR-V byte for byte. Numerical
source files, comments included, contribute to fingerprints.

## Release checks

The public file allowlist in `scripts/check-release.ps1` and its tests:

```powershell
./scripts/test-release.ps1
./scripts/check-release.ps1
```

The tagged release workflow builds wheels for five platforms and a Python
source distribution on GitHub-hosted runners. It installs and tests each
platform's wheel before publishing the wheels and source distribution to
PyPI. Locally built wheels can include the developer's home directory path
in the native extension and software bill of materials.

Pushes to `master` run CI. Pushing a version tag such as `v0.1.0` triggers
the PyPI release and creates a draft GitHub release after publication. For
the first PyPI upload, configure a trusted publisher for `kodamaembed` using
the repository's `release.yml` workflow and `pypi` environment.

Publish the Rust crates separately to crates.io from the same release commit.
Publish `kodamaembed-model` first, then `kodamaembed-cpu`, `kodamaembed-cuda`
and `kodamaembed-vulkan`, and finally `kodamaembed`.

CI also checks the benchmark harness and the published receipts on Linux,
Windows and macOS, without a GPU or model files. These checks use only the
Python standard library:

```text
python scripts/benchmarks/test_published_receipts.py
cd scripts/benchmarks
python -m unittest test_process test_telemetry test_runner test_quality
```

The engine tests run the ONNX Runtime and PyTorch engines on the CPU, in the
environment from `scripts/benchmarks/requirements-export.txt`:

```text
python -m unittest test_matched test_public_acceptance test_torch_query test_rotation test_sequence
```

The receipt checker confirms that the recorded sources match the checkout:
every file under `crates` but its Markdown, the Cargo manifest and lockfile,
the toolchain file and the benchmark scripts. Changing one requires updated
receipts, and a new measurement when the change can alter a measured value.
