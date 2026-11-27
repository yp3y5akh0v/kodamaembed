# Checks candidate public files against an allowlist and secret patterns.
[CmdletBinding()]
param([switch]$RequireSingleCommit)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location $repo
try {
    $allowed = @(
        '^(\.gitattributes|\.gitignore|Cargo\.(toml|lock)|LICENSE|README\.md|CONTRIBUTING\.md|CHANGELOG\.md|SECURITY\.md|deny\.toml|rust-toolchain\.toml|pyproject\.toml)$',
        '^\.github/workflows/[a-z0-9_-]+\.ya?ml$',
        '^\.github/(dependabot|FUNDING)\.yml$',
        '^\.github/ISSUE_TEMPLATE/(bug-report|model-request|feature-request|config)\.yml$',
        '^scripts/(build-kernels|build-shaders|check-release|test-release|check-kernel-repro)\.ps1$',
        '^scripts/install-(nvcc|glslc)\.sh$',
        '^scripts/reference_vectors\.py$',
        '^scripts/benchmarks/(_matched|_process|_sequence|_system|_telemetry|bench_ort_query|bench_torch_query|export_bert|export_receipt|prepare_benchmark_fixtures|prepare_quality_subset|receipts|run_length_sequence|run_quality|run_query_comparison|scripted_nvidia_smi|sequence_receipt)\.py$',
        '^scripts/benchmarks/test_(matched|process|public_acceptance|published_receipts|quality|rotation|runner|sequence|telemetry|torch_query)\.py$',
        '^scripts/benchmarks/(requirements(-export|-torch)?\.txt|(cases|sequence)\.example\.json)$',
        '^docs/(integration|architecture|benchmarks|python|migration)\.md$',
        '^docs/assets/(logo\.svg|crest\.svg|favicon\.ico|social-preview\.png)$',
        '^docs/benchmarks/(release|changing-lengths)\.json$',
        '^paper/kodamaembed\.(tex|pdf)$',
        '^notebooks/quickstart\.ipynb$',
        '^packaging/licenses/(about\.toml|about\.hbs|THIRD_PARTY_LICENSES\.html)$',
        '^crates/kodamaembed(-(cpu|cuda|model|python|validate|vulkan))?/(Cargo\.toml|LICENSE|README\.md)$',
        '^crates/kodamaembed(-(cpu|cuda|model|python|validate|vulkan))?/(src|tests|examples)/([a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\.rs$',
        '^python/README\.md$',
        '^python/kodamaembed/([a-z_]+\.pyi?|py\.typed)$',
        '^python/tests/[a-z_]+\.py$',
        '^crates/kodamaembed-cuda/kernels/[a-z0-9_]+\.(cu|cuh)$',
        '^crates/kodamaembed-cuda/ptx/(attention|gemm|gemm_paired|norm|pool|prediction_head|repro)\.ptx$',
        '^crates/kodamaembed-cpu/src/int8/e5_v1_signs\.bin$',
        '^crates/kodamaembed/calibration/passages\.txt$',
        '^crates/kodamaembed/tests/registry/vectors\.json$',
        '^crates/kodamaembed-vulkan/shaders/[a-z0-9_]+\.(comp|glsl)$',
        '^crates/kodamaembed-vulkan/spv/[a-z0-9_]+\.spv$'
    )
    $rules = [ordered]@{
        'personal filesystem path' = '(?i)([A-Z]:(?:\\\\|\\|/)(Users|Documents and Settings)(?:\\\\|\\|/)|/(home|Users)/[a-z0-9_.-]+/)'
        'private key' = '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
        'credential-shaped token' = '(\bAKIA[A-Z0-9]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{50,}\b)'
        'API secret token' = '\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'
        'embedded URL credentials' = '(?i)https?://[^\s/:@]+:[^\s/@]+@'
    }
    $files = @(git -c core.quotepath=false ls-files --cached --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate repository files' }
    $problems = [Collections.Generic.List[string]]::new()
    $checked = 0
    foreach ($relative in ($files | Sort-Object -Unique)) {
        $path = [IO.Path]::GetFullPath((Join-Path $repo $relative))
        if (-not $path.StartsWith($repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            $problems.Add("$relative : outside repository")
            continue
        }
        # Unstaged deletions are absent from the candidate working tree.
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
        if (-not ($allowed | Where-Object { $relative -cmatch $_ })) {
            $problems.Add("$relative : not in public file allowlist")
            continue
        }
        if ((Get-Item -LiteralPath $path -Force).LinkType) {
            $problems.Add("$relative : symbolic links require explicit review")
            continue
        }
        if ($relative -cmatch '^crates/[^/]+/LICENSE$' -and
            (Get-FileHash -LiteralPath $path).Hash -ne (Get-FileHash -LiteralPath (Join-Path $repo 'LICENSE')).Hash) {
            $problems.Add("$relative : package license differs from workspace license")
        }
        $text = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($path))
        foreach ($rule in $rules.GetEnumerator()) {
            if ($text -match $rule.Value) { $problems.Add("$relative : $($rule.Key)") }
        }
        $checked++
    }
    if ($RequireSingleCommit) {
        $shallow = git rev-parse --is-shallow-repository
        if ($LASTEXITCODE -ne 0 -or $shallow -ne 'false') {
            $problems.Add('A complete local history is required to verify the initial public commit')
        }
        $count = git rev-list --count HEAD
        if ($LASTEXITCODE -ne 0 -or $count -ne '1') {
            $problems.Add('HEAD must have exactly one reachable commit for the initial public release')
        }
        $dirty = git status --porcelain --untracked-files=normal
        if ($LASTEXITCODE -ne 0 -or $dirty) {
            $problems.Add('The initial public release must have a clean working tree')
        }
    }
    if ($problems.Count) {
        $problems | ForEach-Object { Write-Output $_ }
        throw "Release check failed with $($problems.Count) finding(s); matched content is not printed"
    }
    Write-Output "Validated $checked candidate public files."
} finally {
    Pop-Location
}
