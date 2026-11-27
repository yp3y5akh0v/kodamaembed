# Rebuilds kernels in two checkout locations and compares their PTX bytes.
[CmdletBinding()]
param(
    [ValidateSet('attention', 'gemm', 'gemm_paired', 'norm', 'pool', 'prediction_head', 'repro')]
    [string[]]$Kernels = @('attention', 'gemm', 'gemm_paired', 'norm', 'pool', 'prediction_head', 'repro')
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$scratch = Join-Path $repo ('target/kernel-repro/' + [Guid]::NewGuid().ToString('N'))
$copies = @('first', 'second') | ForEach-Object { Join-Path $scratch $_ }
foreach ($copy in $copies) {
    $scripts = Join-Path $copy 'scripts'
    $sources = Join-Path $copy 'crates/kodamaembed-cuda/kernels'
    New-Item -ItemType Directory -Force $scripts, $sources | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build-kernels.ps1') -Destination $scripts
    Get-ChildItem -LiteralPath (Join-Path $repo 'crates/kodamaembed-cuda/kernels') -Filter '*.cuh' |
        Copy-Item -Destination $sources
    foreach ($name in $Kernels) {
        Copy-Item -LiteralPath (Join-Path $repo "crates/kodamaembed-cuda/kernels/$name.cu") -Destination $sources
    }
    & (Join-Path $scripts 'build-kernels.ps1')
}
foreach ($name in $Kernels) {
    $hashes = @($copies | ForEach-Object {
        (Get-FileHash -LiteralPath (Join-Path $_ "crates/kodamaembed-cuda/ptx/$name.ptx")).Hash
    })
    if ($hashes[0] -ne $hashes[1]) {
        throw "$name.ptx differs between checkout locations; outputs retained in $scratch"
    }
    Write-Output "$name.ptx identical: $($hashes[0])"
}
