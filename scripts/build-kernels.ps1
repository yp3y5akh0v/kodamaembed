# Compiles every kernel under crates/kodamaembed-cuda/kernels to PTX for compute
# capability 8.0 (the driver JIT-compiles it for 8.0 and newer GPUs).
$ErrorActionPreference = "Stop"
# NVIDIA's Windows build of nvcc 12.9.86 emits different PTX from the same
# source, so the shipped PTX comes from the Linux build CI checks.
if (-not $IsLinux) { throw "Build the PTX on Linux (on Windows, in WSL); see CONTRIBUTING.md" }
if (-not ((nvcc --version) -match 'V12\.9\.86$')) { throw "nvcc 12.9.86 is required; scripts/install-nvcc.sh installs it" }
$crate = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../crates/kodamaembed-cuda"))
$ptx = Join-Path $crate "ptx"
New-Item -ItemType Directory -Force $ptx | Out-Null
foreach ($source in Get-ChildItem (Join-Path $crate "kernels") -Filter *.cu) {
    $target = Join-Path $ptx ($source.BaseName + ".ptx")
    nvcc -ptx -arch=compute_80 -O3 -std=c++17 -o $target $source.FullName
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed on $($source.Name)" }
    "built $($source.BaseName).ptx"
}
