# Compiles every compute shader under crates/kodamaembed-vulkan/shaders to SPIR-V
# for Vulkan 1.3 (.glsl files are includes) with the glslc of Vulkan SDK 1.4.328.1,
# the build CI checks.
$ErrorActionPreference = "Stop"
if (-not ((glslc --version | Select-Object -First 1) -match 'v2025\.3-10-gc7e73e8$')) {
    throw "glslc from Vulkan SDK 1.4.328.1 is required; scripts/install-glslc.sh installs it on Linux"
}
$crate = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../crates/kodamaembed-vulkan"))
$shaders = Join-Path $crate "shaders"
$spv = Join-Path $crate "spv"
New-Item -ItemType Directory -Force $spv | Out-Null
# A binary no shader builds any more must not stay.
Remove-Item (Join-Path $spv "*.spv")
foreach ($source in Get-ChildItem $shaders -Filter *.comp) {
    $target = Join-Path $spv ($source.BaseName + ".spv")
    glslc --target-env=vulkan1.3 -O -o $target $source.FullName
    if ($LASTEXITCODE -ne 0) { throw "glslc failed on $($source.Name)" }
    "built $($source.BaseName).spv"
}
# MPNet's relative-position bias and ModernBERT's attention window, paired projection columns
# and prediction head build as variants, so the shaders without them stay unchanged.
$attention = "attention", "attention_matrix"
$variants = @{ Suffix = "bias"; Define = "RELATIVE_BIAS"; Shaders = $attention },
    @{ Suffix = "local"; Define = "LOCAL"; Shaders = $attention },
    @{ Suffix = "paired"; Define = "PAIRED"; Shaders = "gemm_matrix", "gemm_rows", "gemm_portable" },
    @{ Suffix = "prediction"; Define = "PREDICTION"; Shaders = @("rerank") }
foreach ($variant in $variants) {
    foreach ($name in $variant.Shaders) {
        $target = Join-Path $spv "$($name)_$($variant.Suffix).spv"
        glslc --target-env=vulkan1.3 -O "-D$($variant.Define)" -o $target (Join-Path $shaders "$name.comp")
        if ($LASTEXITCODE -ne 0) { throw "glslc failed on $name.comp with $($variant.Define)" }
        "built $($name)_$($variant.Suffix).spv"
    }
}
