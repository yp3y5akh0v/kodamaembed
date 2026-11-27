#requires -Version 7.0
$ErrorActionPreference = 'Stop'
$parent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar)
$root = Join-Path $parent ('kodamaembed-release-test-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path (Join-Path $root 'scripts'), (Join-Path $root 'crates/kodamaembed/calibration')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'check-release.ps1') -Destination (Join-Path $root 'scripts')
$failures = [Collections.Generic.List[string]]::new()

function Assert-Release([string]$Name, [bool]$Accept) {
    $errorFound = $null
    try { & (Join-Path $root 'scripts/check-release.ps1') | Out-Null } catch { $errorFound = $_ }
    if (($null -eq $errorFound) -ne $Accept) {
        $failures.Add($Name)
        Write-Output "FAIL $Name"
    } else { Write-Output "PASS $Name" }
}

try {
    & git init --quiet $root
    if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize release fixture' }
    [IO.File]::WriteAllText((Join-Path $root 'README.md'), '# Fixture')
    [IO.File]::WriteAllText((Join-Path $root 'crates/kodamaembed/calibration/passages.txt'), 'Required calibration input.')
    Assert-Release 'required calibration input is public' $true

    [IO.File]::WriteAllText((Join-Path $root '.gitattributes'), '* text=auto eol=lf')
    Assert-Release 'allowed dotfile is accepted' $true

    $unexpected = Join-Path $root 'private-note.txt'
    [IO.File]::WriteAllText($unexpected, 'Private notes')
    Assert-Release 'path outside the allowlist is rejected' $false
    Remove-Item -LiteralPath $unexpected

    $package = Join-Path $root 'python/kodamaembed'
    $null = New-Item -ItemType Directory -Path $package
    [IO.File]::WriteAllText((Join-Path $package '__init__.py'), '"""Fixture."""')
    Assert-Release 'Python package source is public' $true

    $reference = Join-Path $root 'crates/kodamaembed/tests/registry'
    $null = New-Item -ItemType Directory -Path $reference
    [IO.File]::WriteAllText((Join-Path $reference 'vectors.json'), '{}')
    [IO.File]::WriteAllText((Join-Path $root 'scripts/reference_vectors.py'), '"""Fixture."""')
    Assert-Release 'the reference vectors and their generator are public' $true

    $extension = Join-Path $package '_kodamaembed.pyd'
    [IO.File]::WriteAllText($extension, 'compiled')
    Assert-Release 'a built extension module is rejected' $false
    Remove-Item -LiteralPath $extension

    $credential = 'https://' + 'user:password@' + 'example.test'
    [IO.File]::WriteAllText((Join-Path $root 'README.md'), $credential)
    Assert-Release 'credentials in an allowed file are rejected' $false
    [IO.File]::WriteAllText((Join-Path $root 'README.md'), '# Fixture')
    Assert-Release 'clean fixture is accepted again' $true

    $projectFiles = @(
        'CHANGELOG.md', 'SECURITY.md', 'deny.toml', 'docs/migration.md', 'notebooks/quickstart.ipynb',
        'docs/assets/logo.svg', 'docs/assets/crest.svg', 'docs/assets/favicon.ico', 'docs/assets/social-preview.png',
        '.github/ISSUE_TEMPLATE/bug-report.yml', '.github/ISSUE_TEMPLATE/model-request.yml',
        '.github/ISSUE_TEMPLATE/feature-request.yml', '.github/ISSUE_TEMPLATE/config.yml',
        '.github/FUNDING.yml',
        'packaging/licenses/about.toml', 'packaging/licenses/about.hbs',
        'packaging/licenses/THIRD_PARTY_LICENSES.html'
    )
    foreach ($relative in $projectFiles) {
        $path = Join-Path $root $relative
        $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path)
        [IO.File]::WriteAllText($path, 'Fixture')
    }
    Assert-Release 'project documents, logo files, issue forms and license files are public' $true

    $notebook = Join-Path $root 'notebooks/scratch.ipynb'
    [IO.File]::WriteAllText($notebook, '{}')
    Assert-Release 'an unlisted notebook is rejected' $false
    Remove-Item -LiteralPath $notebook

    foreach ($compiler in 'nvcc', 'glslc') {
        [IO.File]::WriteAllText((Join-Path $root "scripts/install-$compiler.sh"), '# Fixture')
    }
    Assert-Release 'the installers of the pinned GPU compilers are public' $true
    if ($failures.Count) { throw ('Release check tests failed: ' + ($failures -join ', ')) }
} finally {
    $resolved = [IO.Path]::GetFullPath($root)
    if ([IO.Path]::GetDirectoryName($resolved) -ne $parent -or
        [IO.Path]::GetFileName($resolved) -notlike 'kodamaembed-release-test-*') {
        throw 'Unexpected cleanup target'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
