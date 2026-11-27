#!/usr/bin/env bash
# Installs nvcc 12.9.86, which builds the shipped PTX, from NVIDIA's x86-64 Linux archives.
# Usage: scripts/install-nvcc.sh DIRECTORY
set -euo pipefail
target=$1
redist=https://developer.download.nvidia.com/compute/cuda/redist
mkdir -p "$target"
download=$(mktemp)
trap 'rm -f "$download"' EXIT
while read -r component version sha256; do
    curl -fsSL --retry 3 -o "$download" "$redist/$component/linux-x86_64/$component-linux-x86_64-$version-archive.tar.xz"
    echo "$sha256  $download" | sha256sum --check --strict --quiet
    tar -xJf "$download" -C "$target" --strip-components=1
done <<'ARCHIVES'
cuda_nvcc 12.9.86 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
cuda_cudart 12.9.79 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
cuda_cccl 12.9.27 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa
ARCHIVES
"$target/bin/nvcc" --version | tail -1
