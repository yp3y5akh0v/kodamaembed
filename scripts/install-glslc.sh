#!/usr/bin/env bash
# Installs glslc from Vulkan SDK 1.4.328.1, which builds the shipped SPIR-V, from LunarG's x86-64
# Linux archive.
# Usage: scripts/install-glslc.sh DIRECTORY
set -euo pipefail
target=$1
version=1.4.328.1
mkdir -p "$target/bin"
download=$(mktemp)
trap 'rm -f "$download"' EXIT
curl -fsSL --retry 3 -o "$download" "https://sdk.lunarg.com/sdk/download/$version/linux/vulkansdk-linux-x86_64-$version.tar.xz"
echo "241e75b56c91c0d210ed07a7c638ec05a3e5b0e4c66ba9f0ba0f102d823ad6bf  $download" | sha256sum --check --strict --quiet
tar -xJf "$download" -C "$target/bin" --strip-components=3 "$version/x86_64/bin/glslc"
"$target/bin/glslc" --version | head -1
