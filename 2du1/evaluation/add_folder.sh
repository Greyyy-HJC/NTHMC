#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash add_folder.sh <tag>" >&2
    exit 1
fi

tag="$1"

if [[ -z "$tag" || "$tag" == "." || "$tag" == ".." || "$tag" == */* ]]; then
    echo "Invalid tag: $tag" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_dir="${script_dir}/${tag}"

if [ -e "$target_dir" ]; then
    echo "Directory already exists: $target_dir" >&2
    exit 1
fi

mkdir -p \
    "${target_dir}/dumps" \
    "${target_dir}/logs" \
    "${target_dir}/plots" \
    "${target_dir}/scripts"

touch \
    "${target_dir}/dumps/.gitkeep" \
    "${target_dir}/logs/.gitkeep" \
    "${target_dir}/plots/.gitkeep" \
    "${target_dir}/scripts/.gitkeep"

echo "Created evaluation folder: $target_dir"
