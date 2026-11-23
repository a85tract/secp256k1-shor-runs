#!/bin/bash
# Fetch the borrowed point-addition core: odinfree's entry on the public ECDSA.Fail leaderboard (main track,
# submission 7571412f), from the public repository at the exact commit we used. Nothing of that repository is
# redistributed here. The script
#   1. fetches the commit and unpacks its tree into DIR/src,
#   2. checks the sha256 of every file against upstream/SHA256SUMS,
#   3. with --build: builds the entry's generator (Rust; the repository pins rustc 1.93.0 in rust-toolchain), runs it,
#      and checks the sha256 of the raw gate stream bind81-builder.raw that our circuit checker and replays read.
# usage: upstream/fetch_core.sh [--build] [DIR]          (default DIR: upstream/build)
# Afterwards pass DIR/run/bind81-builder.raw as RAW to the scripts of src/ (or export O2_OPS_RAW=<that path>).
set -euo pipefail
REPO=https://github.com/Layr-Labs/ecdsafail-challenge.git
COMMIT=21356d25cae60e32c5bb08b49fc2112c8556a9c6        # branch submissions/7571412f-e901-42c1-a5e9-ed9aa6dd9a58
RAW_SHA256=fb14f5d55f8abead1f6a2a729050e78f0b0a3f4f74e53a45efd6ac9a600830f8   # 12,368,253 ops, 692,622,168 bytes
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=0; [ "${1:-}" = "--build" ] && { BUILD=1; shift; }
DIR=$(mkdir -p "${1:-$HERE/build}" && cd "${1:-$HERE/build}" && pwd)

if [ ! -d "$DIR/src" ]; then
  rm -rf "$DIR/git"; git init -q "$DIR/git"
  git -C "$DIR/git" fetch -q --depth 1 "$REPO" "$COMMIT"
  mkdir -p "$DIR/src"; git -C "$DIR/git" archive "$COMMIT" | tar -x -C "$DIR/src"
fi
(cd "$DIR/src" && sha256sum --quiet -c "$HERE/SHA256SUMS") && echo "source: all $(wc -l < "$HERE/SHA256SUMS") files match SHA256SUMS"
[ $BUILD = 1 ] || exit 0

# The generator is deterministic; thin LTO is switched off only so that it links within 2 GB.
(cd "$DIR/src" && CARGO_PROFILE_RELEASE_LTO=off CARGO_TARGET_DIR="$DIR/target" cargo build --release)
# build_circuit writes the raw builder stream before its post-passes. Under a 2 GB limit it then stops in those
# passes, which leaves bind81-builder.raw complete; the post-passed stream is not used.
mkdir -p "$DIR/run"
(cd "$DIR/run" && ulimit -v 2000000 && "$DIR/target/release/build_circuit" < /dev/null > build.log 2>&1) || true
echo "$RAW_SHA256  $DIR/run/bind81-builder.raw" | sha256sum -c
