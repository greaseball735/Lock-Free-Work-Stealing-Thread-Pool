#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

make -j

export BENCH_THREADS="${BENCH_THREADS:-1,2,4,8}"
export BENCH_IMPLS="${BENCH_IMPLS:-sequential,mutex,custom,openmp}"

./job_bench all > results.csv

echo "CSV: $ROOT/results.csv"
