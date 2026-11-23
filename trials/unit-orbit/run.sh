#!/bin/bash
# The registered trial run of the unit-orbit configuration (79 runs, seed d270fix), with the frozen code in code/.
# usage: ./run.sh [SIZES] [TRIALS] [TRIALS256] [OUT] [extra run.py arguments]
# Defaults are the registered run: sizes 16,24,32,64,128,256, 200 trials per smaller size, 563 at 256 bits.
# Each trial's randomness depends only on (seed, size, trial index), so any subset of trials reproduces the log.
cd "$(dirname "$0")"
SIZES=${1:-16,24,32,64,128,256}; TR=${2:-200}; TR256=${3:-563}; OUT=${4:-out/unit_orbit}; shift $(( $# < 4 ? $# : 4 ))
mkdir -p "$(dirname "$OUT")"; I=inputs
PYTHONDONTWRITEBYTECODE=1 python3 -B code/harness/run.py code/stack_d270_4w.py --sizes $SIZES --trials $TR \
  --trials256 $TR256 --approx --seed d270fix -p RUNS=79 -p SPLIT=1 -p L=11 -p NEED=28 \
  -p "HEX=[9,8,8,8,8,8,8,8,8,8,9,9,9,9,9,3]" \
  --global-mc 256=$I/pmc_d270_cap_P131072_s11.json,$I/pmc_d270_cap_P131072_s12.json \
  --global-mc 16=$I/pmc_toy16_cap_P65536_s21.json --global-mc 24=$I/pmc_toy24_cap_P65536_s22.json \
  --global-mc 32=$I/pmc_toy32_cap_P65536_s23.json --global-mc 64=$I/pmc_toy64_cap_P65536_s24.json \
  --global-mc 128=$I/pmc_toy128_cap_P65536_s25.json --first-lookup $I/first_ss_a18_p512.json \
  --kind w8=$I/hexlead_w8_peak_cap_n256.json --kind w9=$I/hexlead_w9_peak_cap_n256.json \
  --kind term=$I/hexlead_w9_peak_cap_n256.json --json $OUT.json --trial-log ${OUT}_trials.jsonl "$@" > $OUT.txt
