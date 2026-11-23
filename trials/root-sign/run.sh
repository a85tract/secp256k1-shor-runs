#!/bin/bash
# The registered trial run of the configuration with mixed-radix windows and the root-sign last addition
# (102 runs, seed n7termfix), with the frozen code in code/.
# usage: ./run.sh [SIZES] [TRIALS] [TRIALS256] [OUT] [extra run.py arguments]
# Defaults are the registered run: sizes 16,24,32,64,128,256, 200 trials per smaller size, 563 at 256 bits.
# Each trial's randomness depends only on (seed, size, trial index), so any subset of trials reproduces the log.
cd "$(dirname "$0")"
SIZES=${1:-16,24,32,64,128,256}; TR=${2:-200}; TR256=${3:-563}; OUT=${4:-out/root_sign}; shift $(( $# < 4 ? $# : 4 ))
mkdir -p "$(dirname "$OUT")"; I=inputs; TOY=""
for k in 1 2 3 4; do TOY="$TOY --kind w$k=declared:1:1:0:$k"; done
for w in 2 3 6; do TOY="$TOY --kind n$w=declared:1:1:0:n$w --kind m$w=declared:1:1:0:m$w"; done
MC=$I/pmc_term2_n7l9_P131072_s61.json,$I/pmc_term2_n7l9_P131072_s62.json,$I/pmc_term2_n7l9_P131072_s63.json,$I/pmc_term2_n7l9_P131072_s64.json
PYTHONDONTWRITEBYTECODE=1 python3 -B code/harness/run.py code/stack_n7_l9.py --sizes $SIZES \
  --trials $TR --trials256 $TR256 --approx --seed n7termfix -p RUNS=102 \
  --global-mc 256=$MC \
  --global-mc 16=$I/pmc_toy16_P65536_s21.json --global-mc 24=$I/pmc_toy24_P65536_s22.json \
  --global-mc 32=$I/pmc_toy32_P65536_s23.json --global-mc 64=$I/pmc_toy64_P65536_s24.json \
  --global-mc 128=$I/pmc_toy128_P65536_s25.json --first-lookup $I/first_ss_a17_p512.json \
  --kind term=$I/hexterm_w9_n256.json --kind w8=$I/hexlead_w8_peak_cap_n256.json \
  --kind n7=$I/n7lead_n7_peak_cap_n256.json --kind m7=$I/n7lead_m7_peak_cap_n256.json $TOY \
  --json $OUT.json --trial-log ${OUT}_trials.jsonl --progress 10 "$@" > $OUT.txt
