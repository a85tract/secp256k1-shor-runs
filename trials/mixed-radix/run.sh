#!/bin/bash
# The registered trial run of the mixed-radix configuration with a full last window (110 runs, seed n7fix),
# with the frozen code in code/.
# usage: ./run.sh [SIZES] [TRIALS] [TRIALS256] [OUT] [extra run.py arguments]
# Defaults are the registered run: sizes 16,24,32,64,128,256, 200 trials per smaller size, 563 at 256 bits.
# Each trial's randomness depends only on (seed, size, trial index), so any subset of trials reproduces the log.
cd "$(dirname "$0")"
SIZES=${1:-16,24,32,64,128,256}; TR=${2:-200}; TR256=${3:-563}; OUT=${4:-out/mixed_radix}; shift $(( $# < 4 ? $# : 4 ))
mkdir -p "$(dirname "$OUT")"; I=inputs; TOY=""
for k in 1 2 3 4; do TOY="$TOY --kind w$k=declared:1:1:0:$k"; done
for w in 2 3 6; do TOY="$TOY --kind n$w=declared:1:1:0:n$w --kind m$w=declared:1:1:0:m$w"; done
PYTHONDONTWRITEBYTECODE=1 python3 -B code/harness/run.py code/stack_n7_l9.py --sizes $SIZES \
  --trials $TR --trials256 $TR256 --approx --seed n7fix -p RUNS=110 \
  --global-mc 256=$I/pmc_lead_n7s288_cap_P131072_s271.json,$I/pmc_lead_n7s288_cap_P131072_s272.json,$I/pmc_lead_n7s288_cap_P131072_s273.json \
  --global-mc 16=$I/pmc_n7toy16_cap_P65536_s51.json --global-mc 24=$I/pmc_n7toy24_cap_P65536_s52.json \
  --global-mc 32=$I/pmc_n7toy32_cap_P65536_s53.json --global-mc 64=$I/pmc_n7toy64_cap_P65536_s54.json \
  --global-mc 128=$I/pmc_n7toy128_cap_P65536_s55.json --first-lookup $I/first_ss_a17_p512.json \
  --kind term=$I/hexlead_w9_peak_cap_n256.json --kind w8=$I/hexlead_w8_peak_cap_n256.json \
  --kind n7=$I/n7lead_n7_peak_cap_n256.json --kind m7=$I/n7lead_m7_peak_cap_n256.json $TOY \
  --json $OUT.json --trial-log ${OUT}_trials.jsonl --progress 10 "$@" > $OUT.txt
