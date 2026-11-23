#!/bin/bash
# most of it base/u2_decode.py and the Ekera scan). Run from the repository root:
# bash base/run_all.sh [step ...] (default: all steps).
set -e
export PYTHONDONTWRITEBYTECODE=1
ulimit -v 8000000
PY="taskset -c 4-7 nice -n 10 python3 -B"
H=h1/hexcap; N=h1/n7
KINDS="--kind term=s4/results/term/hexterm_w9_n256.json --kind w8=$H/hexlead_w8_peak_cap_n256.json
       --kind w9=$H/hexlead_w9_peak_cap_n256.json --kind n7=$N/n7lead_n7_peak_cap_n256.json
       --kind m7=$N/n7lead_m7_peak_cap_n256.json --first-lookup base/first_ss_a16_p512.json"
U1="harness/run.py base/u1_shor.py --measured base/lead_w16_cap.json"
EK="$PY harness/run.py base/ekera_ctrl.py --measured base/lead_w16_cap.json --sizes 256"
steps=${@:-sweep first u1 u1single t512 u2price u2decode ekscan ekpick ekfinal chain frontier}
for s in $steps; do case $s in
  sweep)    $PY base/u1_sweep.py > base/u1_sweep.txt ;;          # W sweep; writes base/lead_w16_cap.json
  first)    $PY harness/vb.py o1/first_ss.py --n 16 -p P=512 --lanes 64 --batches 1 \
              --json base/first_ss_a16_p512.json > base/first_ss_a16.txt ;;
  u1)       $PY $U1 --trials 200 --trials256 5000 --approx --seed u1R4 --json base/u1_R4.json > base/u1_R4.txt ;;
  u1single) $PY $U1 --sizes 256 --trials256 20000 -p ATTEMPTS=1 --seed u1single_ideal \
              --json base/u1_single_ideal.json > base/u1_single_ideal.txt
            $PY $U1 --sizes 256 --trials256 20000 --approx -p ATTEMPTS=1 --seed u1single_approx \
              --json base/u1_single_approx.json > base/u1_single_approx.txt ;;
  t512)     $PY $U1 --sizes 256 --trials256 5000 --approx -p NU_FRAC=None --seed u1t512 \
              --json base/u1_t512.json > base/u1_t512.txt ;;
  u2price)  for v in h n nt; do $PY harness/run.py base/u2_price.py --sizes 256 --trials256 1 -p VARIANT="'$v'" \
              $KINDS --seed u2p --json base/u2_price_$v.json > base/u2_price_$v.txt; done ;;
  u2decode) for v in nt h; do $PY base/u2_decode.py $v 5000 s1 > base/u2_decode_$v.txt; done ;;   # nt also serves n
  chain)    $PY base/chain.py > base/chain.txt ;;
  ekscan)   mkdir -p base/ekera; : > base/ekera_scan.txt          # one batch per solve: q(s) and the clean-run decoder
            for s in 1 2 3 4 5 6 7 8 10; do
              $EK --trials256 2000 --approx -p S=$s -p ATTEMPTS=1 --seed ekscan_s$s \
                --json base/ekera/scan_s$s.json >> base/ekera_scan.txt
              $EK --trials256 1000 -p S=$s -p ATTEMPTS=1 --seed ekideal_s$s \
                --json base/ekera/ideal_s$s.json >> base/ekera_scan.txt; done ;;
  ekpick)   $PY base/ekera_pick.py > base/ekera_pick.txt ;;
  ekfinal)  $EK --trials256 5000 --approx -p S=4 -p ATTEMPTS=13 --seed ekfinal \
              --json base/ekera_final.json > base/ekera_final.txt ;;   # s and batches (95 % rule): ekera_pick.txt
  frontier) $PY base/frontier.py > base/frontier.txt ;;
  *)        echo "unknown step $s"; exit 1 ;;
esac; done
