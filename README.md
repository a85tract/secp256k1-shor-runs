# How Small Can a Single Shor Run Be? Code and data

This repository holds the code and data behind the paper "How Small Can a Single Shor Run Be? Limits and Constructions
for secp256k1" (Y. Guang, Y. Xiong, Y. Chen, X. Gao).

## What the paper claims

A run is one coherent quantum computation, from its first gate until all of its qubits are measured. Its volume is its
peak number of logical qubits times its Toffoli count. The paper asks how small B, the volume of the largest run, can
get when every key must be found with probability at least 0.99, the total volume T of all runs stays within ten times
the baseline's (three runs of Häner et al., B0 = 1,457 qubits x 39,255,168 Toffoli gates, T0 = 3 B0), and the classical
post-processing stays within 2^40 operations.

- Limit (Section III). One linear run reveals at most log2(1 + (|S| - 1)/r) bits about a uniform key. With generic
  post-processing and the baseline's window costs, splitting alone shrinks the largest run by at most 1.86 times inside
  the budget (55 runs of 258.98 exponent bits), and adaptive runs by at most 1.93 times.
- Construction (Section V). Unit-orbit windows store one table point per orbit of the curve's six units; mixed-radix
  windows add a seven-valued digit; the root-sign last addition measures the final x-coordinate and picks y from a
  slope check. Every addition runs on a point-addition core borrowed from the ECDSA.Fail leaderboard. One random scalar
  and one random start per run give every addition uniform operands, so a lemma on each run's outcomes and a theorem on
  the decoder certify success for every key although the additions are approximate.
- Results (Section VII). All three configurations solved all 563 of their pre-registered trials on secp256k1:

| Configuration | Runs | Toffoli per run | Qubits | B / B0 | T / T0 |
|---|---|---|---|---|---|
| Unit-orbit windows, full last window | 79 | 12,626,051 | 1,246 | 0.2751 (3.64x smaller) | 7.24 |
| Mixed-radix windows, full last window | 110 | 12,264,316 | 1,246 | 0.2672 (3.74x smaller) | 9.80 |
| Mixed-radix windows, root-sign last addition | 102 | 11,866,224 | 1,246 | 0.2585 (3.87x smaller) | 8.79 |

## Requirements

- Linux or macOS with bash, Python 3.10 or newer, `pip install -r requirements.txt` (numpy; matplotlib for the figures).
  The trials and the quick check need numpy only.
- To rebuild the circuits and the Monte Carlo inputs from scratch: git, a Rust toolchain (rustup; the borrowed entry pins
  rustc 1.93.0), and about 2 GB of memory per process. See "The borrowed core" below.

## Quick check

    python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
    python3 quickcheck.py

It prints PASS or FAIL for 32 items: the floors of Table I and of the budget frontier, the Toffoli counts of Table II,
the per-run totals and volumes of Table III for every configuration and baseline, the gain factors of Section VII.B,
the certificate constants of App. B.5, the rows of Table VII at 16, 24 and 32 bits (all 200 registered trials per size,
rerun with the frozen code and compared record by record with the saved trial logs), and the first two registered
trials at 256 bits. It takes about 5 minutes on 3 cores and writes its outputs to `quickcheck_out/`.

## Layout

| Path | Contents |
|---|---|
| `quickcheck.py` | the quick check above |
| `trials/unit-orbit/`, `trials/mixed-radix/`, `trials/root-sign/` | one pre-registered trial run each: `code/` is the harness and decoder exactly as frozen at the commit that registered the run (run count, trial count and seed), `inputs/` the counted circuits and Monte Carlo constants it read, `results/` its report, summary and per-trial log, `run.sh` the registered command |
| `src/harness/` | the end-to-end harness (exact outcome sampler, keep rule of the certificate, lattice decoder, classical-work accounting) and the reversible-circuit checker `vb.py`, as of the final version; `src/harness/tests/` its tests |
| `src/h1/`, `src/o1/`, `src/o2/`, `src/o3/`, `src/s2/`, `src/s4/` | the window circuits, first lookups, digit registers, root-sign last addition, the port of the borrowed core, the bit-sliced replay of its gate stream and the path Monte Carlo of each configuration, with their counted results (JSON). These directory names are the original layout; the scripts find each other by relative path, so they were not renamed |
| `src/s3/theory/` | the numbers of Section III (Table I, QFT information, hash runs) |
| `src/base/` | the unsplit baselines, Ekerå's tradeoff on the same core, the budget frontier and the chain of gain factors, with every output |
| `src/paper/figures/` | the scripts of Figures 1 to 5 and the figures they produce |
| `upstream/` | `fetch_core.sh` and `SHA256SUMS` for the borrowed core (nothing of it is stored here) |

File names keep a few working names: `hex` means unit-orbit, `n7` and `m7` the two mixed-radix windows over (3+ω)2^7
and (2-ω)2^7, `term` or `hexterm` the root-sign last addition, `pmc` a path Monte Carlo, "Lemma G" the paper's Lemma 1
with the run-wide scalar, "V-B" the reversible-circuit checker and "V-A" an end-to-end trial run.

## From the paper to the commands

Run the commands from the repository root unless a directory is given; times are as measured on our machine, one core per process.

| Paper item | Command | Output | Time |
|---|---|---|---|
| Table I (floors), Sec. III.C numbers | `cd src && python3 s3/theory/numbers_n256.py` and `python3 paper/figures/floors.py` | printed | seconds |
| Floors inside 10 T0 and 4 T0, adaptive floor, floor at the core's cost (1.86, 1.82, 1.93, 1.87 times) | `cd src && python3 base/frontier.py` | `base/frontier.txt`, `base/frontier.csv` | seconds |
| Fig. 3 and Sec. III.D (QFT information, zero padding) | `cd src && python3 s3/theory/oversample_info.py R A B PADS ENV_A ENV_B` (cases in `s3/theory/oversample_info.txt`) | printed | minutes per case |
| Table V (hash runs, App. A) | `cd src && python3 s3/theory/hash_check.py` | `s3/theory/hash_check_test1.txt` | minutes |
| Table II and Table VI (Toffoli and qubits of each kind of addition and of each run) | `python3 quickcheck.py` (sums the counted circuits in `trials/*/inputs/`); every run's total is also printed by its trial report, `trials/*/results/*.txt` ("cost at n = 256") | printed | 5 min |
| Counted windows at 256 bits (V-B), unit-orbit W = 8 and 9 | `cd src && O2_OPS_RAW=RAW python3 h1/hexcap/vbx2cap.py h1/hexcap/hexlead_cap.py -p YSEL=2 -p W=8 -p HOLD=peak --n 256 --lanes 1024 --batches 1 --json OUT.json`, then `python3 h1/hexcap/patch_cap_json.py OUT.json 8` | `h1/hexcap/hexlead_w{8,9}_peak_cap_n256.json` | 2-3 min |
| Counted mixed-radix windows | `cd src && O2_OPS_RAW=RAW_PC32 O2_SITEFILE=o2/s7/sites_pc32.json O2_SITEMAP=o2/s7/sitemap_pc32.json python3 h1/n7/vbx2capn7m7.py o1/n7lead.py -p KIND=m7_window_add -p W=7 -p HOLD=peak -p STREAM_CAP=727200 --n 256 --lanes 1024 --batches 1 --json OUT.json`, then `python3 o2/s7/patch_pc_json.py OUT.json h1/n7/n7lead_m7_peak_cap_n256.json` | `h1/n7/n7lead_{n7,m7}_peak_cap_n256.json` | 3 min |
| Counted root-sign last addition | `cd src && O2_OPS_RAW=RAW python3 s4/term/vbxterm.py s4/term/hexterm.py --n 256 --lanes 1024 --batches 1 --json OUT.json`, then `python3 s4/term/patch_json.py OUT.json 9` | `s4/results/term/hexterm_w9_n256.json` | 5 min |
| First lookups (17 and 18 bits), digit registers | `cd src && python3 harness/vb.py o1/first_ss.py --n 17 -p P=512 --lanes 64 --batches 1 --json OUT.json`; `python3 harness/vb.py o1/hexdigit_check.py --n 7 -p MODE=identity`; `python3 harness/vb.py o1/s5/n7digit_check.py --n 7 -p BAR=True -p MODE=identity` | `h1/n7/first_ss_a17_p512.json`, `h1/results/first_ss_a18_p512.json`, printed | minutes |
| Seven-level register circuits (App. D) | `cd src && python3 h1/n7/n7reg_circuits.py` | printed | minutes |
| Signed window on the core (Table II row 2), U1 window sweep | `cd src && O2_OPS_RAW=RAW python3 o2/port/vbx2.py o2/port/windowed_lead.py --n 256 --lanes 2048 --batches 2 --json OUT.json`; `python3 base/u1_sweep.py` | `o2/results4/lead_w16_n256_flags.json`, `base/lead_w16_cap.json` | 7 min |
| Certificate constants, App. B.5 (path Monte Carlo, 131,072 paths per batch) | unit-orbit: `cd src && python3 h1/hexcap/path_mc_cap.py RAW 131072 9,8,8,8,8,8,8,8,8,8,9,9,9,9,9 SEED OUT.json` (seeds 11, 12); mixed-radix: `python3 o2/s7/path_mc_n7_run.py RAW 131072 h9,n7,m7x10,h8x3,s4 SEED OUT.json 726000` (seeds 271-273); root-sign: `python3 o1/s7/path_mc_term2.py RAW 131072 h9,n7,m7x10,h8x3,s4 SEED OUT.json` (seeds 61-64); smaller sizes: `h1/hexcap/path_mc_cap.py`, `o3/s6/path_mc_n7.py`, `o1/s6/path_mc_mix.py` with 65,536 paths (layouts and seeds in the `inputs/` file names and their JSON fields) | `trials/*/inputs/pmc_*.json` | 8-15 min per batch |
| Constants as the harness reads them (1 - g0, path term, phi_LB, e0) | printed at the top of every trial report, `trials/*/results/*.txt` ("Lemma G constants at n = 256") | printed | 1 min |
| Lemma 1 toy check (App. B.4) | `cd src && python3 o3/tests/toy_global_scalar.py` | printed | minutes |
| Decoder and pre-registered trials, Table VII and Sec. VII.A/D, every size | `trials/unit-orbit/run.sh`, `trials/mixed-radix/run.sh`, `trials/root-sign/run.sh` (arguments: sizes, trials per smaller size, trials at 256 bits, output prefix) | `out/` in the trial directory; compare with `results/` | about 2 h, 3.5 h and 3 h for the full runs; seconds per trial below 256 bits |
| Harness tests (sampler against state vectors, keep rule, attempt caps) | `cd src/harness && for t in tests/test_*.py; do python3 $t; done` | printed | minutes |
| Table III unsplit rows, Ekerå's tradeoff on the core, gain factors of Sec. VII.B | `cd src && bash base/run_all.sh` (steps: sweep first u1 u1single t512 u2price u2decode ekscan ekpick ekfinal chain frontier) | `base/*.txt`, `base/*.json`, `base/chain.txt` | about 45 min |
| Figures 1-5 | `cd src && for f in paper/figures/fig*.py; do python3 $f; done` | `src/paper/figures/*.pdf` | 1 min |

`RAW` is the borrowed core's gate stream built by `upstream/fetch_core.sh --build` (see below). The two mixed-radix
window kinds were counted on a variant of this stream (`RAW_PC32`, stream cap 727,200) that a locally modified copy of
the entry's generator built with different comparison widths; that build is not reproduced by this repository, and
its counts are kept in `h1/n7/n7lead_{n7,m7}_peak_cap_n256.json`. The Monte Carlo constants of every configuration use
the unmodified stream.

What is not stored: the raw outputs of the path Monte Carlo (one `_paths.npz` of up to tens of MB per batch) and of
the replay. Their JSON summaries, which are what the harness reads, are in `trials/*/inputs/`; the commands above and
the seeds in the file names regenerate the raw files in 8 to 15 minutes per batch.

## The borrowed core

Every point addition runs on the leading entry of the main track of the public ECDSA.Fail leaderboard
(https://ecdsa.fail), submitted by odinfree (submission 7571412f, 1,227 qubits and 722,917 Toffoli gates on average).
Its source lives in https://github.com/Layr-Labs/ecdsafail-challenge at commit
21356d25cae60e32c5bb08b49fc2112c8556a9c6 (branch `submissions/7571412f-e901-42c1-a5e9-ed9aa6dd9a58`). No file of that
repository and nothing generated from it is included here. `upstream/fetch_core.sh` fetches that commit and checks every
file against `upstream/SHA256SUMS`; `upstream/fetch_core.sh --build` also builds the entry's generator, runs it and
checks that the raw gate stream `bind81-builder.raw` has sha256
fb14f5d55f8abead1f6a2a729050e78f0b0a3f4f74e53a45efd6ac9a600830f8, the stream every counted circuit and Monte Carlo here
was measured on. Our port loads each window's table point at the five sites where the entry's classical addend enters
its gate stream (paper Sec. V.C and App. D). We thank odinfree for making the entry public.

## Notes

- The code under `trials/*/code/` is the code of each registered run. Comments and docstrings were edited to remove
  internal notes; a few printed strings changed with them. Every trial's randomness depends only on the seed, the curve
  size and the trial's index, so any subset of trials reproduces the saved log exactly, as the quick check shows.
- Below 256 bits the additions are stand-ins with the failure constants of ordinary windows, as in the paper.
