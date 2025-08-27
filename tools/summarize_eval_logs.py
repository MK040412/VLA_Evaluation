#!/usr/bin/env python3
import os
import re
import csv
import argparse
from glob import glob

TRIAL_RE = re.compile(r"Trial\s+(\d+)\s+finished,\s+success:\s+(?:tensor\(\[)?(True|False)(?:\]\))?,\s+steps:\s+(\d+)")
SR_RE = re.compile(r"Success rate:\s+([0-9.]+)%")
CKPT_RE = re.compile(r"--pretrained_path\s+(\S+)")

def parse_log(path):
    trials = []
    success_rate = None
    ckpt = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = TRIAL_RE.search(line)
            if m:
                t = int(m.group(1))
                s = True if m.group(2) == "True" else False
                steps = int(m.group(3))
                trials.append((t, s, steps))
                continue
            m = SR_RE.search(line)
            if m:
                success_rate = float(m.group(1))
                continue
            m = CKPT_RE.search(line)
            if m and ckpt is None:
                ckpt = os.path.basename(m.group(1))
    return ckpt, trials, success_rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="log dir (contains *.log)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.logs, "*.log")))
    rows = []
    for fp in files:
        ckpt, trials, sr = parse_log(fp)
        if not trials:
            continue
        total = len(trials)
        succ = sum(1 for _, s, _ in trials if s)
        avg_steps = sum(st for _, _, st in trials) / total
        ckpt = ckpt or os.path.basename(fp)
        # per trial flags
        trial_flags = "".join("1" if s else "0" for _, s, _ in sorted(trials))
        rows.append([ckpt, total, succ, round(100.0*succ/total, 2) if total else sr, round(avg_steps, 2), trial_flags])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "num_trials", "num_success", "success_rate(%)", "avg_steps", "per_trial_success_bits(1/0)"])
        w.writerows(rows)
    print(f"[OK] wrote summary -> {args.out}")

if __name__ == "__main__":
    main()
