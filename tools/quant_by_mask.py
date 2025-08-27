#!/usr/bin/env python3
import sys
import os
import json
import torch
import argparse

# ensure repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.quantize_rdt_weights_adv import quantize_state_dict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-path", dest="in_path", required=True, help="input checkpoint path")
    p.add_argument("--groups-json", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bits-range", default="16-2")
    p.add_argument("--per-channel", action="store_true")
    p.add_argument("--symmetric", action="store_true")
    p.add_argument("--store-ints", action="store_true")
    p.add_argument("--scope", default="all")
    return p.parse_args()

def parse_bits_range(s):
    if "-" in s:
        a,b = map(int,s.split("-"))
        step = -1 if a>b else 1
        return list(range(a, b+step, step))
    return [int(x) for x in s.split(",")]

def main():
    args = parse_args()
    ckpt = torch.load(args.in_path, map_location="cpu")
    sd = ckpt.get("module", ckpt) if isinstance(ckpt, dict) else ckpt
    with open(args.groups_json, "r") as f:
        groups = json.load(f)
    keys = set(groups.get(args.group, []))
    bits_list = parse_bits_range(args.bits_range)
    os.makedirs(args.out_dir, exist_ok=True)

    for bits in bits_list:
        print("Quantizing group", args.group, "bits", bits)
        new_sd = quantize_state_dict(
            sd,
            bits=bits,
            symmetric=args.symmetric,
            per_channel=args.per_channel,
            clip_percentile=1.0,
            scope=args.scope,
            verbose=True,
            report_writer=None,
            store_ints=args.store_ints,
            groups_keys=keys,
        )
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(args.in_path))[0]}__{args.group}__q{bits}.pt")
        if isinstance(ckpt, dict) and "module" in ckpt:
            ckpt["module"] = new_sd
            torch.save(ckpt, out_path)
        else:
            torch.save(new_sd, out_path)
        print("Saved", out_path)

if __name__ == "__main__":
    main()