#!/usr/bin/env python3
import argparse
import os
import torch

try:
    from tools.quantize_rdt_weights_adv import quantize_state_dict
except Exception as e:
    print("[ERROR] failed to import quantize_state_dict from tools.quantize_rdt_weights_adv:", e)
    raise

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-path", required=True, help="input checkpoint path")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--bits-range", default="16-2", help="e.g. '16-2' or '16,8,4'")
    p.add_argument("--per-channel", action="store_true")
    p.add_argument("--symmetric", action="store_true")
    p.add_argument("--store-ints", action="store_true")
    p.add_argument("--scope", default="all")
    return p.parse_args()

def parse_range(s):
    s = str(s).strip()
    if "-" in s and "," not in s:
        a,b = map(int, s.split("-", 1))
        step = -1 if a > b else 1
        return list(range(a, b + step, step))
    # comma separated list
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(x) for x in parts]

def main():
    args = parse_args()
    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"input checkpoint not found: {args.in_path}")
    ckpt = torch.load(args.in_path, map_location="cpu")
    sd = ckpt.get("module", ckpt) if isinstance(ckpt, dict) else ckpt
    bits_list = parse_range(args.bits_range)
    os.makedirs(args.out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.in_path))[0]
    for bits in bits_list:
        if bits < 2 or bits > 16:
            print(f"[WARN] skipping invalid bits={bits}")
            continue
        print(f"[RUN] bits={bits}")
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
        )
        out_path = os.path.join(args.out_dir, f"{base_name}__q{bits}.pt")
        if isinstance(ckpt, dict) and "module" in ckpt:
            ckpt["module"] = new_sd
            torch.save(ckpt, out_path)
        else:
            torch.save(new_sd, out_path)
        print(f"[OK] saved => {out_path}")

if __name__ == "__main__":
    main()