#!/usr/bin/env python3
import json
import os
import re
import argparse
from collections import defaultdict

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def save(d, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def main():
    p = argparse.ArgumentParser(description="Refine key groups into semantic subgroups (per-block, per-module, stages).")
    p.add_argument("in_json", help="input key groups json")
    p.add_argument("out_json", nargs="?", default=None, help="output refined groups json (optional)")
    args = p.parse_args()

    if args.out_json is None:
        base = os.path.splitext(os.path.basename(args.in_json))[0]
        args.out_json = f"{base}_refined_semantic.json"
        print(f"[INFO] out_json not provided, using default: {args.out_json}")

    groups = load(args.in_json)
    all_keys = []
    for v in groups.values():
        all_keys += v

    refined = defaultdict(list)

    # patterns
    block_re = re.compile(r"model\.blocks\.(\d+)\.")
    attn_qkv_re = re.compile(r"\.attn\.qkv")
    attn_proj_re = re.compile(r"\.attn\.proj")
    cross_q_re = re.compile(r"\.cross_attn\.q")
    cross_kv_re = re.compile(r"\.cross_attn\.kv")
    ffn_fc1_re = re.compile(r"\.ffn\.fc1")
    ffn_fc2_re = re.compile(r"\.ffn\.fc2")
    norm_re = re.compile(r"\.norm")
    embed_re = re.compile(r"embed|pos_embed|adaptor|adapt", re.I)

    block_indices = set()

    for k in sorted(set(all_keys)):
        low = k
        m = block_re.search(low)
        blk = int(m.group(1)) if m else None
        if blk is not None:
            block_indices.add(blk)

        # per-block attn qkv
        if attn_qkv_re.search(low):
            if blk is not None:
                refined[f"attn_qkv_block_{blk}"].append(k)
            refined["attn_qkv_all"].append(k)
            continue

        # per-block attn proj
        if attn_proj_re.search(low):
            if blk is not None:
                refined[f"attn_proj_block_{blk}"].append(k)
            refined["attn_proj_all"].append(k)
            continue

        # cross attn q / kv
        if cross_q_re.search(low):
            if blk is not None:
                refined[f"cross_q_block_{blk}"].append(k)
            refined["cross_q_all"].append(k)
            continue
        if cross_kv_re.search(low):
            if blk is not None:
                refined[f"cross_kv_block_{blk}"].append(k)
            refined["cross_kv_all"].append(k)
            continue

        # ffn
        if ffn_fc1_re.search(low):
            if blk is not None:
                refined[f"ffn_fc1_block_{blk}"].append(k)
            refined["ffn_fc1_all"].append(k)
            continue
        if ffn_fc2_re.search(low):
            if blk is not None:
                refined[f"ffn_fc2_block_{blk}"].append(k)
            refined["ffn_fc2_all"].append(k)
            continue

        # norms
        if norm_re.search(low):
            refined["norms_all"].append(k)
            continue

        # embeddings / adaptors
        if embed_re.search(low):
            refined["embeddings_and_adaptors"].append(k)
            continue

        # text-specific small
        if "t_embedder" in low or "freq_embedder" in low or "lang" in low and "embed" in low:
            refined["text_small"].append(k)
            continue

        # fallback
        refined["other_small"].append(k)

    # create stage groups early/mid/late based on block count
    if block_indices:
        blocks = sorted(block_indices)
        n = len(blocks)
        # partition into 3 roughly equal parts
        third = max(1, n // 3)
        early_set = set(blocks[:third])
        mid_set = set(blocks[third:2*third])
        late_set = set(blocks[2*third:])

        for blk in blocks:
            # move per-block entries also into stage groups
            for pat_prefix in ("attn_qkv_block_", "attn_proj_block_", "cross_q_block_", "cross_kv_block_", "ffn_fc1_block_", "ffn_fc2_block_"):
                key_name = f"{pat_prefix}{blk}"
                if key_name in refined:
                    if blk in early_set:
                        refined[f"stage_early"].extend(refined[key_name])
                    elif blk in mid_set:
                        refined[f"stage_mid"].extend(refined[key_name])
                    else:
                        refined[f"stage_late"].extend(refined[key_name])

    # dedupe and sort lists
    out = {k: sorted(list(dict.fromkeys(v))) for k, v in refined.items() if v}
    save(out, args.out_json)
    print("Wrote refined semantic groups to", args.out_json)

if __name__ == "__main__":
    main()