#!/usr/bin/env python3
import sys, os, json, re
import torch

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="checkpoint path")
    p.add_argument("--out", default=None, help="output json path")
    args = p.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("module", ckpt) if isinstance(ckpt, dict) else ckpt

    patterns = {
      "vision": [r"vision", r"siglip", r"image", r"vision_model"],
      "text": [r"t5", r"text", r"embedder", r"token", r"text_model"],
      "policy_attn": [r"attn", r"q_proj", r"k_proj", r"v_proj", r"out_proj"],
      "policy_ffn": [r"mlp", r"fc1", r"fc2", r"w1", r"w2", r"ffn"],
      "embeddings": [r"embed", r"pos_embed", r"token_embed"],
      "heads": [r"head", r"proj_out", r"action_head", r"value_head"],
    }

    groups = {k: [] for k in patterns.keys()}
    groups["other"] = []

    for key in sd.keys():
        low = key.lower()
        matched = False
        for g, pats in patterns.items():
            for p in pats:
                if re.search(p, low):
                    groups[g].append(key)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            groups["other"].append(key)

    out = args.out or (os.path.splitext(os.path.basename(args.ckpt))[0] + "_key_groups.json")
    with open(out, "w") as f:
        json.dump(groups, f, indent=2)
    print("Saved key groups to", out)

if __name__ == "__main__":
    main()