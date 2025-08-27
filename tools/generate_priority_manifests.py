#!/usr/bin/env python3
import json, os, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--priority-json", default="tools/priority_groups.json")
    p.add_argument("--refined-json", default="mp_rank_00_model_states_refined_semantic.json")
    p.add_argument("--ckpt", default="pretrained_models/rdt/mp_rank_00_model_states.pt")
    p.add_argument("--bits", type=int, default=6)
    p.add_argument("--out-dir", default="quant_out/prioritized")
    p.add_argument("--quant-manifest", default="quant_prioritized.sh")
    p.add_argument("--eval-manifest", default="eval_prioritized.sh")
    args = p.parse_args()

    pr = json.load(open(args.priority_json))
    groups = pr["groups"]
    pairs = pr.get("pairs", [])
    os.makedirs(args.out_dir, exist_ok=True)

    # quant manifest
    with open(args.quant_manifest, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\nCKPT=\"%s\"\nOUT_DIR=\"%s\"\nBITS=%d\nmkdir -p \"$OUT_DIR\"\n\n" % (args.ckpt, args.out_dir, args.bits))
        for g in groups:
            f.write("echo '[Q] single %s'\n" % g)
            f.write("python tools/quant_by_mask.py --in-path \"$CKPT\" --groups-json \"%s\" --group \"%s\" --out-dir \"$OUT_DIR\" --bits-range %d --per-channel --symmetric --store-ints\n\n" % (args.refined_json, g, args.bits))
        for a,b in pairs:
            f.write("echo '[Q] pair %s + %s'\n" % (a,b))
            f.write("BASE=$(basename \"$CKPT\")\n")
            f.write("python tools/quant_by_mask.py --in-path \"$CKPT\" --groups-json \"%s\" --group \"%s\" --out-dir \"$OUT_DIR\" --bits-range %d --per-channel --symmetric --store-ints\n" % (args.refined_json, a, args.bits))
            f.write("python tools/quant_by_mask.py --in-path \"$OUT_DIR/${BASE}__%s__q%d.pt\" --groups-json \"%s\" --group \"%s\" --out-dir \"$OUT_DIR\" --bits-range %d --per-channel --symmetric --store-ints\n\n" % (a, args.bits, args.refined_json, b, args.bits))
    os.chmod(args.quant_manifest, 0o755)

    # eval manifest (edit ENV2EMB map in file if needed)
    with open(args.eval_manifest, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\nREPO_ROOT=\"$(cd $(dirname \"$0\")/.. && pwd)\"\ncd \"$REPO_ROOT\"\nOUT_DIR=\"%s\"\nRESULT_ROOT=\"results/prioritized_6bit_$(date +%%Y%%m%%d_%%H%%M)\"\nmkdir -p \"$RESULT_ROOT/videos\" \"$RESULT_ROOT/logs\"\n\ndeclare -A ENV2EMB=(\n  [\"PickCube-v1\"]=\"lang_embeds/text_embed_PickCube-v1.pt\"\n  [\"PushCube-v1\"]=\"lang_embeds/text_embed_PushCube-v1.pt\"\n)\nENVS=(\"PickCube-v1\" \"PushCube-v1\")\nNUM=25\nexport PYTHONPATH=\"${PYTHONPATH:-}:$REPO_ROOT\"\n\nfind \"$OUT_DIR\" -type f -name \"*__q%d.pt\" | while read CKPT; do\n  bname=$(basename \"$CKPT\" .pt)\n  for ENV in \"${ENVS[@]}\"; do\n    LANG=\"${ENV2EMB[$ENV]}\"\n    if [[ ! -f \"$LANG\" ]]; then echo \"[SKIP] no lang embed for $ENV -> $LANG\"; continue; fi\n    VID_DIR=\"$RESULT_ROOT/videos/${ENV}/${bname}\"\n    LOG=\"$RESULT_ROOT/logs/${ENV}__${bname}.log\"\n    mkdir -p \"$VID_DIR\" \"$(dirname \"$LOG\")\"\n    echo \"[RUN] $bname on $ENV\"\n    python -m scripts.eval_qunat_rdt_maniskill \\\n      --pretrained_path \"$CKPT\" \\\n      --env-id \"$ENV\" \\\n      --obs-mode rgb \\\n      --num-traj \"$NUM\" \\\n      --render-mode rgb_array \\\n      --pretrained_text_encoder_name_or_path precomputed \\\n      --lang_embeddings_path \"$LANG\" \\\n      --dtype fp16 \\\n      --max-steps 400 \\\n      --action-downsample 1 \\\n      --save-video-dir \"$VID_DIR\" 2>&1 | tee \"$LOG\"\n  done\ndone\n\npython tools/summarize_eval_logs.py --logs \"$RESULT_ROOT/logs\" --out \"$RESULT_ROOT/summary.csv\"\necho \"[OK] summary -> $RESULT_ROOT/summary.csv\"\n" % (args.out_dir, args.bits))
    os.chmod(args.eval_manifest, 0o755)

    print("[OK] wrote manifests:", args.quant_manifest, args.eval_manifest)

if __name__ == "__main__":
    main()