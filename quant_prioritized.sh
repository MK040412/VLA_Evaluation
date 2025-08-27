#!/usr/bin/env bash
set -euo pipefail
CKPT="pretrained_models/rdt/mp_rank_00_model_states.pt"
OUT_DIR="quantized_checkpoints_6bit"
BITS=6
mkdir -p "$OUT_DIR"

echo '[Q] single embeddings_and_adaptors'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "embeddings_and_adaptors" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single attn_qkv_block_0'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_qkv_block_0" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single attn_qkv_block_9'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_qkv_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single attn_qkv_block_18'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_qkv_block_18" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single attn_proj_block_0'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_proj_block_0" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single attn_proj_block_9'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_proj_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single attn_proj_block_18'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_proj_block_18" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single ffn_fc1_block_0'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "ffn_fc1_block_0" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single ffn_fc1_block_9'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "ffn_fc1_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single ffn_fc1_block_18'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "ffn_fc1_block_18" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single cross_q_all'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "cross_q_all" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] single stage_early'
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "stage_early" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair embeddings_and_adaptors + attn_qkv_block_0'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "embeddings_and_adaptors" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__embeddings_and_adaptors__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_qkv_block_0" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair embeddings_and_adaptors + stage_early'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "embeddings_and_adaptors" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__embeddings_and_adaptors__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "stage_early" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair attn_qkv_block_9 + ffn_fc1_block_9'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_qkv_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__attn_qkv_block_9__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "ffn_fc1_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair attn_qkv_block_18 + ffn_fc1_block_18'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_qkv_block_18" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__attn_qkv_block_18__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "ffn_fc1_block_18" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair attn_proj_block_9 + ffn_fc1_block_9'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_proj_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__attn_proj_block_9__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "ffn_fc1_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair cross_q_all + ffn_fc1_block_9'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "cross_q_all" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__cross_q_all__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "ffn_fc1_block_9" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair attn_qkv_block_0 + attn_proj_block_0'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_qkv_block_0" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__attn_qkv_block_0__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "attn_proj_block_0" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

echo '[Q] pair stage_early + stage_late'
BASE=$(basename "$CKPT")
python tools/quant_by_mask.py --in-path "$CKPT" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "stage_early" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints
python tools/quant_by_mask.py --in-path "$OUT_DIR/${BASE}__stage_early__q6.pt" --groups-json "mp_rank_00_model_states_refined_semantic.json" --group "stage_late" --out-dir "$OUT_DIR" --bits-range 6 --per-channel --symmetric --store-ints

