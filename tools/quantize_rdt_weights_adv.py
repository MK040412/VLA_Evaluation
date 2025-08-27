#!/usr/bin/env python3
import argparse
import os
import csv
from typing import Tuple, Dict, Any, Optional, List, Set

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="input checkpoint (.pt/.pth)")
    p.add_argument("--out", dest="out_path", required=True, help="output checkpoint path or out-dir when --bits-range")
    p.add_argument("--bits", type=int, required=False, help="quantization bits (2..16)")
    p.add_argument("--bits-range", type=str, default=None,
                   help="comma or dash range for multiple outputs, e.g. '16,8,4' or '16-2' to produce many files")
    p.add_argument("--method", type=str, default="uniform", choices=["uniform"], help="quant method")
    p.add_argument("--per-channel", action="store_true", help="per-channel quant (dim=0)")
    p.add_argument("--symmetric", action="store_true", help="use symmetric quantization")
    p.add_argument("--clip-percentile", type=float, default=1.0,
                   help="symmetric clipping percentile in (0,1]; 1.0=disable")
    p.add_argument("--scope", type=str, default="all", choices=["all", "attn", "ffn"],
                   help="filter target layers by name substring")
    p.add_argument("--dry-run", action="store_true", help="do not save, just print what would happen")
    p.add_argument("--report-csv", type=str, default=None, help="write per-layer error report")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--store-ints", action="store_true", help="store integer quantized tensors + metadata in checkpoint")
    p.add_argument("--groups-json", type=str, default=None, help="optional groups json to restrict which keys to quantize")
    p.add_argument("--group", type=str, default=None, help="when --groups-json provided, quantize only this group")
    return p.parse_args()


def _match_scope(name: str, scope: str) -> bool:
    n = name.lower()
    if scope == "all":
        return True
    if scope == "attn":
        return ("attn" in n) or ("attention" in n) or ("q_proj" in n) or ("k_proj" in n) \
               or ("v_proj" in n) or ("out_proj" in n)
    if scope == "ffn":
        return ("ffn" in n) or ("mlp" in n) or ("feedforward" in n) \
               or ("fc1" in n) or ("fc2" in n) or ("w1" in n) or ("w2" in n)
    return True


def _percentile_clip_symmetric(w: torch.Tensor, pct: float) -> float:
    if pct >= 1.0:
        return float(w.abs().amax().item())
    abs_w = w.detach().float().abs().view(-1).cpu()
    return float(torch.quantile(abs_w, torch.tensor(pct)).item())


def _choose_store_dtype(bits: int):
    if bits <= 8:
        return torch.uint8
    elif bits <= 16:
        return torch.uint16
    else:
        raise ValueError("bits>16 not supported")


def _quantize_dequantize_uniform(
    w: torch.Tensor,
    bits: int,
    symmetric: bool,
    per_channel: bool,
    clip_percentile: float,
    channel_axis: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
    orig_device = w.device
    w_cpu = w.detach().float().cpu()
    if bits < 2 or bits > 16:
        raise ValueError("bits must be in [2,16]")
    qmax = (1 << bits) - 1
    shift_for_storage = qmax // 2
    store_dtype = _choose_store_dtype(bits)

    if per_channel:
        t = w_cpu.contiguous()
        permute_back = None
        if channel_axis != 0:
            perm = [channel_axis] + [i for i in range(t.ndim) if i != channel_axis]
            t = t.permute(*perm).contiguous()
            permute_back = perm

        c = t.shape[0]
        flat = t.view(c, -1)
        dq_flat = torch.empty_like(flat)
        q_flat = torch.empty_like(flat, dtype=store_dtype)

        scales = torch.zeros(c)
        zeros = torch.zeros(c)
        lo_hi = torch.zeros((c, 2))

        for i in range(c):
            v = flat[i]
            if symmetric:
                a = _percentile_clip_symmetric(v, clip_percentile)
                a = max(a, 1e-12)
                scale = (2 * a) / qmax
                zp = 0.0
                q_signed = torch.clamp(torch.round(v / scale), -qmax/2, qmax/2)
                dq_flat[i] = q_signed * scale
                q_store = (q_signed + shift_for_storage).to(torch.int64)
                lo_hi[i, 0] = -a; lo_hi[i, 1] = a
            else:
                if clip_percentile < 1.0:
                    lo = float(torch.quantile(v, torch.tensor((1 - clip_percentile) / 2)).item())
                    hi = float(torch.quantile(v, torch.tensor(1 - (1 - clip_percentile) / 2)).item())
                else:
                    lo = float(v.min().item()); hi = float(v.max().item())
                if hi - lo < 1e-12:
                    dq_flat[i] = v
                    q_flat[i] = (v.to(torch.int64) & 0xFFFF).to(store_dtype)
                    scales[i] = 1.0; zeros[i] = 0.0
                    lo_hi[i, 0] = lo; lo_hi[i, 1] = hi
                    continue
                scale = (hi - lo) / qmax
                zp = -lo / scale
                q = torch.clamp(torch.round(v / scale + zp), 0, qmax)
                dq_flat[i] = (q - zp) * scale
                q_store = q.to(torch.int64)
                lo_hi[i, 0] = lo; lo_hi[i, 1] = hi

            scales[i] = float(scale)
            zeros[i] = float(zp)
            q_flat[i] = q_store.to(store_dtype)

        dq = dq_flat.view_as(t)
        q_int = q_flat.view_as(t)
        if permute_back is not None:
            inv = [permute_back.index(i) for i in range(len(permute_back))]
            dq = dq.permute(*inv).contiguous()
            q_int = q_int.permute(*inv).contiguous()
        dq = dq.to(orig_device)
        info = dict(mode="per-channel", scales=scales.numpy().tolist(), zeros=zeros.numpy().tolist(), range=lo_hi.numpy().tolist(), bits=bits, symmetric=symmetric)
        return dq, info, q_int.cpu()

    else:
        if symmetric:
            a = _percentile_clip_symmetric(w_cpu, clip_percentile)
            a = max(a, 1e-12)
            scale = (2 * a) / qmax
            zp = 0.0
            q_signed = torch.clamp(torch.round(w_cpu / scale), -qmax/2, qmax/2)
            dq = (q_signed * scale).to(orig_device)
            q_store = (q_signed + shift_for_storage).to(torch.int64).to(_choose_store_dtype(bits)).cpu()
            info = dict(mode="per-tensor", scale=float(scale), zero=float(zp), range=[-a, a], bits=bits, symmetric=symmetric)
            return dq, info, q_store
        else:
            if clip_percentile < 1.0:
                lo = float(torch.quantile(w_cpu, torch.tensor((1 - clip_percentile) / 2)).item())
                hi = float(torch.quantile(w_cpu, torch.tensor(1 - (1 - clip_percentile) / 2)).item())
            else:
                lo = float(w_cpu.min().item()); hi = float(w_cpu.max().item())
            if hi - lo < 1e-12:
                info = dict(mode="per-tensor", scale=1.0, zero=0.0, range=[lo, hi], bits=bits, symmetric=symmetric)
                return w.to(orig_device), info, (w_cpu.to(torch.int64) & 0xFFFF).to(_choose_store_dtype(bits)).cpu()
            scale = (hi - lo) / qmax
            zp = -lo / scale
            q = torch.clamp(torch.round(w_cpu / scale + zp), 0, qmax)
            dq = ((q - zp) * scale).to(orig_device)
            info = dict(mode="per-tensor", scale=float(scale), zero=float(zp), range=[lo, hi], bits=bits, symmetric=symmetric)
            return dq, info, q.to(_choose_store_dtype(bits)).cpu()


def _should_quantize_key(k: str, scope: str, groups_keys: Optional[Set[str]] = None) -> bool:
    if groups_keys is not None:
        return k in groups_keys
    return _match_scope(k, scope)


def quantize_state_dict(
    sd: Dict[str, torch.Tensor],
    bits: int,
    symmetric: bool,
    per_channel: bool,
    clip_percentile: float,
    scope: str,
    verbose: bool = False,
    report_writer: Optional[csv.writer] = None,
    store_ints: bool = False,
    groups_keys: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    new_sd = {}
    num_total = 0
    num_q = 0
    for k, v in sd.items():
        new_sd[k] = v
        if (not isinstance(v, torch.Tensor)) or (not torch.is_floating_point(v)) or v.ndim < 2:
            continue
        low = k.lower()
        if "norm" in low or "layernorm" in low or "ln_" in low or "bn" in low:
            continue
        if not _should_quantize_key(k, scope, groups_keys):
            continue

        num_total += 1
        if verbose:
            print(f"[Q] {k}: shape={tuple(v.shape)}")

        dq, info, q_int = _quantize_dequantize_uniform(
            v, bits=bits, symmetric=symmetric, per_channel=per_channel,
            clip_percentile=clip_percentile, channel_axis=0
        )

        if report_writer is not None:
            with torch.no_grad():
                diff = (v.detach().float().cpu() - dq.detach().float().cpu()).view(-1)
                mse = float((diff ** 2).mean().item())
                mae = float(diff.abs().mean().item())
            report_writer.writerow([k, "|".join(map(str, v.shape)), bits,
                                    "sym" if symmetric else "asym",
                                    "pc" if per_channel else "pt",
                                    clip_percentile, mse, mae])

        if store_ints and q_int is not None:
            meta = {
                "__quant__": True,
                "bits": bits,
                "symmetric": symmetric,
                "per_channel": per_channel,
                "meta": info,
                "q": q_int,  # cpu int tensor
                "orig_dtype": str(v.dtype),
            }
            new_sd[k] = meta
        else:
            new_sd[k] = dq.to(v.dtype)
        num_q += 1

    if verbose:
        print(f"[DONE] quantized {num_q}/{num_total} target tensors (scope={scope})")
    return new_sd


def dequantize_meta_entry(meta: Dict[str, Any]) -> torch.Tensor:
    if not (isinstance(meta, dict) and meta.get("__quant__", False)):
        raise ValueError("not a quant meta dict")
    bits = meta["bits"]
    per_channel = meta["per_channel"]
    info = meta["meta"]
    q = meta["q"]
    q = q.to(torch.int64)
    qmax = (1 << bits) - 1
    shift = qmax // 2
    if info["mode"] == "per-tensor":
        if meta["symmetric"]:
            scale = float(info["scale"])
            q_signed = (q.to(torch.float32) - float(shift))
            dq = q_signed * scale
            return dq
        else:
            scale = float(info["scale"]); zp = float(info["zero"])
            dq = (q.to(torch.float32) - zp) * scale
            return dq
    else:
        scales = np.array(info["scales"])
        zeros = np.array(info["zeros"])
        lohi = np.array(info["range"])
        qf = q.to(torch.float32)
        if meta["symmetric"]:
            q_signed = qf - float(shift)
            c = qf.shape[0]
            scales_t = torch.from_numpy(scales).view(c, *([1] * (qf.ndim - 1)))
            dq = (q_signed * scales_t)
            return dq
        else:
            scales_t = torch.from_numpy(scales).view(len(scales), *([1] * (qf.ndim - 1)))
            zeros_t = torch.from_numpy(zeros).view(len(zeros), *([1] * (qf.ndim - 1)))
            dq = (qf - zeros_t) * scales_t
            return dq


def _parse_bits_range(s: str) -> List[int]:
    if s is None:
        return []
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a = int(a); b = int(b)
        step = -1 if a > b else 1
        return list(range(a, b + step, step))
    else:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [int(x) for x in parts]


def main():
    args = parse_args()
    bits_list = []
    if args.bits_range is not None:
        bits_list = _parse_bits_range(args.bits_range)
    elif args.bits is not None:
        bits_list = [args.bits]
    else:
        raise RuntimeError("Either --bits or --bits-range must be provided")
    for bits in bits_list:
        if bits < 2 or bits > 16:
            raise RuntimeError("--bits must be in [2,16]")

    assert args.method == "uniform", "only uniform supported for now"

    print(f"[LOAD] {args.in_path}")
    ckpt = torch.load(args.in_path, map_location="cpu")
    if isinstance(ckpt, dict) and "module" in ckpt and isinstance(ckpt["module"], dict):
        sd = ckpt["module"]
        wrapper = ckpt
        uses_wrapper = True
    elif isinstance(ckpt, dict):
        sd = ckpt
        wrapper = None
        uses_wrapper = False
    else:
        raise RuntimeError("Unsupported checkpoint format")

    groups_keys = None
    if args.groups_json is not None and args.group is not None:
        import json
        groups = json.load(open(args.groups_json, "r"))
        groups_keys = set(groups.get(args.group, []))

    report_writer = None
    f = None
    if args.report_csv is not None:
        os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)
        f = open(args.report_csv, "w", newline="")
        report_writer = csv.writer(f)
        report_writer.writerow(["name", "shape", "bits", "sym/asym", "pc/pt", "clip_pct", "mse", "mae"])

    try:
        for bits in bits_list:
            print(f"[RUN] bits={bits}")
            new_sd = quantize_state_dict(
                sd,
                bits=bits,
                symmetric=args.symmetric,
                per_channel=args.per_channel,
                clip_percentile=args.clip_percentile,
                scope=args.scope,
                verbose=args.verbose,
                report_writer=report_writer,
                store_ints=args.store_ints,
                groups_keys=groups_keys,
            )
            if args.dry_run:
                print("[DRY] no file written for bits", bits)
                continue

            if len(bits_list) > 1:
                out_dir = args.out_path
                os.makedirs(out_dir, exist_ok=True)
                out_name = os.path.join(out_dir, f"model_q{bits}.pt")
            else:
                out_name = args.out_path
                os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)

            if uses_wrapper:
                wrapper["module"] = new_sd
                torch.save(wrapper, out_name)
            else:
                torch.save(new_sd, out_name)
            print(f"[OK] saved => {out_name}")
    finally:
        if f is not None:
            f.close()


if __name__ == "__main__":
    main()
