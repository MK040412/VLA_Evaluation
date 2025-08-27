#!/usr/bin/env python3
import os, argparse, torch
# 파일에 존재하면 사용; 없으면 에러 안내
try:
    from tools.quantize_rdt_weights_adv import dequantize_meta_entry
except Exception as e:
    raise RuntimeError("tools/quantize_rdt_weights_adv.dequantize_meta_entry not found. 확인하세요.") from e

def walk_and_dequant(sd):
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict):
            # assume quant-meta entry -> dequantize helper returns tensor
            try:
                t = dequantize_meta_entry(v)
                out[k] = t.to(torch.float16)
            except Exception:
                # fallback: keep original dict
                out[k] = v
        else:
            out[k] = v
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", default=None)
    args = p.parse_args()

    ck = torch.load(args.in_path, map_location="cpu")
    # support mp_rank style wrapping
    if "module" in ck and isinstance(ck["module"], dict):
        sd = ck["module"]
        new_sd = walk_and_dequant(sd)
        ck["module"] = new_sd
    elif isinstance(ck, dict):
        # try to find first dict-like tensor map
        if any(isinstance(v, dict) for v in ck.values()):
            # dequant all dict entries at top-level
            new = walk_and_dequant(ck)
            ck = new
        else:
            raise RuntimeError("입력 체크포인트에 quant-meta가 없는 것 같습니다.")
    else:
        raise RuntimeError("지원하지 않는 체크포인트 포맷")

    outp = args.out_path or (os.path.splitext(args.in_path)[0] + "_dequant_fp16.pt")
    torch.save(ck, outp)
    print("[OK] wrote dequantized checkpoint ->", outp)

if __name__ == "__main__":
    main()