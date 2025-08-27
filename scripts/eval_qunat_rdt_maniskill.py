# eval_sim/eval_rdt_maniskill.py
from typing import Optional, Any
import sys
sys.path.append('/')

import os
import argparse
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
import yaml
import tqdm
import cv2  # live view

from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from mani_skill.utils import common, gym_utils   # noqa: F401
from gymnasium.wrappers.time_limit import TimeLimit

from scripts.maniskill_model import create_model

# -----------------------------
# Helpers to robustly read RGBs
# -----------------------------
def _to_numpy_rgb(x: Any) -> Optional[np.ndarray]:
    if torch.is_tensor(x):
        arr = x.detach().cpu()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        arr = arr.numpy()
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return None

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return None
    return None


def _find_first_frame(x: Any) -> Optional[np.ndarray]:
    arr = _to_numpy_rgb(x)
    if arr is not None:
        return arr
    if isinstance(x, (list, tuple)):
        for item in x:
            arr = _find_first_frame(item)
            if arr is not None:
                return arr
    if isinstance(x, dict):
        for v in x.values():
            arr = _find_first_frame(v)
            if arr is not None:
                return arr
    return None


def _extract_rgb(frame: Any, obs: Optional[dict] = None) -> np.ndarray:
    arr = _find_first_frame(frame)
    if arr is not None:
        return arr
    if obs is not None:
        arr = _find_first_frame(obs)
        if arr is not None:
            return arr
    raise RuntimeError(
        f"Could not extract an RGB ndarray from env.render()/obs. "
        f"Got types: render={type(frame)}, obs={type(obs)}. "
        f"Try obs_mode='rgb' and render_mode='rgb_array'."
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1",
                        help="Environment to run")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb")
    parser.add_argument("-n", "--num-traj", type=int, default=25)
    parser.add_argument("--only-count-success", action="store_true")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        help="'rgb_array' 권장 (human은 모델 입력 프레임 제공 안 함)")
    parser.add_argument("--shader", default="default", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to the pretrained model (.pt or .safetensors)")
    parser.add_argument("--random_seed", type=int, default=0)

    # 언어 임베딩/인코더/비전 설정
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, default=None,
                        help="텍스트 인코더 ID/경로. precomputed/skip로 주면 T5 로딩을 건너뜀")
    parser.add_argument("--lang_embeddings_path", type=str, default=None,
                        help="사전계산 언어 임베딩(.pt) 경로. 지정하지 않으면 text_embed_<ENV>.pt 자동 탐색")
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str,
                        default="google/siglip-so400m-patch14-384", help="비전 인코더 ID/경로")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                        help="모델 추론 dtype")

    # 렌더링
    parser.add_argument("--live-view", action="store_true",
                        help="실시간 OpenCV 창으로 프레임 표시")

    # 에피소드 길이/스텝
    parser.add_argument("--max-steps", type=int, default=400,
                        help="에피소드 최대 스텝 수(TimeLimit). ManiSkill 기본 50을 덮어씀.")
    parser.add_argument("--action-downsample", type=int, default=4,
                        help="RDT 64-step 예측에서 몇 스텝마다 실행할지(기본 4 → 16회 실행). 1로 낮추면 더 오래 실행.")

    # (선택) 양자화 아블레이션 옵션(오프라인 가중치 저장 방식과는 독립)
    parser.add_argument("--quant", type=str, default="none",
                        choices=["none", "8bit", "4bit"], help="weight-only 양자화 모드")
    parser.add_argument("--quant-modules", type=str, nargs="*", default=None,
                        help="양자화할 모듈 이름의 정규식 패턴 목록. 지정하지 않으면 --quant-scope 사용")
    parser.add_argument("--quant-scope", type=str, default="all",
                        choices=["all", "attn", "ffn"], help="양자화 대상 범위")
    parser.add_argument("--quant-compute-dtype", type=str, default=None,
                        choices=[None, "fp16", "bf16", "fp32"],
                        help="bnb 연산 dtype(활성/계산 경로)")
    parser.add_argument("--quant-include-vision", action="store_true",
                        help="SigLIP 비전 인코더도 양자화")

    # ▶ 영상 저장 옵션
    parser.add_argument("--save-video-dir", type=str, default=None,
                        help="지정하면 에피소드별 mp4를 저장")
    parser.add_argument("--video-fps", type=int, default=20, help="저장 영상 FPS")
    parser.add_argument("--video-fourcc", type=str, default="mp4v",
                        help="cv2.VideoWriter_fourcc code. 예: mp4v, avc1, H264")

    return parser.parse_args(args)


def set_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_lang_embed(env_id: str, explicit_path: Optional[str]):
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates += [
        f'./text_embed_{env_id}.pt',
        f'lang_embeds/text_embed_{env_id}.pt',
        f'data/text_embed_{env_id}.pt',
    ]
    for p in candidates:
        if p and os.path.exists(p):
            print(f"[INFO] Using precomputed language embedding: {p}")
            emb = torch.load(p, map_location="cpu")
            if isinstance(emb, torch.Tensor) and emb.ndim == 2:
                emb = emb.unsqueeze(0)
            return emb
    raise FileNotFoundError(
        f"Precomputed language embedding not found. "
        f"Pass --lang_embeddings_path or place text_embed_{env_id}.pt at repo root."
    )


def _force_time_limit(env: gym.Env, max_steps: int) -> gym.Env:
    cur = env
    while isinstance(cur, TimeLimit):
        cur = cur.env
    return TimeLimit(cur, max_episode_steps=max_steps)


def main():
    args = parse_args()

    if args.render_mode != "rgb_array":
        print("[WARN] 'human' viewer는 ndarray 프레임을 돌려주지 않습니다. "
              "모델 입력/라이브뷰/영상저장을 위해 'rgb_array'로 전환합니다.")
        args.render_mode = "rgb_array"

    set_seeds(args.random_seed)

    # ---------- Gym env ----------
    try:
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            reward_mode="dense" if args.reward_mode is None else args.reward_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            sim_backend=args.sim_backend,
            max_episode_steps=args.max_steps,
        )
    except TypeError:
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            reward_mode="dense" if args.reward_mode is None else args.reward_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            sim_backend=args.sim_backend,
        )
    env = _force_time_limit(env, args.max_steps)

    # ---------- Config ----------
    with open('configs/base.yaml', "r") as fp:
        config = yaml.safe_load(fp)

    # ---------- DType ----------
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # ---------- Model ----------
    policy = create_model(
        args=config,
        dtype=torch_dtype,
        pretrained=args.pretrained_path,
        pretrained_text_encoder_name_or_path=args.pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=args.pretrained_vision_encoder_name_or_path,
        quant_mode=args.quant,
        quant_modules=args.quant_modules,
        quant_scope=args.quant_scope,
        quant_compute_dtype=args.quant_compute_dtype,
        quant_include_vision=args.quant_include_vision
    )

    # 추론 모드 및 성능 옵션
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.policy = policy.policy.to(device, dtype=torch_dtype)
    if getattr(policy, "vision_model", None) is not None:
        policy.vision_model = policy.vision_model.to(device, dtype=torch_dtype)
    if getattr(policy, "text_model", None) is not None:
        policy.text_model = policy.text_model.to(device, dtype=torch_dtype)
    policy.reset()

    # ---------- Precomputed language embedding ----------
    text_embed = load_lang_embed(args.env_id, args.lang_embeddings_path)

    # ---------- Eval loop ----------
    MAX_EPISODE_STEPS = args.max_steps
    total_episodes = args.num_traj
    success_count = 0
    base_seed = 20241201
    down = max(1, int(args.action_downsample))

    window_name = f"RDT Live - {args.env_id}"
    if args.live_view:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # 비디오 저장 준비
    save_video = args.save_video_dir is not None
    if save_video:
        os.makedirs(args.save_video_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.video_fourcc)

    try:
        for episode in tqdm.trange(total_episodes):
            obs_window = deque(maxlen=2)
            obs, _ = env.reset(seed=episode + base_seed)

            # 첫 프레임
            frame_any = env.render()
            img = _extract_rgb(frame_any, obs)
            obs_window.append(None)
            obs_window.append(np.array(img))

            if args.live_view:
                cv2.imshow(window_name, img[..., ::-1])
                cv2.waitKey(1)

            # 비디오 라이터 준비
            writer = None
            if save_video:
                h, w = img.shape[:2]
                ckpt_tag = os.path.basename(args.pretrained_path or "ckpt")
                out_name = f"{ckpt_tag}__{args.env_id}__ep{episode+1:02d}.mp4"
                out_path = os.path.join(args.save_video_dir, out_name)
                writer = cv2.VideoWriter(out_path, fourcc, args.video_fps, (w, h))
                writer.write(img[..., ::-1])  # RGB->BGR

            # proprio
            proprio = obs['agent']['qpos'][:, :-1]

            global_steps = 0
            done = False

            while global_steps < MAX_EPISODE_STEPS and not done:
                # 이미지 6장 슬롯 준비
                image_arrs = []
                for window_img in obs_window:
                    image_arrs.append(window_img)
                    image_arrs.append(None)
                    image_arrs.append(None)
                images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]

                # 모델 추론
                with torch.amp.autocast('cuda', enabled=(device == "cuda"), dtype=torch_dtype):
                    actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()

                # downsample
                actions = actions[::down, :]

                for idx in range(actions.shape[0]):
                    action = actions[idx]
                    obs, reward, terminated, truncated, info = env.step(action)

                    # 다음 프레임
                    frame_any = env.render()
                    try:
                        img = _extract_rgb(frame_any, obs)
                        obs_window.append(img)
                        if args.live_view:
                            cv2.imshow(window_name, img[..., ::-1])
                            cv2.waitKey(1)
                        if writer is not None:
                            writer.write(img[..., ::-1])
                    except RuntimeError:
                        pass

                    proprio = obs['agent']['qpos'][:, :-1]
                    global_steps += 1

                    if terminated or truncated:
                        if info.get('success', False):
                            success_count += 1
                        done = True
                        break

            print(f"Trial {episode+1} finished, success: {info.get('success', False)}, steps: {global_steps}")

            if writer is not None:
                writer.release()

    finally:
        if args.live_view:
            cv2.destroyAllWindows()

    success_rate = success_count / total_episodes * 100
    print(f"Success rate: {success_rate:.2f}%")
    env.close()


if __name__ == "__main__":
    main()