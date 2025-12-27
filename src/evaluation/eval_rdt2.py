# eval_sim/eval_rdt2_maniskill.py
# RDT-2 (Qwen2.5-VL-7B based) evaluation script for ManiSkill

from typing import Optional, Any
import sys
sys.path.append('/')

import os
import argparse
import random
from collections import deque

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
import torch
from PIL import Image
import tqdm
import cv2  # live view

from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from mani_skill.utils import common, gym_utils   # noqa: F401
from mani_skill.utils.wrappers.record import RecordEpisode

# RDT-2 dependencies
try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    HAS_RDT2_DEPS = True
except ImportError:
    HAS_RDT2_DEPS = False
    print("[WARN] RDT-2 dependencies not found. Install transformers>=4.40 and flash-attn")


# -----------------------------
# Rotation conversion utilities
# -----------------------------
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to rotation matrix.
    rot6d: (..., 6) - two 3D vectors (first two columns of rotation matrix)
    Returns: (..., 3, 3) rotation matrix
    """
    # Gram-Schmidt orthogonalization
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Normalize first vector
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

    # Make second vector orthogonal to first
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)

    # Third vector is cross product
    b3 = np.cross(b1, b2)

    # Stack to form rotation matrix
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_euler(mat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to euler angles (XYZ convention).
    mat: (..., 3, 3)
    Returns: (..., 3) euler angles in radians
    """
    # Handle batch dimension
    shape = mat.shape[:-2]
    mat = mat.reshape(-1, 3, 3)

    euler = np.zeros((mat.shape[0], 3))

    for i in range(mat.shape[0]):
        m = mat[i]
        # XYZ euler angles
        sy = np.sqrt(m[0, 0]**2 + m[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            euler[i, 0] = np.arctan2(m[2, 1], m[2, 2])  # roll
            euler[i, 1] = np.arctan2(-m[2, 0], sy)       # pitch
            euler[i, 2] = np.arctan2(m[1, 0], m[0, 0])   # yaw
        else:
            euler[i, 0] = np.arctan2(-m[1, 2], m[1, 1])
            euler[i, 1] = np.arctan2(-m[2, 0], sy)
            euler[i, 2] = 0

    return euler.reshape(*shape, 3)


def rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to euler angles."""
    mat = rot6d_to_matrix(rot6d)
    return matrix_to_euler(mat)


def euler_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles (XYZ) to rotation matrix."""
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


# =============================================================================
# APPROACH 1: Delta-based conversion (original approach with configurable scale)
# =============================================================================
def convert_approach_delta(
    rdt2_action: np.ndarray,
    prev_pos: np.ndarray = None,
    prev_euler: np.ndarray = None,
    pos_scale: float = 1.0,
    rot_scale: float = 1.0
) -> np.ndarray:
    """
    Delta-based conversion: compute deltas between consecutive RDT-2 predictions.

    Args:
        rdt2_action: (T, 20) or (T, 10) - RDT-2 action prediction
        prev_pos: Previous position for first frame delta
        prev_euler: Previous euler for first frame delta
        pos_scale: Scale factor for position deltas
        rot_scale: Scale factor for rotation deltas

    Returns:
        (T, 7) ManiSkill pd_ee_delta_pose action
    """
    if rdt2_action.shape[-1] == 20:
        rdt2_action = rdt2_action[..., :10]

    T = rdt2_action.shape[0]
    maniskill_actions = np.zeros((T, 7))

    all_pos = rdt2_action[:, :3]
    all_rot6d = rdt2_action[:, 3:9]
    all_gripper = rdt2_action[:, 9]
    all_euler = np.array([rot6d_to_euler(rot6d) for rot6d in all_rot6d])

    for t in range(T):
        pos = all_pos[t]
        euler = all_euler[t]
        gripper = all_gripper[t]

        if t == 0:
            if prev_pos is not None and prev_euler is not None:
                delta_pos = pos - prev_pos
                delta_euler = euler - prev_euler
            else:
                delta_pos = pos * 0.1
                delta_euler = euler * 0.1
        else:
            delta_pos = pos - all_pos[t - 1]
            delta_euler = euler - all_euler[t - 1]

        delta_pos_scaled = np.clip(delta_pos * pos_scale, -1.0, 1.0)
        delta_euler_scaled = np.clip(delta_euler * rot_scale, -1.0, 1.0)
        gripper_action = np.clip(gripper, -1.0, 1.0)

        maniskill_actions[t] = np.concatenate([delta_pos_scaled, delta_euler_scaled, [gripper_action]])

    return maniskill_actions


# =============================================================================
# APPROACH 2: Raw action (no scaling, direct use of RDT-2 output)
# =============================================================================
def convert_approach_raw(rdt2_action: np.ndarray) -> np.ndarray:
    """
    Raw conversion: use RDT-2 output directly as delta actions.
    Assumes normalizer already produces reasonable values.

    Args:
        rdt2_action: (T, 20) or (T, 10) - RDT-2 action prediction

    Returns:
        (T, 7) ManiSkill pd_ee_delta_pose action
    """
    if rdt2_action.shape[-1] == 20:
        rdt2_action = rdt2_action[..., :10]

    T = rdt2_action.shape[0]
    maniskill_actions = np.zeros((T, 7))

    for t in range(T):
        pos = rdt2_action[t, :3]
        rot6d = rdt2_action[t, 3:9]
        gripper = rdt2_action[t, 9]

        euler = rot6d_to_euler(rot6d)

        # Clip to valid range but don't scale
        delta_pos = np.clip(pos, -1.0, 1.0)
        delta_euler = np.clip(euler, -1.0, 1.0)
        gripper_action = np.clip(gripper, -1.0, 1.0)

        maniskill_actions[t] = np.concatenate([delta_pos, delta_euler, [gripper_action]])

    return maniskill_actions


# =============================================================================
# APPROACH 3: RDT-2 style conversion (based on original RDT2 deploy code)
# =============================================================================
def convert_approach_rdt2_style(
    rdt2_action: np.ndarray,
    current_ee_pos: np.ndarray = None,
    current_ee_euler: np.ndarray = None,
    pos_scale: float = 1.0,
    rot_scale: float = 1.0
) -> np.ndarray:
    """
    RDT-2 style conversion: treat RDT-2 output as relative transformation
    and apply it to current EE pose.

    Based on RDT2's get_real_umi_action() logic.

    Args:
        rdt2_action: (T, 20) or (T, 10) - RDT-2 action prediction
        current_ee_pos: Current end-effector position (3,)
        current_ee_euler: Current end-effector euler angles (3,)
        pos_scale: Scale factor for position
        rot_scale: Scale factor for rotation

    Returns:
        (T, 7) ManiSkill pd_ee_delta_pose action
    """
    if rdt2_action.shape[-1] == 20:
        rdt2_action = rdt2_action[..., :10]

    T = rdt2_action.shape[0]
    maniskill_actions = np.zeros((T, 7))

    for t in range(T):
        action_pos = rdt2_action[t, :3]
        action_rot6d = rdt2_action[t, 3:9]
        gripper = rdt2_action[t, 9]

        # Convert 6D rotation to matrix
        action_rot_mat = rot6d_to_matrix(action_rot6d)

        if current_ee_pos is not None and current_ee_euler is not None:
            current_rot_mat = euler_to_matrix(current_ee_euler)

            # RDT-2 uses relative action: target = current @ action
            # So delta = target - current = current @ action - current
            target_pos = current_ee_pos + action_pos  # Simplified: assume action_pos is delta
            target_rot_mat = current_rot_mat @ action_rot_mat

            delta_pos = target_pos - current_ee_pos
            target_euler = matrix_to_euler(target_rot_mat.reshape(1, 3, 3))[0]
            delta_euler = target_euler - current_ee_euler
        else:
            # Without current pose, use action directly
            delta_pos = action_pos
            delta_euler = matrix_to_euler(action_rot_mat.reshape(1, 3, 3))[0]

        delta_pos_scaled = np.clip(delta_pos * pos_scale, -1.0, 1.0)
        delta_euler_scaled = np.clip(delta_euler * rot_scale, -1.0, 1.0)
        gripper_action = np.clip(gripper, -1.0, 1.0)

        maniskill_actions[t] = np.concatenate([delta_pos_scaled, delta_euler_scaled, [gripper_action]])

    return maniskill_actions


# =============================================================================
# APPROACH 4: Direct use with different scales (sweep approach)
# =============================================================================
def convert_approach_direct(
    rdt2_action: np.ndarray,
    pos_scale: float = 1.0,
    rot_scale: float = 1.0
) -> np.ndarray:
    """
    Direct conversion: treat RDT-2 position output as delta, scale accordingly.

    Args:
        rdt2_action: (T, 20) or (T, 10) - RDT-2 action prediction
        pos_scale: Scale factor for position
        rot_scale: Scale factor for rotation

    Returns:
        (T, 7) ManiSkill pd_ee_delta_pose action
    """
    if rdt2_action.shape[-1] == 20:
        rdt2_action = rdt2_action[..., :10]

    T = rdt2_action.shape[0]
    maniskill_actions = np.zeros((T, 7))

    for t in range(T):
        pos = rdt2_action[t, :3]
        rot6d = rdt2_action[t, 3:9]
        gripper = rdt2_action[t, 9]

        euler = rot6d_to_euler(rot6d)

        # Scale and clip
        delta_pos = np.clip(pos * pos_scale, -1.0, 1.0)
        delta_euler = np.clip(euler * rot_scale, -1.0, 1.0)
        gripper_action = np.clip(gripper, -1.0, 1.0)

        maniskill_actions[t] = np.concatenate([delta_pos, delta_euler, [gripper_action]])

    return maniskill_actions


# =============================================================================
# Master conversion function that dispatches to appropriate approach
# =============================================================================
def convert_rdt2_to_maniskill_action(
    rdt2_action: np.ndarray,
    approach: str = "delta",
    prev_pos: np.ndarray = None,
    prev_euler: np.ndarray = None,
    current_ee_pos: np.ndarray = None,
    current_ee_euler: np.ndarray = None,
    pos_scale: float = 1.0,
    rot_scale: float = 1.0
) -> np.ndarray:
    """
    Master conversion function that dispatches to the appropriate approach.

    Args:
        rdt2_action: (T, 20) or (T, 10) - RDT-2 action prediction
        approach: Conversion approach ("delta", "raw", "rdt2_style", "direct")
        prev_pos: Previous position (for delta approach)
        prev_euler: Previous euler (for delta approach)
        current_ee_pos: Current EE position (for rdt2_style approach)
        current_ee_euler: Current EE euler (for rdt2_style approach)
        pos_scale: Position scale factor
        rot_scale: Rotation scale factor

    Returns:
        (T, 7) ManiSkill action
    """
    if approach == "delta":
        return convert_approach_delta(rdt2_action, prev_pos, prev_euler, pos_scale, rot_scale)
    elif approach == "raw":
        return convert_approach_raw(rdt2_action)
    elif approach == "rdt2_style":
        return convert_approach_rdt2_style(rdt2_action, current_ee_pos, current_ee_euler, pos_scale, rot_scale)
    elif approach == "direct":
        return convert_approach_direct(rdt2_action, pos_scale, rot_scale)
    else:
        raise ValueError(f"Unknown approach: {approach}")


# -----------------------------
# Helpers to robustly read RGBs
# -----------------------------
def _to_numpy_rgb(x: Any) -> Optional[np.ndarray]:
    """Convert torch.Tensor/np.ndarray to (H, W, 3) numpy RGB if possible."""
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
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return None
    return None


def _find_first_frame(x: Any) -> Optional[np.ndarray]:
    """Find first RGB frame in nested dict/list/tuple/tensor/ndarray."""
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
    """Try to extract (H, W, 3) numpy RGB from env.render() or obs."""
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


# Task instructions for ManiSkill environments
TASK_INSTRUCTIONS = {
    "PickCube-v1": "Pick up the cube.",
    "PushCube-v1": "Push the cube to the target.",
    "StackCube-v1": "Stack the cubes.",
    "PegInsertionSide-v1": "Insert the peg into the hole.",
    "PlugCharger-v1": "Plug in the charger.",
}


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
    parser.add_argument("--pretrained_path", type=str,
                        default="robotics-diffusion-transformer/RDT2-VQ",
                        help="Path or HF model ID for RDT-2")
    parser.add_argument("--vae_path", type=str,
                        default="robotics-diffusion-transformer/RVQActionTokenizer",
                        help="Path or HF model ID for RVQ VAE")
    parser.add_argument("--normalizer_path", type=str,
                        default=None,
                        help="Path to normalizer .pt file")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=None,
                        help="Custom instruction for the task")

    # 렌더링
    parser.add_argument("--live-view", action="store_true",
                        help="실시간 OpenCV 창으로 프레임 표시")

    # 비디오 저장
    parser.add_argument("--save-video", action="store_true",
                        help="RecordEpisode를 사용하여 비디오 저장")
    parser.add_argument("--output-dir", type=str, default="videos",
                        help="비디오 저장 경로")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="저장할 비디오의 FPS")

    # 에피소드 길이/스텝
    parser.add_argument("--max-steps", type=int, default=400,
                        help="에피소드 최대 스텝 수(TimeLimit). ManiSkill 기본 50을 덮어씀.")
    parser.add_argument("--action-downsample", type=int, default=1,
                        help="RDT-2 24-step 예측에서 몇 스텝마다 실행할지")

    # dtype
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                        help="모델 추론 dtype")

    # Multi-approach experiment arguments
    parser.add_argument("--approach", type=str, default="delta",
                        choices=["delta", "raw", "rdt2_style", "direct"],
                        help="Action conversion approach")
    parser.add_argument("--pos-scale", type=float, default=1.0,
                        help="Position scale factor for delta/direct approaches")
    parser.add_argument("--rot-scale", type=float, default=1.0,
                        help="Rotation scale factor for delta/direct approaches")

    return parser.parse_args(args)


def set_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _force_time_limit(env: gym.Env, max_steps: int) -> gym.Env:
    """ManiSkill 기본 50 스텝 TimeLimit을 원하는 값으로 강제 교체."""
    cur = env
    while isinstance(cur, TimeLimit):
        cur = cur.env
    return TimeLimit(cur, max_episode_steps=max_steps)


class RDT2Policy:
    """Wrapper for RDT-2 model inference."""

    def __init__(
        self,
        model_path: str = "robotics-diffusion-transformer/RDT2-VQ",
        vae_path: str = "robotics-diffusion-transformer/RVQActionTokenizer",
        normalizer_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        print(f"[INFO] Loading RDT-2 model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        # Try flash attention, fallback to sdpa
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                device_map=device
            ).eval()
        except Exception as e:
            print(f"[WARN] Flash attention failed ({e}), using sdpa...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map=device
            ).eval()

        # Load VAE
        print(f"[INFO] Loading RVQ VAE from {vae_path}...")
        try:
            from vqvae import MultiVQVAE
            self.vae = MultiVQVAE.from_pretrained(vae_path).eval()
            self.vae = self.vae.to(device=device, dtype=torch.float32)
            self.valid_action_id_length = self.vae.pos_id_len + self.vae.rot_id_len + self.vae.grip_id_len
        except ImportError:
            print("[WARN] vqvae module not found. Clone https://github.com/thu-ml/RDT2")
            self.vae = None
            self.valid_action_id_length = None

        # Load normalizer
        self.normalizer = None
        if normalizer_path and os.path.exists(normalizer_path):
            print(f"[INFO] Loading normalizer from {normalizer_path}...")
            try:
                from models.normalizer import LinearNormalizer
                self.normalizer = LinearNormalizer.load(normalizer_path)

                # DEBUG: Print normalizer statistics
                print("[DEBUG NORMALIZER] Normalizer loaded. Checking statistics...")
                if hasattr(self.normalizer, 'params_dict'):
                    for key, params in self.normalizer.params_dict.items():
                        print(f"[DEBUG NORMALIZER] Key: {key}")
                        if hasattr(params, 'scale'):
                            print(f"  scale shape: {params.scale.shape}, range: [{params.scale.min():.6f}, {params.scale.max():.6f}]")
                        if hasattr(params, 'offset'):
                            print(f"  offset shape: {params.offset.shape}, range: [{params.offset.min():.6f}, {params.offset.max():.6f}]")
                        if hasattr(params, 'input_stats'):
                            stats = params.input_stats
                            if 'min' in stats:
                                print(f"  input_stats min[:10]: {stats['min'][:10].tolist()}")
                            if 'max' in stats:
                                print(f"  input_stats max[:10]: {stats['max'][:10].tolist()}")
            except ImportError:
                print("[WARN] normalizer module not found")

    def predict_action(
        self,
        images: list,
        instruction: str,
    ) -> np.ndarray:
        """
        Predict action chunk from images and instruction.

        Args:
            images: List of PIL Images or numpy arrays (expects 2 cameras: camera0, camera1)
            instruction: Text instruction for the task

        Returns:
            action_chunk: (T, D) numpy array where T=24, D=20 (or adapted for ManiSkill)
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Please install RDT2 dependencies.")

        # Prepare observation dict
        obs = {"meta": {"num_camera": len(images)}}
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                # Ensure uint8
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                # Resize to 384x384
                if img.shape[:2] != (384, 384):
                    img = cv2.resize(img, (384, 384))
                # Convert to torch tensor (batch_predict_action expects torch tensors)
                img = torch.from_numpy(img).to(self.device)
                img = img.unsqueeze(0)  # (1, 384, 384, 3)
            elif torch.is_tensor(img):
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)
            obs[f"camera{i}_rgb"] = img

        try:
            from utils import batch_predict_action
            result = batch_predict_action(
                self.model,
                self.processor,
                self.vae,
                self.normalizer,
                examples=[{"obs": obs, "meta": obs["meta"]}],
                valid_action_id_length=self.valid_action_id_length,
                apply_jpeg_compression=True,
                instruction=instruction
            )
            return result["action_pred"][0]  # (24, 20)
        except ImportError:
            raise RuntimeError("RDT2 utils not found. Clone https://github.com/thu-ml/RDT2")


def main():
    args = parse_args()

    if not HAS_RDT2_DEPS:
        print("[ERROR] RDT-2 dependencies not available. Please install:")
        print("  pip install transformers>=4.40 flash-attn")
        print("  git clone https://github.com/thu-ml/RDT2")
        return

    # 사람이 보기 위해 human을 주더라도 모델 입력을 위해 rgb_array가 필요
    if args.render_mode != "rgb_array":
        print("[WARN] 'human' viewer는 ndarray 프레임을 돌려주지 않습니다. "
              "모델 입력/라이브뷰를 위해 'rgb_array'로 전환합니다.")
        args.render_mode = "rgb_array"

    set_seeds(args.random_seed)

    # ---------- Gym env ----------
    # Check if we should disable rendering (for headless environments without GPU Vulkan)
    if os.environ.get("MANI_SKILL_NO_RENDER", "0") == "1":
        args.render_mode = "sensors"  # state-based, no visual rendering
        args.save_video = False
        print("[INFO] Rendering disabled (MANI_SKILL_NO_RENDER=1)")

    # Use pd_ee_delta_pose for RDT-2 (outputs EE poses, not joint positions)
    try:
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_ee_delta_pose",
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
            control_mode="pd_ee_delta_pose",
            render_mode=args.render_mode,
            reward_mode="dense" if args.reward_mode is None else args.reward_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            sim_backend=args.sim_backend,
        )
    env = _force_time_limit(env, args.max_steps)

    # ---------- RecordEpisode Wrapper ----------
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=args.output_dir,
            save_trajectory=True,
            trajectory_name=f"trajectory_{args.env_id}_rdt2",
            save_video=True,
            video_fps=args.video_fps
        )
        print(f"[INFO] Video recording enabled. Saving to: {args.output_dir}")

    # ---------- DType ----------
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # ---------- Model ----------
    policy = RDT2Policy(
        model_path=args.pretrained_path,
        vae_path=args.vae_path,
        normalizer_path=args.normalizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch_dtype,
    )

    # ---------- Instruction ----------
    instruction = args.instruction or TASK_INSTRUCTIONS.get(args.env_id, "Complete the task.")
    print(f"[INFO] Using instruction: {instruction}")
    print(f"[INFO] Approach: {args.approach}, pos_scale: {args.pos_scale}, rot_scale: {args.rot_scale}")

    # ---------- Eval loop ----------
    MAX_EPISODE_STEPS = args.max_steps
    total_episodes = args.num_traj
    success_count = 0
    base_seed = 20241201
    down = max(1, int(args.action_downsample))

    window_name = f"RDT-2 Live - {args.env_id}"
    if args.live_view:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        for episode in tqdm.trange(total_episodes):
            obs_window = deque(maxlen=2)
            obs, _ = env.reset(seed=episode + base_seed)

            # 첫 프레임
            frame_any = env.render()
            img = _extract_rgb(frame_any, obs)
            obs_window.append(np.array(img))

            if args.live_view:
                cv2.imshow(window_name, img[..., ::-1])  # RGB -> BGR
                cv2.waitKey(1)

            global_steps = 0
            done = False

            prev_pos = None
            prev_euler = None

            while global_steps < MAX_EPISODE_STEPS and not done:
                # Prepare images for RDT-2 (expects 2 cameras)
                images = list(obs_window)
                if len(images) < 2:
                    images = images + [images[-1]]  # Duplicate if only 1

                # 모델 추론
                try:
                    actions = policy.predict_action(images, instruction)
                except Exception as e:
                    print(f"[ERROR] Prediction failed: {e}")
                    break

                # RDT-2 outputs (24, 20): 2 arms × (pos(3) + rot6d(6) + gripper(1))
                # Convert to ManiSkill pd_ee_delta_pose format (7D)
                actions_np = actions.detach().cpu().numpy()
                actions_np = actions_np[::down]  # Downsample temporally

                # Debug: Print raw RDT-2 output values (only first few steps)
                if global_steps < 3:
                    print(f"\n[DEBUG] Step {global_steps}, Approach: {args.approach}")
                    print(f"  RDT-2 raw output shape: {actions_np.shape}")
                    first_action = actions_np[0, :10]  # First arm, first timestep
                    print(f"  pos[:3] = {first_action[:3]}")
                    print(f"  rot6d[3:9] = {first_action[3:9]}")
                    print(f"  gripper[9] = {first_action[9]}")

                # Convert RDT-2 format to ManiSkill format using selected approach
                maniskill_actions = convert_rdt2_to_maniskill_action(
                    actions_np,
                    approach=args.approach,
                    prev_pos=prev_pos,
                    prev_euler=prev_euler,
                    current_ee_pos=None,  # TODO: extract from obs for rdt2_style
                    current_ee_euler=None,
                    pos_scale=args.pos_scale,
                    rot_scale=args.rot_scale
                )

                # Debug: Print converted ManiSkill actions
                if global_steps < 3:
                    print(f"  -> ManiSkill action[0]: {maniskill_actions[0]}")

                # Update prev state from last action for next iteration
                last_action = actions_np[-1, :10] if actions_np.shape[-1] >= 10 else actions_np[-1]
                prev_pos = last_action[:3]
                prev_euler = rot6d_to_euler(last_action[3:9])

                for idx in range(maniskill_actions.shape[0]):
                    action = maniskill_actions[idx]
                    obs, reward, terminated, truncated, info = env.step(action)

                    # 다음 프레임
                    frame_any = env.render()
                    try:
                        img = _extract_rgb(frame_any, obs)
                        obs_window.append(img)
                        if args.live_view:
                            cv2.imshow(window_name, img[..., ::-1])
                            cv2.waitKey(1)
                    except RuntimeError:
                        pass

                    global_steps += 1

                    if terminated or truncated:
                        if info.get('success', False):
                            success_count += 1
                        done = True
                        break

            print(f"Trial {episode+1} finished, success: {info.get('success', False)}, steps: {global_steps}")

    finally:
        if args.live_view:
            cv2.destroyAllWindows()

    success_rate = success_count / total_episodes * 100
    print(f"Success rate: {success_rate:.2f}%")
    env.close()


if __name__ == "__main__":
    main()
