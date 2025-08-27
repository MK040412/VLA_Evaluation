# scripts/maniskill_model.py
import os

import torch
import torch.nn as nn
import numpy as np
import re
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner

try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


MANISKILL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [STATE_VEC_IDX_MAPPING[f"right_gripper_open"]]


DATA_STAT = {
    'state_min': [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0],
    'state_max': [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033],
    'action_min': [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0],
    'action_max': [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0]
}


def create_model(args, pretrained, **kwargs):
    """
    kwargs may include:
      - device, dtype, image_size, control_frequency
      - pretrained_text_encoder_name_or_path
      - pretrained_vision_encoder_name_or_path
      - quant_mode ("none"|"8bit"|"4bit")
      - quant_scope ("all"|"attn"|"ffn")
      - quant_compute_dtype (None|"fp16"|"bf16"|"fp32")
      - quant_include_vision (bool)
    """
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    if pretrained is not None:
        model.load_pretrained_weights(pretrained)
    # 양자화 옵션 적용 (가중치 로드 후!)
    quant_mode = kwargs.get("quant_mode", "none")
    if quant_mode != "none":
        quant_modules = kwargs.get("quant_modules")
        if not quant_modules:
            quant_scope = kwargs.get("quant_scope", "all")
            if quant_scope == "attn":
                quant_modules = ["attn\\."]
            elif quant_scope == "ffn":
                quant_modules = ["mlp\\."]
            # quant_scope가 다른 값일 경우(예: 'all') quant_modules를 None으로 유지
            else:
                quant_modules = None

        model.apply_quantization(
            mode=quant_mode,
            quant_modules=quant_modules,
            compute_dtype=kwargs.get("quant_compute_dtype", None),
            include_vision=kwargs.get("quant_include_vision", False),
        )
    if quant_mode != "none":
        print(f"[DEBUG] Checking policy layers after quantization for mode: {quant_mode}")
        for name, module in model.policy.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"[DEBUG] Layer {name} type: {type(module)}")
                break # Just check the first one

    return model


class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model.
       1) Model initialization
       2) (Optional) Encodings of instructions
       3) Model inference
    """
    def __init__(
        self, args,
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=25,
        pretrained_text_encoder_name_or_path=None,
        pretrained_vision_encoder_name_or_path=None,
        **_,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency

        # Text encoder (can be skipped for precomputed embeddings)
        self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)

        # Vision encoder
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)

        # Policy
        self.policy = self.get_policy()

        # Norm stats
        self.state_min = torch.tensor(DATA_STAT['state_min']).to(device)
        self.state_max = torch.tensor(DATA_STAT['state_max']).to(device)
        self.action_min = torch.tensor(DATA_STAT['action_min']).to(device)
        self.action_max = torch.tensor(DATA_STAT['action_max']).to(device)

        self.reset()

    # ---------------- Model builders ----------------
    def get_policy(self):
        img_cond_len = (
            self.args["common"]["img_history_size"]
            * self.args["common"]["num_cameras"]
            * self.vision_model.num_patches
        )

        _model = RDTRunner(
            action_dim=self.args["common"]["state_dim"],
            pred_horizon=self.args["common"]["action_chunk_size"],
            config=self.args["model"],
            lang_token_dim=self.args["model"]["lang_token_dim"],
            img_token_dim=self.args["model"]["img_token_dim"],
            state_token_dim=self.args["model"]["state_token_dim"],
            max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                ("image", (
                    self.args["common"]["img_history_size"],
                    self.args["common"]["num_cameras"],
                    -self.vision_model.num_patches
                )),
            ],
            lang_pos_embed_config=[
                ("lang", -self.args["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.dtype,
        )
        return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        # --- skip T5 loading if using precomputed embeddings ---
        if pretrained_text_encoder_name_or_path in (None, "precomputed", "skip"):
            print("[INFO] Skipping T5 text encoder — using precomputed embeddings.")
            return None, None
        text_embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path,
            model_max_length=self.args["dataset"]["tokenizer_max_length"],
            device=self.device
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    # ---------------- Quantization (after loading) ----------------
    def apply_quantization(self, mode="4bit", compute_dtype=None, quant_modules: list = None, include_vision=False):
        if not _HAS_BNB:
            print("[WARN] bitsandbytes 가 감지되지 않아 양자화를 건너뜁니다.")
            return
        assert mode in ("4bit", "8bit")
        print(f"\n[DEBUG] Attempting quantization with mode={mode}, modules={quant_modules}, include_vision={include_vision}")

        # dtype 해석
        if compute_dtype is None:
            cdtype = None
        else:
            s = str(compute_dtype).lower()
            if s in ("fp16","float16","torch.float16"):
                cdtype = torch.float16
            elif s in ("bf16","bfloat16","torch.bfloat16"):
                cdtype = torch.bfloat16
            elif s in ("fp32","float32","torch.float32"):
                cdtype = torch.float32
            else:
                cdtype = None
        
        quantized_layers = 0

        def want_quant(name, module):
            print(f"  [DEBUG] Checking module: {name} (type: {type(module).__name__})")
            if not isinstance(module, nn.Linear):
                print(f"    [DEBUG] Skipping: Not a Linear layer.")
                return False
            
            if quant_modules is None or len(quant_modules) == 0:
                print(f"    [DEBUG] Match: No specific modules provided, quantizing all Linear layers.")
                return True # Quantize all if no specific modules are provided

            for pattern in quant_modules:
                if re.search(pattern, name):
                    print(f"    [DEBUG] Match: Found pattern '{pattern}' in module name.")
                    return True
            
            print(f"    [DEBUG] Skipping: Module name does not match any provided patterns.")
            return False

        def convert_linear(module):
            nonlocal quantized_layers
            if mode == "8bit":
                result = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=True,
                    threshold=6.0,
                )
                result.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    result.bias.data.copy_(module.bias.data)
                quantized_layers += 1
                return result
            else:  # 4bit
                quantized_layers += 1
                result = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=cdtype if cdtype is not None else torch.float16,
                    quant_type="nf4",
                )
                result.load_state_dict(module.state_dict())
                return result

        # policy 내부
        print("[DEBUG] --- Processing Policy Model ---")
        for name, mod in list(self.policy.named_modules()):
            for child_name, child in list(mod.named_children()):
                full = f"{name}.{child_name}" if name else child_name
                if want_quant(full, child):
                    print(f"    [DEBUG] Quantizing {full}...")
                    setattr(mod, child_name, convert_linear(child))

        # vision tower 포함할지
        if include_vision and getattr(self, "vision_model", None) is not None:
            print("\n[DEBUG] --- Processing Vision Model ---")
            for name, mod in list(self.vision_model.named_modules()):
                for child_name, child in list(mod.named_children()):
                    full = f"vision.{name}.{child_name}"
                    if want_quant(full, child):
                        print(f"    [DEBUG] Quantizing {full}...")
                        setattr(mod, child_name, convert_linear(child))
        
        print(f"\n[INFO] Finished quantization process for mode={mode}.")
        print(f"[INFO] Total layers quantized: {quantized_layers}")
        print(f"[INFO] Applied bitsandbytes quantization: mode={mode}, quant_modules={quant_modules}, "
              f"compute_dtype={'none' if cdtype is None else cdtype}, include_vision={include_vision}")

    # ---------------- Runtime utils ----------------
    def reset(self):
        device = self.device
        weight_dtype = self.dtype

        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.policy.eval()

        if getattr(self, "text_model", None) is not None:
            self.text_model = self.text_model.to(device, dtype=weight_dtype)
            self.text_model.eval()

        if getattr(self, "vision_model", None) is not None:
            self.vision_model = self.vision_model.to(device, dtype=weight_dtype)
            self.vision_model.eval()

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint = torch.load(pretrained, map_location="cpu")
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def encode_instruction(self, instruction, device="cuda"):
        if self.text_tokenizer is None or self.text_model is None:
            raise RuntimeError("Text encoder disabled. Use precomputed embeddings.")
        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()
        return pred

    # ---------- formatting ----------
    def _format_joint_to_state(self, joints):
        joints = (joints - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state[:, :, MANISKILL_INDICES] = joints

        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, MANISKILL_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        joints = action[:, :, MANISKILL_INDICES]
        joints = (joints + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        return joints

    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        device = self.device
        dtype = self.dtype

        background_color = np.array(
            [int(x * 255) for x in self.image_processor.image_mean],
            dtype=np.uint8
        ).reshape(1, 1, 3)
        background_image = np.ones(
            (self.image_processor.size["height"], self.image_processor.size["width"], 3),
            dtype=np.uint8
        ) * background_color

        image_tensor_list = []
        for image in images:
            if image is None:
                image = Image.fromarray(background_image)

            if self.image_size is not None:
                image = transforms.Resize(self.image_size)(image)

            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, bg_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    if width > height:
                        result = Image.new(pil_img.mode, (width, width), bg_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    result = Image.new(pil_img.mode, (height, height), bg_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))

            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        joints = proprio.to(device).unsqueeze(0)   # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]
        ctrl_freqs = torch.tensor([self.control_frequency], device=device)

        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)
        text_embeds = text_embeds.to(device, dtype=dtype)

        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)
        return trajectory