#!/usr/bin/env python3
import sys
from huggingface_hub import snapshot_download

def download_hf_model():
    if len(sys.argv) < 3:
        print("Usage:\n")
        print("  python download_hf_model.py <repo_id> <local_dir> [<pattern1> <pattern2> ...]\n")
        print("Examples:\n")
        print("  # Download the entire repository")
        print("  python download_hf_model.py robotics-diffusion-transformer/maniskill-model ./pretrained_models\n")
        print("  # Download specific folders only (add / at the end of folder names)")
        print("  python download_hf_model.py robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt/ octo/ diffusion_policy/ lang_embeds/\n")
        print("  # Download specific files only (can use wildcards *)")
        print("  python download_hf_model.py robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt/mp_rank_00_model_states.pt openvla-7b*/config.json\n")
        sys.exit(1)

    repo_id = sys.argv[1]
    local_dir = sys.argv[2]
    patterns = sys.argv[3:]

    try:
        allow_patterns = []
        if patterns:
            for p in patterns:
                if p.endswith("/"):
                    allow_patterns.append(f"{p}**")
                else:
                    allow_patterns.append(p)
            print(f"Downloading selected folders/files from {repo_id}: {', '.join(patterns)}")
        else:
            allow_patterns = None
            print(f"Downloading entire model from {repo_id}")

        local_folder = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="model",
            allow_patterns=allow_patterns
        )

        print(f"\nDownload complete: {local_folder}")

    except Exception as e:
        print("\nError occurred during download:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    download_hf_model()
