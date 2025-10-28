#!/usr/bin/env python3
import sys
from huggingface_hub import snapshot_download

def download_hf_model():
    if len(sys.argv) < 3:
        print("Usage:\n")
        print("  python download_hf_model.py <repo_id> <local_dir> [<folder1> <folder2> ...]")
        print("\nExamples:\n")
        print("  python download_hf_model.py robotics-diffusion-transformer/maniskill-model ./pretrained_models")
        print("  python download_hf_model.py robotics-diffusion-transformer/maniskill-model ./pretrained_models rdt octo diffusion_policy lang_embeds \"openvla-7b*\"")
        sys.exit(1)

    repo_id = sys.argv[1]
    local_dir = sys.argv[2]
    folders = sys.argv[3:]

    try:
        if folders:
            allow_patterns = [f"{folder}/**" for folder in folders]
            print(f"Downloading selected folders from {repo_id}: {', '.join(folders)}")
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
