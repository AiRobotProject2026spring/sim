# --- π0 ベースモデル(lerobot/pi0_base, 約14GB)の事前ダウンロード ---
# from_pretrained でも自動取得されるが、ここで先に落として進捗を確認しておく。
from huggingface_hub import snapshot_download

MODEL_ID = "lerobot/pi0_base"

CKPT_DIR = snapshot_download(
    repo_id=MODEL_ID,
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "policy_preprocessor*",
        "policy_postprocessor*",
        "*.json",
    ],
)

import os
print("MODEL_ID :", MODEL_ID)
print("ローカル :", CKPT_DIR)
for f in sorted(os.listdir(CKPT_DIR)):
    size = os.path.getsize(os.path.join(CKPT_DIR, f)) / 1e6
    print(f"  {f}  ({size:.1f} MB)")