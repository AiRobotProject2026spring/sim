# --- π0(pi0_base) ポリシーの読み込み（約14GB。数分かかります）---
import torch
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from transformers import AutoTokenizer
import mediapy as media
import gymnasium as gym
import gym_so100      

# 再起動などで 6節(MODEL_ID 定義)を飛ばしていても動くようにする安全ガード
try:
    MODEL_ID
except NameError:
    MODEL_ID = "lerobot/pi0_base"
print("MODEL_ID:", MODEL_ID, "(未定義なら from_pretrained が自動DLします)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

TASK_ID = "SO100EETransferCube-v0"   # 名前空間プレフィックス無しが正解
TASK_TEXT = "pick up the cube"   # 微調整時のタスク文に合わせて変更
STEPS = 120
TOKENIZER_MAX_LENGTH = 48


def load_paligemma_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", use_fast=False)
    except OSError as exc:
        raise RuntimeError(
            "PaliGemma tokenizer を読み込めませんでした。pi0_base は task 文字列を "
            "observation.language.tokens に変換する必要があります。\n"
            "https://huggingface.co/google/paligemma-3b-pt-224 で利用条件に同意し、"
            "`pixi run huggingface-cli login` または HF_TOKEN を設定してから再実行してください。"
        ) from exc


tokenizer = load_paligemma_tokenizer()

# メモリ節約のため bf16 で読み込む（T4 では fp32 だと 14GB で OOM しやすい）
try:
    policy = PI0Policy.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
except TypeError:
    # 古い/新しい lerobot で torch_dtype を受け取らない場合のフォールバック
    policy = PI0Policy.from_pretrained(MODEL_ID)

policy = policy.to(device)
policy.eval()
if hasattr(policy, "reset"):
    policy.reset()

print("π0(pi0_base) を読み込みました。")
print("入力特徴:", getattr(getattr(policy, "config", None), "input_features", "config参照"))
# CUDA out of memory の場合は L4/A100 ランタイムに変更してください。


# --- π0(pi0_base) で配線確認: 観測→ポリシー→行動→sim を一周させる ---
# ★ pi0_base は未微調整なので動きは無意味。ここは「エラーなく一周するか」の確認。
import numpy as np
import torch
import torch.nn.functional as F

model_dtype = next(policy.parameters()).dtype


def to_img(img_hwc):
    """(H,W,3)uint8 -> (1,3,224,224) float[0,1] をモデルのdtype/deviceで返す。"""
    t = torch.from_numpy(np.asarray(img_hwc)).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return t.to(device=device, dtype=model_dtype)


def build_batch(obs):
    """sim 観測 -> pi0_base の入力(カメラ3つ + state[32] + task)へ詰め替え。"""
    base = to_img(obs["pixels"]["top"])     # top をベースカメラに
    wrist = to_img(obs["pixels"]["angle"])  # angle を左手首カメラに
    # sim はカメラ2つしかないので右手首は top を流用（要調整）
    state = np.zeros(32, dtype=np.float32)
    state[:6] = np.asarray(obs["agent_pos"], dtype=np.float32)[:6]
    state_t = torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=model_dtype)
    lang = tokenizer(
        [TASK_TEXT if TASK_TEXT.endswith("\n") else f"{TASK_TEXT}\n"],
        padding="max_length",
        max_length=getattr(policy.config, "tokenizer_max_length", TOKENIZER_MAX_LENGTH),
        truncation=True,
        return_tensors="pt",
    )
    return {
        "observation.images.base_0_rgb": base,
        "observation.images.left_wrist_0_rgb": wrist,
        "observation.images.right_wrist_0_rgb": base,
        "observation.state": state_t,
        "observation.language.tokens": lang["input_ids"].to(device=device),
        "observation.language.attention_mask": lang["attention_mask"].to(device=device, dtype=torch.bool),
    }


env = gym.make(TASK_ID, obs_type="pixels_agent_pos", render_mode="rgb_array")
obs, info = env.reset(seed=0)
act_dim = int(np.prod(env.action_space.shape))
print("sim action_dim:", act_dim)

frames = []
for step in range(STEPS):
    with torch.no_grad():
        raw = policy.select_action(build_batch(obs))   # 近年の pi0 は1ステップ分を返す
    a = np.asarray(raw.detach().float().cpu().numpy()).reshape(-1)   # (>=6,)
    # action の先頭6要素を関節指令に。未微調整でスケール不明なので安全に [-1,1] へクリップ。
    cmd6 = np.clip(a[:6], -1.0, 1.0).astype(np.float32)
    sim_a = np.zeros(act_dim, dtype=np.float32)
    sim_a[:min(6, act_dim)] = cmd6[:min(6, act_dim)]

    obs, reward, terminated, truncated, info = env.step(sim_a)
    frames.append(env.render())
    if step % 20 == 0:
        print(f"  step {step:>3}: reward={reward:.3f}")
    if terminated or truncated:
        obs, info = env.reset()
env.close()

media.show_video(frames, fps=25)
print("配線確認 完了（動きは無意味でOK）。意味あるタスクには 9節の微調整が必要です。")
