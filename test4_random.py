# --- まずランダム行動で「MuJoCo + gym-so100 + Colab描画」が成立するか確認 ---
# ここが動けばシミュレーションと描画は正常。ポリシーは次節(8節)で接続する。
import gymnasium as gym
import gym_so100          # 環境登録のため import が必要
import numpy as np
import mediapy as media

# gym-so100 は "gym_so100/" 名前空間を付けずに素のIDで登録している。実際のIDを確認:
so100_ids = [k for k in gym.registry.keys() if "SO100" in k]
print("登録済み SO100 環境:", so100_ids)

# 末尾が EE のものはエンドエフェクタ制御、それ以外は関節制御
ENV_ID = "SO100EETransferCube-v0"   # ←名前空間プレフィックス無しが正解
env = gym.make(
    ENV_ID,
    obs_type="pixels_agent_pos",
    render_mode="rgb_array",
)
obs, info = env.reset(seed=0)

# ★ここで実際の観測・行動の構造を確認しておく（8節の調整に必須）
print("action_space :", env.action_space)
print("obs keys     :", list(obs.keys()))
print("camera keys  :", list(obs["pixels"].keys()))
print("agent_pos dim:", np.asarray(obs["agent_pos"]).shape)

frames = []
for _ in range(120):
    action = env.action_space.sample()   # ランダム行動
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        obs, info = env.reset()
env.close()

media.show_video(frames, fps=25)
print("OK: gym-so100 のシミュレーションを EGL 描画できました。")