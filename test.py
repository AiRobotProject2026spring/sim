import os

# NVIDIA の EGL ベンダ設定（ICD）を書き込む（公式チュートリアルと同じ内容）
NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
    with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
        f.write('{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}')

# 描画バックエンドを EGL に設定（import mujoco より前）
os.environ["MUJOCO_GL"] = "egl"
print("MUJOCO_GL =", os.environ["MUJOCO_GL"])



import mujoco
import matplotlib.pyplot as plt

print("MuJoCo   :", mujoco.__version__)
print("MUJOCO_GL:", __import__("os").environ.get("MUJOCO_GL"))

# (2) 最小の有効な MJCF モデルをコンパイル
mujoco.MjModel.from_xml_string("<mujoco/>")
print("OK: 空のモデルをコンパイルできました。")

# 真っ黒にならないようライトを入れた小さなシーン
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# (3) オフスクリーン描画。EGL バックエンドが壊れているとここで例外になる。
with mujoco.Renderer(model) as renderer:
    mujoco.mj_forward(model, data)   # 描画前に派生量（Cartesian 座標など）を伝播：必須
    renderer.update_scene(data)
    pixels = renderer.render()

print("OK: shape", pixels.shape, "のフレームを描画しました。")
plt.imshow(pixels)
plt.axis("off")
plt.show()
print("\nMuJoCo の環境構築が完了しました。")