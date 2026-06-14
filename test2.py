import importlib, mujoco
print("mujoco :", mujoco.__version__, " (== 3.8.1 であること)")
for m in ["dm_control", "lerobot", "gym_so100", "transformers", "torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s}:", getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(f"{m:12s}: 未導入/エラー ->", e)
print("\n★ mujoco が 3.8.1 以外、または上で import エラーが出たら、")
print("  メニューから[ランタイムを再起動]→3節(EGL設定)から実行し直してください。")