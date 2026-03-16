import sys

print("Python executable:", sys.executable)

modules = [
    "torch",
    "torchvision",
    "sklearn",
    "matplotlib",
    "pandas",
    "numpy",
    "PIL",
]

for m in modules:
    try:
        mod = __import__(m)
        print(f"{m}: OK")
        if hasattr(mod, "__version__"):
            print(f"  version: {mod.__version__}")
    except Exception as e:
        print(f"{m}: FAILED -> {e}")

try:
    import torchxrayvision as xrv
    print("torchxrayvision: OK")
    print("  version:", xrv.__version__)
except Exception as e:
    print("torchxrayvision: FAILED ->", e)