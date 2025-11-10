import os
import glob
import shutil
import kagglehub

# Download latest version of the model from Kaggle
path = kagglehub.model_download("noorsaeed/mri_brain_tumor_model/keras/default")

print("Path to model files:", path)

# Ensure models/ directory exists in the project root
os.makedirs("models", exist_ok=True)

# Look recursively for any .h5 file inside the downloaded path
pattern = os.path.join(path, "**", "*.h5")
h5_files = glob.glob(pattern, recursive=True)

if h5_files:
    src = h5_files[0]  # [Inference] نستخدم أول ملف .h5 نراه
    dst = os.path.join("models", "model.h5")
    shutil.copy(src, dst)
    print(f"Copied model file:\n  {src}\n→ {dst}")
else:
    print("No .h5 file found inside the downloaded model path.")
    print("Contents of the downloaded folder:")
    for root, dirs, files in os.walk(path):
        print(f"[DIR] {root}")
        for f in files:
            print(f"    - {f}")
