import os
import shutil
from pathlib import Path
from TTS.utils.manage import ModelManager

# ------------------------
# List of models to download
# ------------------------
models = [
    'tts_models/en/ljspeech/tacotron2-DDC',   # English (female)
    'tts_models/es/css10/vits',              # Spanish male (available)
    'vocoder_models/universal/libri-tts/fullband-melgan',  # valid vocoder
]

# ------------------------
# Target directory inside the Docker image
# ------------------------
TARGET_DIR = Path(os.environ.get('TTS_MODEL_PATH', '/app/tts_models'))
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the Coqui TTS model manager
manager = ModelManager()

for m in models:
    try:
        # Preserve hierarchy under TARGET_DIR
        if m.startswith("tts_models/"):
            rel_path = m.replace("tts_models/", "")
            dest = TARGET_DIR / "tts_models" / rel_path
        else:
            # vocoder_models
            rel_path = m.replace("vocoder_models/", "")
            dest = TARGET_DIR / "vocoder_models" / rel_path

        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".download_complete"

        if marker.exists():
            print(f"✅ {m} already marked complete at {dest}, skipping")
            continue

        if any(dest.iterdir()):
            print(f"⚠️ {m} found at {dest} but no completion marker; verifying download")

        print(f"Downloading model {m} ...")
        model_path, config_path, _ = manager.download_model(m)

        src = Path(model_path)

        # Copy model files
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / src.name)

        # Copy config if exists
        if config_path:
            cfg = Path(config_path)
            if cfg.exists():
                shutil.copy2(cfg, dest / cfg.name)

        # Basic verification: check if config.json exists
        expected_files = ['config.json']
        if not any((dest / f).exists() for f in expected_files):
            print(f"⚠️ Warning: {m} copied to {dest} but expected files not found; may be incomplete")

        # Write completion marker
        marker.write_text(f"{m}\n")
        print(f"✅ {m} ready")

    except Exception as e:
        print(f"⚠️ Failed to download {m}: {e}")
        print("⚠️ Skipping and continuing build...")

print("Build completed (with available models)")
