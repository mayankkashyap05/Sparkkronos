import os
import subprocess
import sys
import shutil

def safe_clear_path(path):
    if os.path.isfile(path):
        os.remove(path)
        print(f"🧹 Deleted file: {path}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"🧹 Deleted directory: {path}")

def main():
    print("=" * 70)
    print("🚀 RUNNING MAIN.PY (FORCED OVERWRITE MODE)")
    print("=" * 70)

    # -------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LIVE_DEMO_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(LIVE_DEMO_DIR))

    MAIN_PY = os.path.join(PROJECT_ROOT, "main.py")
    SOURCE_CONFIG = os.path.join(SCRIPT_DIR, "config.json")
    TARGET_CONFIG = os.path.join(PROJECT_ROOT, "config.json")

    if not os.path.exists(MAIN_PY):
        raise FileNotFoundError(f"❌ main.py not found: {MAIN_PY}")

    if not os.path.exists(SOURCE_CONFIG):
        raise FileNotFoundError(f"❌ config.json not found: {SOURCE_CONFIG}")

    # -------------------------------------------------------------
    # Load config to know output paths
    # -------------------------------------------------------------
    import json
    with open(SOURCE_CONFIG, "r") as f:
        config = json.load(f)

    summary_path = os.path.join(PROJECT_ROOT, config["output"]["summary_file"])
    predictions_dir = os.path.join(PROJECT_ROOT, config["output"]["prediction_files_dir"])

    # -------------------------------------------------------------
    # FORCE CLEAN OUTPUTS (🔥 THIS IS THE FIX)
    # -------------------------------------------------------------
    print("\n🧨 Clearing previous outputs...")
    safe_clear_path(summary_path)
    safe_clear_path(predictions_dir)

    # -------------------------------------------------------------
    # Backup existing root config.json
    # -------------------------------------------------------------
    backup_path = None
    if os.path.exists(TARGET_CONFIG):
        backup_path = TARGET_CONFIG + ".bak"
        shutil.copy2(TARGET_CONFIG, backup_path)

    try:
        # Inject correct config
        shutil.copy2(SOURCE_CONFIG, TARGET_CONFIG)
        print("✅ Injected config.json into project root")

        # Run main.py
        print("\n▶ Launching pipeline...\n")
        subprocess.run(
            [sys.executable, MAIN_PY],
            cwd=PROJECT_ROOT,
            check=True
        )

        print("\n" + "=" * 70)
        print("✅ PIPELINE FINISHED (FILES OVERWRITTEN)")
        print("=" * 70)

    finally:
        # Restore config.json
        if backup_path and os.path.exists(backup_path):
            shutil.move(backup_path, TARGET_CONFIG)
        elif os.path.exists(TARGET_CONFIG):
            os.remove(TARGET_CONFIG)

if __name__ == "__main__":
    main()
