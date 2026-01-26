import subprocess
import sys
import os
from pathlib import Path

# =========================================================
# SILENT PIPELINE ORCHESTRATOR (NO SCRIPT MODIFICATION)
# =========================================================

def run_script(script_path: Path, capture_output=False, show_live_output=False):
    env = os.environ.copy()

    # 🔥 FORCE UTF-8 (prevents emoji crash on Windows)
    env["PYTHONUTF8"] = "1"

    if show_live_output:
        # For long-running scripts, show live progress
        import threading
        
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=0  # Unbuffered
        )
        
        output_lines = []
        
        # Read output line by line and show progress indicators
        for line in iter(process.stdout.readline, ''): # type: ignore
            if not line:
                break
            output_lines.append(line)
            line_stripped = line.strip()
            
            # Show key progress indicators
            if "Loading Model" in line_stripped:
                print("\n    ⏳ Loading model...", flush=True)
            elif "Model loaded" in line_stripped or "Model ready" in line_stripped:
                print("    ✓ Model loaded", flush=True)
            elif "Processing Batches" in line_stripped:
                print("    ⏳ Processing batches...", flush=True)
            elif "Running prediction" in line_stripped:
                print("    ⏳ Generating predictions...", flush=True)
            elif "PROCESSING COMPLETE" in line_stripped:
                print("    ✓ Processing complete", flush=True)
            elif "%" in line_stripped and ("█" in line_stripped or "|" in line_stripped or "it/s" in line_stripped or "s/it" in line_stripped):
                # Progress bar detected (tqdm format)
                print(f"    {line_stripped}", flush=True)
        
        process.wait()
        
        # Create a result object similar to subprocess.run
        class Result:
            def __init__(self, returncode, stdout):
                self.returncode = returncode
                self.stdout = stdout
        
        return Result(process.returncode, ''.join(output_lines))
    else:
        return subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of crashing
            capture_output=capture_output
        )


def main():
    LIVE_DEMO_DIR = Path(__file__).resolve().parent
    SCRIPTS_DIR = LIVE_DEMO_DIR / "scripts"

    script_01 = SCRIPTS_DIR / "01_convert_trojon_aligned.py"
    script_02 = SCRIPTS_DIR / "02_run_main_with config.py"
    script_03 = SCRIPTS_DIR / "03_json_to_csv.py"
    script_04 = SCRIPTS_DIR / "04_live_strategy_executor.py"

    pipeline = [script_01, script_02, script_03, script_04]
    
    script_names = [
        "Converting to Trojan-Aligned format",
        "Running Kronos prediction model",
        "Converting predictions to CSV",
        "Executing trading strategy"
    ]

    for script in pipeline:
        if not script.exists():
            print(f"ERROR: Missing script {script}")
            sys.exit(1)

    print("=" * 60)
    print("🚀 KRONOS PIPELINE RUNNING")
    print("=" * 60)

    final_output = ""

    # Run all scripts silently except the final one
    for idx, script in enumerate(pipeline):
        is_final = (script == script_04)
        is_kronos_step = (script == script_02)  # The long-running model step
        
        print(f"[{idx + 1}/4] {script_names[idx]}...", end="" if is_kronos_step else " ", flush=True)

        result = run_script(
            script,
            capture_output=True,
            show_live_output=is_kronos_step  # Show live progress for Kronos step
        )

        if result.returncode != 0:
            print("❌ FAILED")
            print("PIPELINE FAILED")
            sys.exit(1)

        if not is_kronos_step:
            print("✅")
        else:
            print("\n    ✅ Kronos model complete")

        if is_final:
            final_output = result.stdout.strip()

    print("=" * 60)
    print()

    # -----------------------------------------------------
    # ONLY FINAL RESULT IS PRINTED
    # -----------------------------------------------------
    if final_output:
        print(final_output)
    else:
        print("No output from executor")


if __name__ == "__main__":
    main()