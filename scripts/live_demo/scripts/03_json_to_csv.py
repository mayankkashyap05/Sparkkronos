import json
import csv
import os
from pathlib import Path

def main():
    print("=" * 70)
    print("📄 AUTO JSON → CSV CONVERTER (SINGLE FILE MODE)")
    print("=" * 70)

    # -------------------------------------------------------------
    # Resolve paths (relative, robust)
    # -------------------------------------------------------------
    SCRIPT_DIR = Path(__file__).resolve().parent
    LIVE_DEMO_DIR = SCRIPT_DIR.parent
    PROJECT_ROOT = LIVE_DEMO_DIR.parent.parent

    PREDICTIONS_DIR = PROJECT_ROOT / "scripts" / "live_demo" / "data" / "result" / "predictions"
    INPUT_JSON = PREDICTIONS_DIR / "prediction.json"
    OUTPUT_CSV = PREDICTIONS_DIR / "prediction.csv"

    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"❌ JSON file not found:\n{INPUT_JSON}")

    print(f"📥 Input JSON : {INPUT_JSON}")
    print(f"📤 Output CSV : {OUTPUT_CSV}")

    # -------------------------------------------------------------
    # Load JSON
    # -------------------------------------------------------------
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------------------------------------------
    # Extract last values
    # -------------------------------------------------------------
    last_values = data.get("input_data_summary", {}).get("last_values", {})

    actual_data = data.get("actual_data", [])
    prediction_data = data.get("prediction_results", [])

    max_n = max(len(actual_data), len(prediction_data))

    # -------------------------------------------------------------
    # Generate headers
    # -------------------------------------------------------------
    headers = [
        "last_value_open",
        "last_value_high",
        "last_value_low",
        "last_value_close",
        "last_value_volume",
    ]

    for i in range(1, max_n + 1):
        headers.extend([
            f"actual_open_{i}", f"predicted_open_{i}",
            f"actual_high_{i}", f"predicted_high_{i}",
            f"actual_low_{i}", f"predicted_low_{i}",
            f"actual_close_{i}", f"predicted_close_{i}",
            f"actual_volume_{i}", f"predicted_volume_{i}",
        ])

    # -------------------------------------------------------------
    # Build row
    # -------------------------------------------------------------
    row = [
        last_values.get("open", ""),
        last_values.get("high", ""),
        last_values.get("low", ""),
        last_values.get("close", ""),
        last_values.get("volume", ""),
    ]

    for i in range(max_n):
        if i < len(actual_data):
            a = actual_data[i]
        else:
            a = {}

        if i < len(prediction_data):
            p = prediction_data[i]
        else:
            p = {}

        row.extend([
            a.get("open", ""), p.get("open", ""),
            a.get("high", ""), p.get("high", ""),
            a.get("low", ""),  p.get("low", ""),
            a.get("close", ""), p.get("close", ""),
            a.get("volume", ""), p.get("volume", ""),
        ])

    # -------------------------------------------------------------
    # Write CSV (overwrite mode)
    # -------------------------------------------------------------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(row)

    print("=" * 70)
    print("✅ CONVERSION COMPLETE (FILE OVERWRITTEN)")
    print("=" * 70)

if __name__ == "__main__":
    main()
