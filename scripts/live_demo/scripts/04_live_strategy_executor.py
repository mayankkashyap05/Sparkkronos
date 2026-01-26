import pandas as pd
from pathlib import Path

# =========================================================
# LIVE STRATEGY (AUTO MODE — FILE BASED)
# =========================================================

def execute_live_strategy_from_csv(csv_path: Path):
    """
    Reads prediction.csv, takes the latest row, and decides whether to BUY
    based on the Dip Hunter strategy (Long Only).

    Strategy Rules:
    1. Setup: last_value_open > 0.70  (Inverted RSI → Oversold)
    2. Trigger: Sum of predicted_close_* columns > 0
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"❌ CSV file not found:\n{csv_path}")

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("❌ prediction.csv is empty")

    # Use the LAST row (most recent prediction)
    row = df.iloc[-1]

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------
    if "last_value_open" not in df.columns:
        raise ValueError("❌ Missing column: last_value_open")

    pred_close_cols = [c for c in df.columns if c.startswith("predicted_close_")]

    if not pred_close_cols:
        raise ValueError("❌ No predicted_close_* columns found")

    # -----------------------------------------------------
    # Extract signals
    # -----------------------------------------------------
    dip_score = float(row["last_value_open"])
    predicted_return = float(row[pred_close_cols].sum())

    # -----------------------------------------------------
    # Strategy Logic
    # -----------------------------------------------------
    is_oversold = dip_score > 0.70
    model_says_up = predicted_return > 0

    decision = "NO TRADE"
    reason = ""

    if is_oversold:
        if model_says_up:
            decision = "BUY"
            reason = "Oversold + Model Confirms Bounce"
        else:
            reason = "Oversold but Model Predicts Drop (Falling Knife)"
    else:
        reason = "Market not Oversold (Score ≤ 0.70)"

    # -----------------------------------------------------
    # Output (Live Friendly)
    # -----------------------------------------------------
    print("\n" + "=" * 50)
    print("🤖 LIVE STRATEGY DECISION")
    print("=" * 50)
    print(f"Source file:      {csv_path.name}")
    print(f"Dip score:        {dip_score:.4f}  (threshold > 0.70)")
    print(f"Predicted return:{predicted_return:.6f}")
    print("-" * 50)
    print(f"DECISION: {decision}")
    print(f"REASON:   {reason}")
    print("=" * 50 + "\n")

    return decision


# =========================================================
# ENTRY POINT (NO USER INPUT)
# =========================================================
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    LIVE_DEMO_DIR = SCRIPT_DIR.parent
    PROJECT_ROOT = LIVE_DEMO_DIR.parent.parent

    PREDICTION_CSV = (
        PROJECT_ROOT
        / "scripts"
        / "live_demo"
        / "data"
        / "result"
        / "predictions"
        / "prediction.csv"
    )

    execute_live_strategy_from_csv(PREDICTION_CSV)
