import json
import os
import pandas as pd
from collections import Counter
from pathlib import Path

# CONFIG
LOG_DIR = Path("trivalaya_data/04_corrections")
CORRECTIONS_FILE = LOG_DIR / "corrections_log.jsonl"
CONFIRMATIONS_FILE = LOG_DIR / "feedback_log.jsonl"

def load_logs():
    data = []
    
    # 1. Load Corrections (Model was WRONG)
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry["result"] = "WRONG"
                    # Flatten the structure for DataFrame
                    entry["predicted"] = entry["original_prediction"]
                    entry["actual"] = entry["corrected_label"]
                    entry["confidence"] = entry["original_confidence"]
                    data.append(entry)
                except json.JSONDecodeError:
                    continue

    # 2. Load Confirmations (Model was RIGHT)
    if CONFIRMATIONS_FILE.exists():
        with open(CONFIRMATIONS_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry["result"] = "CORRECT"
                    entry["predicted"] = entry["label"]
                    entry["actual"] = entry["label"]
                    entry["confidence"] = entry["confidence"]
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return pd.DataFrame(data)

def analyze():
    df = load_logs()
    
    if df.empty:
        print("❌ No feedback logs found yet. Go use the app!")
        return

    print("\n📊 TRIVALAYA LIVE FEEDBACK REPORT")
    print("=" * 40)

    # 1. High-Level Metrics
    total = len(df)
    correct = len(df[df["result"] == "CORRECT"])
    accuracy = (correct / total) * 100
    
    print(f"Total Reviewed:    {total}")
    print(f"Live Accuracy:     {accuracy:.1f}%")
    print("-" * 40)

    # 2. The "Confusion" List (Where is it failing?)
    errors = df[df["result"] == "WRONG"]
    if not errors.empty:
        print("\n📉 Top Failures (Predicted -> Actual):")
        confusion = errors.groupby(["predicted", "actual"]).size().reset_index(name="count")
        confusion = confusion.sort_values("count", ascending=False)
        
        for _, row in confusion.iterrows():
            print(f"  ❌ {row['predicted']:<18} ➔ {row['actual']:<18} : {row['count']} times")
            
        # 3. Scary Errors (High Confidence but WRONG)
        scary = errors[errors["confidence"] > 0.85]
        if not scary.empty:
            print("\n⚠️  DANGEROUS ERRORS (Model was >85% confident):")
            for _, row in scary.iterrows():
                print(f"  • {row['filename']} (Said {row['predicted']} @ {row['confidence']*100:.0f}%, actually {row['actual']})")
    else:
        print("\n✅ No errors logged yet!")

    # 4. Class Performance
    print("\n📈 Performance by Class (Live):")
    class_stats = df.groupby("actual")["result"].value_counts(normalize=True).unstack().fillna(0)
    if "CORRECT" in class_stats.columns:
        print(class_stats["CORRECT"].sort_values(ascending=False).apply(lambda x: f"{x*100:.0f}%"))
    else:
        print("No correct predictions yet.")

if __name__ == "__main__":
    analyze()