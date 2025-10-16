from pathlib import Path
import json
import pandas as pd

def main():
    out_dir = Path("./outputs")
    feats = pd.read_csv(out_dir / "features_all.csv", nrows=10)
    report = json.load(open(out_dir / "training_report.json"))
    print("=== Sample features ===")
    print(feats.head(5))
    print("\n=== Training report ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
