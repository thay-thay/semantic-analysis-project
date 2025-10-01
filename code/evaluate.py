from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report

clean_path = Path("outputs/results/questions_clean.csv")
model_path = Path("outputs/models/baseline.joblib")

if not clean_path.exists() or not model_path.exists():
    print("Run clean.py then train.py first.")
    raise SystemExit(1)

df = pd.read_csv(clean_path)
X, y = df["text"], df["label"]
model = joblib.load(model_path)
pred = model.predict(X)
print(classification_report(y, pred))
