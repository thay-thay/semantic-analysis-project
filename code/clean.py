import pandas as pd
from pathlib import Path
data _dir = Path("data")
cat > code/clean.py << 'EOF'
import pandas import pandas

from pathlib import Path
data_dir = Path("data")
out_dir = Path("outputs/results")
out_dir.mkdir(parents=True, exist_ok=True)
src = data_dir / "questions.csv"
if not src.exists():
print("Put questions.csv in data first.")
 raise SystemExit(1)
df = pd.read_csv(src)
# train.py — placeholder for a very simple baseline
cat > code/train.py << 'EOF'
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

clean_path = Path("outputs/results/questions_clean.csv")
model_path = Path("outputs/models/baseline.joblib")

if not clean_path.exists():
    print("Run: python code/clean.py first.")
    raise SystemExit(1)

df = pd.read_csv(clean_path)
if "text" not in df.columns or "label" not in df.columns:
    print("questions_clean.csv must have columns: text and label.")
    raise SystemExit(1)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, model_path)
print(f"Saved model to {model_path} — accuracy: {acc:.3f}")
