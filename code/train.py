from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

CLEAN_PATH = Path("outputs/results/questions_clean.csv")
MODEL_PATH = Path("outputs/models/baseline.joblib")

def main():
    if not CLEAN_PATH.exists():
        print("Run: python code/clean.py d'abord (outputs/results/questions_clean.csv manquant).")
        raise SystemExit(1)

    df = pd.read_csv(CLEAN_PATH)

    # On attend deux colonnes : text (X) et label (y)
    if "text" not in df.columns or "label" not in df.columns:
        print("Le fichier clean doit contenir les colonnes 'text' et 'label'.")
        print(f"Colonnes trouvées: {list(df.columns)}")
        raise SystemExit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    print(f"Modèle sauvegardé -> {MODEL_PATH}")
    print(f"Accuracy sur le test: {acc:.3f}")

if __name__ == "__main__":
    main()
