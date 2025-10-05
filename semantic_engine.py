# - INPUT  : data/user_responses.csv  (column: "response")
# - DATA   : data/competencies.csv, data/job_skills.csv
# - OUTPUT : 
#     outputs/competency_scores.csv     (each competency + similarity score)
#     outputs/block_scores.csv          (average score per competency block)
#     outputs/job_scores.csv            (each job + final score)
#     outputs/results/summary.json      (compact summary for the front-end)
#
# What it does (high-level):
# 1) Reads the user's free-text answers from CSV (1 answer per row in column "response").
# 2) Uses SBERT (SentenceTransformer) to turn texts into vectors (embeddings).
# 3) Computes cosine similarity between the user profile and every competency.
# 4) Aggregates competency scores to produce job scores (Top-K strategy by default).
# 5) Saves detailed CSVs and a small JSON file that the front-end can read directly.
#
# How to run (dev machine, once deps are installed):
#   python semantic_engine.py
#
# Requirements (put these in requirements.txt and install once):
#   sentence-transformers
#   torch
#   pandas
#   numpy
#   matplotlib   (only needed if you later add plots)
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# --------------------------- Configuration (edit here) ------------------------

# Pretrained model used to compute sentence embeddings.
# "all-mpnet-base-v2" = strong, general-purpose, good quality.
MODEL_NAME: str = "all-mpnet-base-v2"

# How we combine multiple user answers before scoring competencies:
#   - "avg": average the user answers into one profile vector (stable)
#   - "max": for each competency, take the strongest match among answers
MODE: str = "avg"

# When scoring jobs, we can reward strongest matches:
#   - For each job, take the Top-K competency scores and average them.
TOP_K: int = 3

# Folder layout (relative paths so it works the same locally and in CI)
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"   # reserved if you add charts later
RES_DIR = OUT_DIR / "results"


# ------------------------------ Helper functions ------------------------------

def _ensure_folders() -> None:
    """Create output folders if they do not exist."""
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load the three inputs needed by the engine:
      - competencies.csv: list of competencies with IDs, text, and block names
      - job_skills.csv : mapping job -> list of competency IDs (long format)
      - user_responses.csv: user's free-text answers (column "response")

    Returns:
      (competencies_df, jobs_df_grouped, user_inputs_list)

    Raises:
      FileNotFoundError if required files are missing.
      ValueError if user_responses.csv lacks a "response" column.
    """
    comp_path = DATA_DIR / "competencies.csv"
    jobs_path = DATA_DIR / "job_skills.csv"
    user_path = DATA_DIR / "user_responses.csv"

    if not comp_path.exists():
        raise FileNotFoundError("Missing data/competencies.csv")
    if not jobs_path.exists():
        raise FileNotFoundError("Missing data/job_skills.csv")
    if not user_path.exists():
        raise FileNotFoundError(
            "Missing data/user_responses.csv (must contain a 'response' column)."
        )

    competencies = pd.read_csv(comp_path)
    jobs_long = pd.read_csv(jobs_path)
    user_df = pd.read_csv(user_path)

    # Normalize headers
    competencies.columns = competencies.columns.str.strip()
    jobs_long.columns = jobs_long.columns.str.strip()
    user_df.columns = user_df.columns.str.strip()

    if "response" not in user_df.columns:
        raise ValueError("CSV must contain a 'response' column.")

    # Keep non-empty text answers only
    user_inputs = (
        user_df["response"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .replace("", np.nan)
        .dropna()
        .tolist()
    )

    # Group the long job table to get one row per job with a list of required competency IDs
    jobs = (
        jobs_long
        .groupby(["JobID", "JobTitle"])["CompetencyID"]
        .apply(list)
        .reset_index()
        .rename(columns={"CompetencyID": "RequiredCompetencies"})
    )

    return competencies, jobs, user_inputs


def encode(model: SentenceTransformer, texts: List[str]):
    """Convert a list of texts into SBERT embeddings (PyTorch tensors)."""
    return model.encode(texts, convert_to_tensor=True)


def compute_comp_scores(user_emb, comp_emb, mode: str = "avg") -> np.ndarray:
    """
    Compute similarity scores between the user profile and each competency.

    Parameters
    ----------
    user_emb : tensor
        Embeddings of all user answers (shape: n_answers x d).
    comp_emb : tensor
        Embeddings of competencies (shape: n_competencies x d).
    mode : str
        "avg" -> average the user's answers into one vector, then compare.
        "max" -> for each competency, take the strongest match across answers.

    Returns
    -------
    np.ndarray
        1D array of size n_competencies with cosine similarity scores in [0,1].
    """
    if mode == "max":
        S = util.cos_sim(user_emb, comp_emb)
        return S.max(dim=0).values.cpu().numpy()
    elif mode == "avg":
        user_avg = user_emb.mean(dim=0, keepdim=True)
        S = util.cos_sim(user_avg, comp_emb)
        return S.squeeze(0).cpu().numpy()
    else:
        raise ValueError("mode must be 'avg' or 'max'")


def score_job_mean(required_ids: List, score_map: dict) -> float:
    """Baseline job score: simple mean of all required competency scores."""
    vals = [score_map.get(cid, 0.0) for cid in required_ids if cid in score_map]
    return float(np.mean(vals)) if vals else 0.0


def score_job_topk(required_ids: List, score_map: dict, k: int = 3) -> float:
    """
    Top-K strategy: focus on the strongest matches only.
    - Sort competency scores for this job descending.
    - Average the best K (or all if < K).

    Useful when user profiles are short (few answers).
    """
    vals = [score_map.get(cid, 0.0) for cid in required_ids if cid in score_map]
    if not vals:
        return 0.0
    vals.sort(reverse=True)
    return float(np.mean(vals[:k]))


# --------------------------------- Main pipeline --------------------------------

def main() -> None:
    """Full pipeline: load data, compute embeddings, score, and save outputs."""
    _ensure_folders()
    print("Loading data...")
    competencies, jobs, user_inputs = load_inputs()
    if not user_inputs:
        raise ValueError("No user responses found in data/user_responses.csv.")

    # Prepare reference data
    cid2block = dict(zip(competencies["CompetencyID"], competencies["BlockName"]))
    comp_ids = competencies["CompetencyID"].tolist()
    comp_texts = competencies["CompetencyText"].astype(str).tolist()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding texts...")
    user_emb = encode(model, user_inputs)
    comp_emb = encode(model, comp_texts)

    print(f"Scoring competencies (mode='{MODE}')...")
    comp_scores = compute_comp_scores(user_emb, comp_emb, mode=MODE)

    # Table of all competencies with their similarity score
    comp_df = (
        pd.DataFrame({
            "CompetencyID": comp_ids,
            "CompetencyText": comp_texts,
            "BlockName": [cid2block[c] for c in comp_ids],
            "Score": comp_scores
        })
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )

    # Average score per block
    block_scores = comp_df.groupby("BlockName")["Score"].mean().sort_values(ascending=False)
    final_coverage = float(block_scores.mean())

    # Prepare job scoring
    score_map = dict(zip(comp_df["CompetencyID"], comp_df["Score"]))

    jobs_mean = jobs.copy()
    jobs_mean["JobScore"] = jobs_mean["RequiredCompetencies"].apply(
        lambda ids: score_job_mean(ids, score_map)
    )

    jobs_topk = jobs.copy()
    jobs_topk["JobScore"] = jobs_topk["RequiredCompetencies"].apply(
        lambda ids: score_job_topk(ids, score_map, k=TOP_K)
    )

    # Use Top-K ranking by default
    jobs_ranked = jobs_topk.sort_values("JobScore", ascending=False).reset_index(drop=True)

    # Save CSVs
    comp_df.to_csv(OUT_DIR / "competency_scores.csv", index=False)
    block_scores.to_csv(OUT_DIR / "block_scores.csv")
    jobs_ranked.to_csv(OUT_DIR / "job_scores.csv", index=False)

    # Save summary JSON for the front-end
    top_job = jobs_ranked.iloc[0] if len(jobs_ranked) else None
    summary = {
        "mode": MODE,
        "top_k": TOP_K,
        "final_coverage": final_coverage,
        "top_job": top_job["JobTitle"] if top_job is not None else None,
        "top_job_score": float(top_job["JobScore"]) if top_job is not None else None,
        "top_competencies": comp_df.head(5)[
            ["CompetencyID", "CompetencyText", "Score"]
        ].to_dict(orient="records")
    }
    with open(RES_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done. Results available in outputs/ and outputs/results/summary.json")


if __name__ == "__main__":
    main()
