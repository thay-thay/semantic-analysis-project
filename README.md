# Semantic Analysis Project  

> Automatically matches user responses to relevant **competencies** and **job profiles**,  
> using **Sentence-BERT embeddings (all-mpnet-base-v2)** and **cosine similarity**, powered by **GitHub Actions CI/CD**.  

---

## Overview  

The **Semantic Engine** processes user inputs from the Streamlit app, computes semantic similarities with reference datasets,  
and generates job recommendations automatically.  

Whenever a new response is submitted, **GitHub Actions** triggers the engine, runs the full pipeline,  
and publishes updated results directly to the repository for visualization and front-end integration.  

---

## How It Works  

### Data Flow  
```

Streamlit App → pushes user_responses.csv
→ GitHub Actions triggers the engine
→ SBERT (all-mpnet-base-v2) computes embeddings
→ Cosine similarity scores generated
→ CSV + JSON results pushed back to GitHub
→ Front-end fetches results live

````

### Main Components  

| File | Description |
|------|--------------|
| `semantic_engine.py` | Core engine – loads data, encodes text, computes scores, and saves outputs |
| `data/` | Input files (`competencies.csv`, `job_skills.csv`, `user_responses.csv`) |
| `outputs/` | Auto-generated results (`competency_scores.csv`, `block_scores.csv`, `job_scores.csv`, `summary.json`) |
| `.github/workflows/semantic-engine.yml` | GitHub Actions workflow automation (CI/CD) |
| `app.py` | Streamlit app – collects user responses and triggers engine |
| `code/` | Baseline scripts (TF-IDF, logistic regression) for comparison |

---

## Model Design  

### Chosen Approach  
- **Model:** Sentence-BERT (`all-mpnet-base-v2`)  
- **Similarity:** Cosine similarity  
- **Aggregation:** Block-level → Job-level ranking  

### Why `all-mpnet-base-v2`?  
`all-mpnet-base-v2` provides **state-of-the-art performance** on semantic similarity tasks,  
capturing **context, rephrasing, and nuanced meanings** far better than traditional models.

| Metric | TF-IDF | SBERT (all-mpnet-base-v2) |
|:-------|:--------|:---------------------------|
| Synonym / rephrase handling | ❌ | ✅ |
| Context understanding | ❌ | ✅ |
| Computation time | Fast | Moderate |
| Accuracy | Medium | **Very High** |

---

##  CI/CD Automation  

Whenever `user_responses.csv` or `semantic_engine.py` changes, **GitHub Actions** runs automatically.  

### Workflow Summary  
1.  **Set up Python 3.10 environment**  
2.  **Install dependencies** from `requirements.txt`  
3.  **Run** the semantic engine (`semantic_engine.py`)  
4.  **Generate outputs** (competency, block, job, and summary files)  
5.  **Commit & push** results to the repo  
6.  **Upload artifacts** for backup or download  

### Produced Files  

| Output | Purpose |
|--------|----------|
| `competency_scores.csv` | Similarity scores per competency |
| `block_scores.csv` | Aggregated mean scores per block |
| `job_scores.csv` | Top-K ranked job matches |
| `results/summary.json` | Compact summary used by front-end |

Example:
```json
{
  "mode": "cosine",
  "final_coverage": 0.83,
  "top_job": "Data Scientist",
  "top_job_score": 0.91,
  "top_competencies": [
    {"CompetencyText": "Analytical thinking", "Score": 0.94}
  ]
}
````

---

##  Pipeline & Data Quality

The pipeline ensures **data consistency and reproducibility** through each stage:
 Ingestion → Cleaning → Embedding → Scoring → Export

**Text normalization steps:**

* Lowercasing
* Tokenization
* Lemmatization
* Stopword removal
* Multi-language support (EN/FR)

Inline comments and docstrings describe every key function in the code.

---

##  Baseline Comparison

| Model                         | Technique           | Result | Note                                 |
| ----------------------------- | ------------------- | ------ | ------------------------------------ |
| **TF-IDF**                    | Lexical similarity  | ⚪⚪⚪⚫⚫  | Fails on paraphrasing                |
| **SBERT (all-mpnet-base-v2)** | Semantic embeddings | ⚪⚪⚪⚪⚪  | State-of-the-art contextual accuracy |

---

##  Folder Structure

```
semantic-analysis-project/
├── app.py
├── semantic_engine.py
├── run_engine.py
├── requirements.txt
├── data/
│   ├── competencies.csv
│   ├── job_skills.csv
│   └── user_responses.csv
├── outputs/
│   ├── competency_scores.csv
│   ├── block_scores.csv
│   ├── job_scores.csv
│   └── results/summary.json
├── code/
│   ├── clean.py
│   ├── evaluate.py
│   └── train.py
└── .github/workflows/semantic-engine.yml
```

---

## Key Strengths

 Fully automated semantic pipeline
 Real-time synchronization between app & model
 Transparent & reproducible outputs
 SBERT baseline vs `all-mpnet-base-v2` comparison
 Lightweight CI/CD integration (GitHub Actions)

---

## Installation

To run locally:

```bash
pip install -r requirements.txt
python semantic_engine.py
```

**Main Dependencies:**

```
sentence-transformers
pandas
numpy
scikit-learn
matplotlib
qrcode[pil]
```

---

## Live Streamlit App
 **Launch App:**
[semantic-analysis-project.streamlit.app](https://semantic-analysis-project.streamlit.app)

 **Scan the QR code below to open:**

<img width="210" height="210" alt="image" src="https://github.com/user-attachments/assets/825d6744-f0c3-4a23-8620-4b6cb8c60f35" />


---

## Team

Ikram AMINE, Thayri BOUAICH, Victor CHEVALLIER, Corentin COFFRE, Vincent HASCOAT, Adeline EL BOUHOUTI, Geoffroy BOCCAN-LIAUDET.

**Institution:** ECE Paris – Data & AI
**Mentor:** MALAEB Sarah

---

## License

This project is for **academic and educational purposes only**.
Commercial reproduction requires prior authorization.

```
