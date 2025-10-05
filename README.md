# semantic-analysis-project
# Semantic engine

## Input
Place a CSV at data/user_responses.csv with a column named response.

Example:
response
I worked with Python and SQL for analytics
I used Power BI for dashboards and some NLP

## Run
This project uses Sentence Transformers.
Install dependencies (once), then run the engine to generate outputs:
- outputs/competency_scores.csv
- outputs/block_scores.csv
- outputs/job_scores.csv
- outputs/results/summary.json

## Front integration
Front only needs to write data/user_responses.csv
Then read outputs/results/summary.json to display the recommended job and top competencies.
