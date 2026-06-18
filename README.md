# Predicting 30-Day Hospital Readmission Risk

A healthcare analytics portfolio project that demonstrates a baseline workflow for preparing clinical encounter data, training a predictive model, and framing results for care coordination and follow-up planning.

This project uses the publicly available Diabetes 130-US hospitals dataset to explore whether patient encounter data can help identify patients at higher risk of 30-day hospital readmission.

## Project Status

**Status:** Baseline technical demonstration / portfolio project

This repository is intended to show healthcare analytics thinking, Python workflow structure, model development, and responsible communication of limitations. It is not a deployed clinical decision-support tool.

## Healthcare Problem

Thirty-day hospital readmissions are costly and can signal gaps in discharge planning, follow-up care, medication management, or patient support. A practical analytics workflow can help care teams prioritize outreach, follow-up calls, education, and additional support for patients who may be at higher risk.

## Dataset

**Dataset:** Diabetes 130-US hospitals for years 1999-2008

The dataset includes more than 100,000 hospital encounters for patients with diabetes. Records include demographics, admission type, discharge disposition, diagnoses, medications, and readmission status.

## Tools and Technologies

- Python
- pandas
- scikit-learn
- matplotlib
- Logistic regression
- Git/GitHub

## What This Project Demonstrates

- Healthcare problem framing
- Data ingestion and preprocessing
- Feature preparation for a baseline model
- Logistic regression model training
- Model evaluation workflow
- Responsible discussion of model limitations
- Reproducible project structure for portfolio review

## Repository Contents

- `src/data.py` - downloads and preprocesses the dataset into a clean dataframe
- `src/model.py` - trains a baseline logistic regression model and outputs evaluation metrics
- `requirements.txt` - Python dependencies
- `LICENSE` - MIT license

## How to Run

```bash
git clone https://github.com/Juan-R1/healthcare-readmission-risk.git
cd healthcare-readmission-risk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data.py
python src/model.py
```

## Expected Output

Running the model script should train a baseline readmission-risk model and print evaluation metrics. The goal is not to claim a production-ready model, but to demonstrate a clean, explainable healthcare analytics workflow.

## Portfolio Value

This project is most relevant for roles such as:

- Healthcare Data Analyst
- Clinical Data Analyst
- Healthcare Business Analyst
- Quality Improvement Analyst
- Program Analyst
- Healthcare Operations Analyst

## Limitations

- This is a baseline model, not a clinical deployment.
- The dataset is historical and may not reflect current hospital workflows.
- Readmission prediction involves complex social, clinical, and operational factors that are not fully captured in this dataset.
- Additional feature engineering and model comparison are needed before drawing stronger conclusions.
- No protected health information is used in this public repository.

## Next Improvements

- Add an exploratory data analysis notebook with visualizations.
- Add a model evaluation section with accuracy, recall, precision, F1 score, ROC-AUC, and confusion matrix.
- Compare logistic regression against tree-based models.
- Add a dashboard mockup for stakeholder-facing readmission insights.
- Add screenshots so recruiters can quickly see the workflow and outputs.

## Resume Bullet

Built a Python-based healthcare analytics portfolio project using a 100,000+ encounter hospital dataset to preprocess clinical data, train a baseline 30-day readmission risk model, and communicate potential applications for care coordination and follow-up planning.
