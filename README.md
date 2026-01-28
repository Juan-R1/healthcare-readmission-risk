## Predicting 30-Day Hospital Readmission Risk

This project demonstrates an end-to-end workflow for exploring and modelling hospital readmission risk using the publicly available **Diabetes 130-US hospitals** dataset (1999‑2008). The goal is to build a baseline predictive model to identify patients at risk of being readmitted within 30 days, which can inform targeted interventions and reduce costs.

### Problem statement

Hospital readmissions within 30 days of discharge are expensive and often preventable. By predicting which patients are at risk, care teams can allocate additional resources (e.g. follow-up calls, education) to those who need them most. This project uses data collected from 130 US hospitals to train a simple machine-learning model for predicting readmission risk.

### Dataset

The **Diabetes 130-US hospitals for years 1999‑2008** dataset is provided by the UCI Machine Learning Repository and contains over 100 000 hospital encounters for patients with diabetes. Each record includes demographics (age, gender, race), admission type, discharge disposition, diagnoses codes, medications, and whether the patient was readmitted within 30 days, after 30 days, or not readmitted.

> **Note:** The dataset used in this project is released under a permissive licence for educational and research purposes. See the UCI repository for full licensing details.

### Repository contents

* `src/data.py` – downloads and pre-processes the dataset into a clean dataframe.
* `src/model.py` – trains a baseline model (logistic regression) to predict 30-day readmission, evaluates accuracy and AUC, and outputs metrics.
* `notebooks/01_eda.ipynb` – exploratory data analysis of demographic and clinical variables (placeholder; you can expand with your own analysis).
* `notebooks/02_modeling.ipynb` – modelling workflow (placeholder; replicate the logic from `src/model.py` in a notebook if desired).
* `requirements.txt` – lists Python dependencies (pandas, scikit-learn, matplotlib, etc.).
* `LICENSE` – MIT licence for this repository.

### Getting started

```
bash
git clone https://github.com/<your-username>/healthcare-readmission-risk.git
cd healthcare-readmission-risk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data.py  # downloads and prepares the dataset
python src/model.py # trains the model and prints metrics
```

For exploratory analysis and custom visualisations, open the Jupyter notebooks in the `notebooks` directory with `jupyter notebook`.

### Next steps

This repository provides a baseline pipeline; there are many ways to improve it:

* Feature engineering (e.g. one-hot encoding of diagnoses, aggregating medications into categories).
* Trying different algorithms (random forests, gradient boosting, neural networks).
* Incorporating cost-sensitive metrics to account for the imbalance between readmitted and non-readmitted patients.
* Building a dashboard to visualise key insights for stakeholders.

Feel free to fork this project and build upon it. Contributions and pull requests are welcome.
