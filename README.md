# UK Data Science Job Market Analysis & Graduate Role Classifier
> End-to-end NLP and machine learning project on 355 real UK job listings scraped from Indeed — classifying graduate vs non-graduate roles with **90% accuracy**.

## Overview
This project analyses the UK data science job market using real job posting data from Indeed (January 2024). It goes beyond simple exploratory analysis — building a full machine learning pipeline to classify whether a job posting is suitable for a graduate or early-career candidate, using NLP-based feature engineering, seniority detection, and skill extraction.

## Results
| Model | Accuracy | Precision (non-grad) | Recall (non-grad) | F1 (grad class) |
|-------|----------|----------------------|-------------------|-----------------|
| Logistic Regression | **90.1%** | 0.95 | 0.94 | 0.53 |
| Random Forest (400 trees) | **90.1%** | 0.94 | 0.95 | 0.46 |

**Confusion Matrix — Random Forest:**
```
Predicted:    Non-Grad    Grad
Actual Non-Grad    61        3
Actual Grad         4        3
```

## Project Workflow

### 1. Data Loading & Exploration
- Loaded 355 UK data science job listings from `indeed_uk_datascience_jobs_jan2024.csv`
- Dataset: 6 columns — positionName, salary, company, rating, reviewsCount, jobTypeConsolidated
- 272 unique job titles; 255 unique companies; 59% of roles had no salary listed

### 2. Text Cleaning (NLP Preprocessing)
- Lowercased, stripped whitespace, and removed special characters from job titles using regex
- Created clean `position_clean` column as base for all feature engineering

### 3. Seniority Indicator Features
Engineered binary flags from job title text using word-boundary regex:
- `junior` — detected "junior" in title
- `senior` — detected "senior" in title
- `graduate_keyword` — detected "graduate" in title
- `intern_keyword` — detected "intern" or "internship"
- `mid` — detected "mid", "midlevel", or "mid-level"

### 4. Skill Indicator Features
Extracted skill signals from job title text:
- `python`, `sql`, `r_lang`, `machine_learning`
- `data_analyst`, `data_scientist`, `business_analyst`

### 5. Job Type Encoding
- Filled missing job type with "Unknown"
- One-hot encoded `jobTypeConsolidated` into 22 dummy variables

### 6. Salary Flag
- Created binary `has_salary_info` feature (1 = salary listed, 0 = not listed)
- 146 of 355 roles (41%) had salary information

### 7. Target Label Construction
- Defined graduate-suitable roles using keyword detection: `['graduate', 'junior', 'entry', 'trainee', 'intern', 'associate']`
- **35 graduate roles** identified out of 355 total (9.9% of dataset)
- Highly imbalanced dataset — handled using `class_weight='balanced'`

### 8. Model Training
- **Feature set:** 35 features (skill indicators + seniority flags + job type dummies + salary flag)
- **Train/test split:** 80/20 with stratification to preserve class balance
- **Logistic Regression:** `class_weight='balanced'`, `max_iter=1000`
- **Random Forest:** 400 estimators, `class_weight='balanced_subsample'`

### 9. Feature Importance Analysis
Top predictors identified by Random Forest:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | junior (title keyword) | 0.180 |
| 2 | type_Fixed term contract | 0.127 |
| 3 | data_scientist (title keyword) | 0.078 |
| 4 | intern_keyword | 0.073 |
| 5 | has_salary_info | 0.063 |
| 6 | senior (title keyword) | 0.063 |
| 7 | type_Permanent | 0.053 |
| 8 | machine_learning (title keyword) | 0.049 |

### 10. Output Files
- `processed_uk_datascience_jobs_with_features.csv` — full engineered dataset
- `feature_importance_results.csv` — ranked feature importances
- `confusion_matrix_rf.png` — Random Forest confusion matrix visualisation
- `feature_importance_top15.png` — top 15 features bar chart

## Key Findings
- Only **9.9% of UK data science job listings** are explicitly targeted at graduates or entry-level candidates — the market is dominated by mid-to-senior roles
- The word **"junior"** in a job title is the strongest single predictor that a role is graduate-appropriate (importance: 0.18)
- Roles listed as **Fixed Term Contract** are disproportionately entry-level compared to Permanent roles
- **41% of job listings have no salary information** — a significant data quality challenge in real-world job market datasets
- The most frequently appearing company: **Harnham** (specialist data & analytics recruiter)

## Dataset
Real UK data science job listings scraped from Indeed in January 2024.
355 records · 6 raw features · 35 engineered features

## Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, NumPy, scikit-learn, Matplotlib, re (regex)
- **Models:** Logistic Regression, Random Forest Classifier
- **Environment:** Jupyter Notebook (Anaconda)

## Key Skills Demonstrated
- Real-world messy data cleaning and preprocessing
- NLP feature engineering using regex on unstructured text
- Handling class imbalance (`class_weight='balanced'`)
- Comparative model evaluation (accuracy, precision, recall, F1, confusion matrix)
- Feature importance analysis and interpretation
- End-to-end ML pipeline from raw CSV to saved model outputs
- Data-driven insight generation from job market data

## How to Run
```bash
git clone https://github.com/yashparmar1/uk-job-market-analysis.git
cd uk-job-market-analysis
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook multidisciplinary_employability_uk_jobs.ipynb
```

> **Note:** The dataset `indeed_uk_datascience_jobs_jan2024.csv` is required. Place it in the same directory as the notebook before running.

## Author
**Yashkumar Parmar**
MSc Data Science — Middlesex University London (2026)
[LinkedIn](https://www.linkedin.com/in/yashkumar-parmar-7752b8238) · [GitHub](https://github.com/yashparmar1)

