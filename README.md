# 📡 Telecom Customer Churn Analysis
### A Full-Stack Data Science Portfolio Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project demonstrates an end-to-end data science workflow applied to a high-impact business problem: **predicting which telecom customers are likely to cancel their subscription (churn)**, and translating those predictions into a financially-justified retention strategy.

The analysis goes far beyond model accuracy. It asks — and answers — the questions a real business would ask:

- Which customers are most likely to leave, and why?
- How much revenue is at risk?
- What does it cost to retain them, and is it worth it?
- Who should we target first, and with what kind of offer?

---

## 🎯 Business Problem

Customer churn is one of the most expensive problems in telecommunications. Acquiring a new customer typically costs **5–7× more** than retaining an existing one. Yet without a systematic early-warning system, retention teams are left reacting after the fact — when it's already too late.

This project builds a machine learning pipeline that:
1. Identifies at-risk customers *before* they cancel
2. Quantifies the revenue exposure of predicted churners
3. Recommends cost-effective, prioritised intervention strategies
4. Provides interpretable, actionable reasons behind each prediction

---

## 🗂️ Repository Structure

```
telecom-churn-analysis/
│
├── telecom_churn_portfolio_annotated.ipynb   # Main analysis notebook (fully annotated)
├── data/
│   └── synthetic_customer_churn_100k.csv     # Dataset (~100K customers)
├── outputs/                                   # Generated CSVs and artefacts
│   ├── cv_model_comparison.csv
│   ├── threshold_comparison.csv
│   ├── test_predictions.csv
│   ├── strategy_simulation.csv
│   ├── lift_table.csv
│   ├── risk_value_priority_matrix.csv
│   ├── feature_importance.csv
│   ├── uplift_segments.csv
│   └── executive_summary.csv
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/telecom-churn-analysis.git
cd telecom-churn-analysis

pip install -r requirements.txt
```

Then open the notebook:
```bash
jupyter notebook telecom_churn_portfolio_annotated.ipynb
```

**requirements.txt:**
```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.10
statsmodels>=0.14
xgboost>=2.0          # optional but recommended
shap>=0.42            # optional — for SHAP explainability
imbalanced-learn>=0.11 # optional — for SMOTE comparison
lifelines>=0.27       # optional — for survival analysis
```

---

## 🔬 Analysis Walkthrough

The notebook is structured as a logical progression from raw data to business action:

### Phase 1 — Data Foundation
| Section | Description |
|---|---|
| **1. Environment Setup** | Imports, configuration, business assumptions |
| **2. Data Loading** | Flexible ingestion, column standardisation, target encoding |
| **3. Data Quality Audit** | Missing values, duplicates, class imbalance detection |
| **4. Feature Engineering** | Tenure bands, charge ratios, service bundles, interaction terms |

### Phase 2 — Exploratory Analysis
| Section | Description |
|---|---|
| **5. EDA** | Churn rates and lift by segment; distribution plots |
| **6. Correlation & VIF** | Multicollinearity detection, Pearson heatmap |
| **7. Statistical Testing** | Chi-square + Cramér's V (categorical); Welch t-test + Mann-Whitney + Cohen's d (numeric) |
| **8. Survival Analysis** | Kaplan-Meier curves stratified by contract type; log-rank test |

### Phase 3 — Modelling
| Section | Description |
|---|---|
| **9. Train/Test Split** | Stratified 80/20 split |
| **10. Preprocessing Pipelines** | Sklearn pipelines for numeric and categorical features |
| **11. Class Imbalance** | `class_weight='balanced'` vs. SMOTE comparison |
| **12. Model Comparison** | 5-fold CV across Dummy, LogReg, Random Forest, GBM, XGBoost |
| **13. Hyperparameter Tuning** | GridSearchCV on best model |

### Phase 4 — Evaluation & Interpretability
| Section | Description |
|---|---|
| **14. Test Set Evaluation** | ROC-AUC, PR-AUC, Brier score, classification report |
| **15. Threshold Optimisation** | F1-optimal threshold; precision-recall tradeoff analysis |
| **16. ROC & PR Curves** | Full curve analysis across all thresholds |
| **17. Calibration** | Reliability diagram; Brier score; probability distribution |
| **18. Feature Importance & SHAP** | Global importances + SHAP summary plot |
| **19. Error Analysis** | False positive/negative profiling by segment |

### Phase 5 — Business Value
| Section | Description |
|---|---|
| **20. Revenue at Risk** | Probability-weighted monthly and annualised revenue exposure |
| **21. Cumulative Gains & Lift** | Campaign efficiency curves; decile lift table |
| **22. Uplift Segmentation** | Persuadable vs. decided churner classification |
| **23. Retention Strategy Simulation** | Multi-scenario ROI analysis across target size, cost, and success rate |
| **24. Risk-Value Priority Matrix** | 2×2 heatmap for campaign prioritisation |
| **25. Business Recommendations** | Immediate actions, deployment guidance, campaign design, research roadmap |
| **26. Limitations & Next Steps** | Honest caveats and improvement opportunities |
| **27. Executive Summary** | KPI table for stakeholder reporting |

---

## 📊 Key Results

> *Exact figures will vary depending on dataset and random state. Below are representative results typical for this type of analysis.*

| Metric | Value |
|---|---|
| Best Model | XGBoost / Gradient Boosting |
| Test ROC-AUC | ~0.87–0.92 |
| Test PR-AUC | ~0.72–0.82 |
| Brier Score | ~0.08–0.12 |
| Optimised Classification Threshold | ~0.35–0.45 |
| Top 10% Lift | ~3.0–4.5x |
| Monthly Revenue at Risk (test set) | Depends on dataset |
| Best Campaign ROI | Positive at 20–25% retention success rate |

---

## 💡 Key Business Insights

1. **Contract type is the strongest churn predictor.** Month-to-month customers churn at 3–4× the rate of customers on annual contracts — implying that contract conversion campaigns have structural, long-term value beyond any single retention effort.

2. **Service bundling reduces churn.** Each additional add-on service (security, backup, tech support) reduces churn probability. Customers with 4+ services have dramatically lower churn risk — cross-selling is a retention strategy, not just a revenue strategy.

3. **Early-life churn is distinct.** Kaplan-Meier curves show a steep drop in survival probability in the first 6 months. Onboarding experience and first-90-days engagement are as important as long-term loyalty programmes.

4. **Revenue concentration means targeted campaigns beat broad ones.** The top 10% of customers by risk score typically account for 40–60% of total revenue at risk. Broad campaigns dilute the signal and waste budget on customers who would have stayed anyway.

5. **Not all high-risk customers are worth retaining.** The risk-value matrix separates high-risk/high-value customers (immediate action) from high-risk/low-value customers (low-cost automated outreach or accept the churn). Budget should be allocated accordingly.

---

## 🏗️ Technical Highlights

- **End-to-end sklearn Pipelines** — preprocessing is bundled with the model, preventing data leakage and enabling one-step production serialisation
- **SHAP values** — per-customer explanations of model predictions, enabling personalised retention conversations rather than generic offers
- **Calibration analysis** — validates that predicted probabilities are trustworthy inputs for revenue-at-risk calculations
- **Threshold optimisation** — moves beyond default 0.50 to the business-relevant operating point
- **Monte Carlo-style strategy simulation** — stress-tests ROI across multiple cost and success rate assumptions to find the robust recommendation
- **Uplift proxy segmentation** — separates customers who need intervention from those who will churn regardless, reducing wasted spend

---

## ⚠️ Limitations

- **No temporal holdout:** The random train/test split doesn't simulate the actual deployment scenario (train on past, predict future). A time-based split would be more representative.
- **Synthetic/simplified dataset:** Real telecom data includes multi-product households, network usage patterns, call centre interactions, and NPS scores — all of which would improve model performance.
- **Uplift is proxied:** Without a randomised A/B test, the uplift segmentation estimates treatment receptivity from tenure rather than measuring it causally.
- **Static threshold:** A single threshold ignores customer-level heterogeneity. In practice, different customer segments may warrant different decision boundaries.

---

## 🚀 Next Steps & Extensions

- [ ] Implement time-based cross-validation (walk-forward validation)
- [ ] Add Customer Lifetime Value (CLV) to replace MonthlyCharges in revenue calculations
- [ ] Run a randomised controlled trial to measure true retention uplift
- [ ] Explore deep learning approaches (TabNet, TabTransformer)
- [ ] Build a real-time scoring API (FastAPI + model serialisation)
- [ ] Implement model monitoring dashboard (PSI, concept drift detection)
- [ ] Integrate call centre and NPS data as additional features

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
