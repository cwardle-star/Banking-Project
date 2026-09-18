# Banking Financial Performance & Analytics

**Comparative financial analysis, risk classification, and statistical modeling across four major U.S. banks.**

This portfolio examines how profitability, operating efficiency, and balance sheet structure vary across Bank of America, JPMorgan Chase, Citigroup, and Wells Fargo. Using financial data compiled from SEC 10-K filings, I consolidated 20 years of bank financial information and developed four analyses that progress from historical performance comparisons to regression and clustering.

| Coverage | Scope |
| :--- | :--- |
| **Period** | 2005-2024 |
| **Institutions** | Bank of America, JPMorgan Chase, Citigroup, Wells Fargo |
| **Dataset** | 80 annual bank observations |
| **Financial measures** | Assets, equity, revenue, net income, profitability ratios, and efficiency measures |
| **Primary tools** | Excel and Python |

## Project Overview

### 01 · Comparative Financial Performance

**How does Bank of America’s financial performance compare with its peers?**

Evaluates historical trends in revenue, net income, return on assets, return on equity, profit margin, asset turnover, and equity-to-assets. The analysis combines measures of financial scale with ratios to compare profitability, efficiency, and financing structure across institutions and economic periods.

*Methods: peer benchmarking, financial ratio analysis, and time-series visualization.*

### 02 · Financial Risk Classification

**Which bank-year observations meet defined indicators of financial weakness?**

Develops rule-based classifications using profitability, annual changes in financial measures, and equity-to-assets thresholds. A random forest classifier is evaluated against these project-defined labels, with attention to class imbalance, precision, recall, and the distinction between reproducing financial rules and predicting future distress.

*Methods: feature preparation, rule-based labeling, random forest classification, and model evaluation.*

### 03 · Bank Profitability Modeling

**How are financial scale and performance measures associated with net income?**

Compares baseline linear regression, ratio-based feature engineering, and Ridge regression to estimate net income. Model comparisons use RMSE and R² while examining correlated predictors and target leakage, demonstrating why strong numerical results require careful interpretation.

*Methods: ordinary least squares, logarithmic transformation, Ridge regularization, and residual analysis.*

### 04 · Financial Performance Clustering

**Do bank-year observations form groups with similar financial characteristics?**

Applies k-means and hierarchical clustering to explore patterns across financial measures. Elbow analysis, a hierarchical dendrogram, and principal component analysis provide complementary views of cluster structure and similarities among observations.

*Methods: feature standardization, k-means, Ward hierarchical clustering, and PCA visualization.*

## Technical & Financial Skills

| Area | Application |
| :--- | :--- |
| **Financial analysis** | Financial statement data consolidation, peer comparisons, profitability and efficiency ratios, and historical trend analysis |
| **Data preparation** | Excel-based organization, combining datasets, numeric conversion, feature engineering, and standardization |
| **Statistical modeling** | Classification, linear and Ridge regression, clustering, and dimensionality reduction |
| **Analytical communication** | Interpreting results, evaluating model limitations, and presenting findings through charts and written analysis |

**Technology:** Excel · Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · SciPy

---

Each project folder contains the analysis, code implementations, and supporting charts. This portfolio was developed through academic coursework to connect financial analysis with practical applications of data analytics.
