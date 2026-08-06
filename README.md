# Online Retail Customer Analytics

## Data Source

**[UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)**

Actual invoices from a UK-based online retailer (2010–2011), 541,909 transaction records. Not synthetic data.

Columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

## Problem Statement

The raw dataset had no churn label, no customer segments, and no CLTV target — all three had to be defined from scratch before any model could be trained.

- **Churn**: Cohort Analysis revealed that the `retention rate` by month since first purchase showed less than 20% of customers return in month 1. This shaped the observation window, 60 days (30 to interact, 30 to establish a repeat pattern), with the following 100 days used as the prediction target, allowing the business to intervene before clients actually churn.
- **Segments**: built from scratch with KMeans on the same 60-day window.
- **CLTV target**: 100-day forward spend, constructed from raw transaction history.

## Executive Summary

Goal: This project was aimed at identifying which at-risk customers are worth keeping or winning back, and how much to spend to retain or re-acquire them.

Using each customer's first 60 days of activity, three models work together:

- **Segmentation (KMeans + PCA)** - 4 personas: `High-Value Elite`, `Loyal Performers`, `One-Time Bulk-Buyers`, `One-Time Acquaintances`. The KMeans personas were validated against real behavior 100 days later, and the `High-Value Elite` registered the highest retention rate at 95%, while the `One-Time Acquaintance` segment presented highest churn rate, at 79%.
- **Churn model (Logistic Regression)** - predicts who goes quiet for 100+ days. The model delivered 82% recall, with a 74% accuracy out-of-sample.
- **CLTV model (Random Forest)** - predicts next 100-day spend per customer. An average error  of ~$247 was registered.

Churn risk + predicted CLTV combine into a 2×2 risk matrix. Out of the 430 customers, 100 customers were identified as the priority re-acquisition target, which are the `Low Churn - High CLTV` profile, made up of One-Time Acquaintances and One-Time Bulk-Buyers who are predicted to spend above $400 in the next 100 days but are still at risk of not returning.

The recommendation: re-engage these customers at a 1:10 risk-to-reward ratio, with a proposed budget of $4,565 for One-Time Acquaintances and $740 for One-Time Bulk-Buyers. Any at-risk customer outside this risk profile should be treated the same as a new acquisition.

## Notebooks

> **Quick start**: Jump straight to `7.0 Final Delivery: Inference Pipeline` to see the end output, risk profiles, churn predictions, and budget recommendations. That said, for deeper insight into how everything was built, work through the notebooks in order.

| # | Notebook | What it does |
|---|----------|---------------|
| 1.0 | Data Cleaning Pipeline | Cleans raw transactions from S3, outputs clean parquet |
| 2.0 | Feature Engineering | Shared feature-extraction function used by segmentation, churn, CLTV |
| 4.0 | Customer Segmentation | KMeans on PCA-reduced first-60-day behavior, 4 personas assigned |
| 5.0 | Predicting Customer Churn | Logistic Regression, 100-day churn prediction, SHAP analysis |
| 6.0 | Predicting CLTV | Random Forest, 100-day forward spend prediction |
| 7.0 | Final Delivery: Inference Pipeline | Applies all 3 models to new data, outputs risk matrix + budget recommendation |

## How It All Fits Together

The raw Excel file is stored in AWS S3 and streamed into the Data Cleaning Pipeline, where bad records are flagged and removed and a revenue column is added. The clean dataset is written back to S3 as a parquet file.

The Feature Engineering notebook then computes all customer-level features from the clean data and stores the extraction function in S3, where every notebook calls it directly, so tgat the segmentation, churn, and CLTV all work from the same feature definitions.

The three modeling notebooks run independently from there: Segmentation assigns each customer a persona, Churn predicts who goes quiet, and CLTV predicts forward spend. Each trained model is saved as a joblib file back to S3.

The Final Delivery notebook pulls all three models from S3, runs them on new unseen customer data, and combines the outputs into a risk matrix and a dollar-figure re-engagement budget. Results are written to PostgreSQL (Neon) for querying and business intelligence access.

## Storage

- **AWS S3** — raw and cleaned data (parquet), plus trained model files (joblib).
- **PostgreSQL (Neon)** — final outputs: customer segment, churn probability, predicted CLTV, risk profile.

## Tech Stack

Python · pandas / NumPy · scikit-learn · SHAP · AWS S3 · Parquet · PostgreSQL (Neon) · Google Colab

## Structure

```
online-retail-customer-analytics/
├── 1.0 Data Cleaning Pipeline.ipynb
├── 2.0 Feature_Engineering.ipynb
├── 4.0 Customer Segmentation.ipynb
├── 5.0 Predicting Customer Churn.ipynb
├── 6.0 Predicting_Customer_Life_Time_Value_(CLTV).ipynb
└── 7.0 Final_Delivery: Churn and CLTV Inference Pipeline.ipynb
```
