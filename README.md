# Netflix & Menstruation: A Personal Data Science Project

## Project Overview

Since 2019, I have tracked my menstruation cycles using the Apple Health app and maintained a comprehensive record of my Netflix viewing history dating back to 2018. This project aims to analyze whether hormonal changes during menstruation influence my media consumption behavior—specifically genre preferences and viewing duration. Additionally, it explores the potential for such behavioral patterns to inform personalized content recommendations or advertising strategies in the future, based on hormonal cycles.

By merging these two personal datasets, I explore correlations between menstrual phases and entertainment choices, using data science techniques such as exploratory data analysis (EDA), statistical testing, and machine learning classification.

## Objectives

* Identify behavioral patterns: Examine how genre preference and watch duration vary across menstruation and non-menstruation periods.
* Analyze statistical correlations: Use hypothesis testing and correlation analysis to evaluate the impact of menstruation.
* Apply data science techniques: Utilize tools such as pandas, seaborn, t-tests, chi-square tests, and ML models (Logistic Regression, Random Forest, KNN).

## Datasets Used

* NetflixViewingHistory.csv: Raw Netflix viewing history with timestamps.
* Genre\_Assignment\_Final.csv: Manually matched genres for each title using IMDb.
* menstruation\_dates.csv: Menstruation start and end dates from Apple Health.
* Feature\_Table.csv: Preprocessed feature table combining date, genre, season, view counts, weekday, and menstruation status.

## Methodology

### 1. Data Preparation

* Merged Netflix viewing records with genre assignments and date features.
* Created new variables: is\_menstruation, season, weekday, and view\_count.
* Engineered a feature table that marks if a day falls within a menstruation window.

### 2. Exploratory Data Analysis (EDA)

* Univariate plots: Frequency of genres, viewing patterns by weekday/season.
* Bivariate/Multivariate plots: Genre preferences during menstruation vs. non-menstruation, view count variations, heatmaps, and rolling averages.

### 3. Hypothesis Testing

* H0: Menstruation status has no impact on viewing behavior (view count or genre).
* H1: Menstruation status significantly affects viewing behavior.

Results:

* T-Test (View Count vs Menstruation): p-value = 0.4924 → Fail to reject H0. No significant difference in view count.
* Chi-Square Test (Genre vs Menstruation): p-value = 0.0346 → Reject H0. Genre is associated with menstruation status.

### 4. Machine Learning Modeling

To predict whether a day is part of a menstruation period based on viewing behavior, three classification models were tested:

| Model               | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| ------------------- | -------- | ------------------- | ---------------- | ------------------ |
| Logistic Regression | 0.81     | 0.95                | 0.66             | 0.78               |
| Random Forest       | 0.86     | 0.94                | 0.76             | 0.84               |
| K-Nearest Neighbors | 0.75     | 0.69                | 0.92             | 0.79               |

* All models used SMOTE for class balancing (443 samples for each class).
* Random Forest provided the highest accuracy (86%) and well-balanced precision-recall.
* Logistic Regression also performed well and provides interpretability.
* KNN had strong recall but lower precision, making it less reliable for this use case.

## Key Findings

* Seasonal and Weekly Trends: Analysis showed that certain seasons (e.g., winter) and specific days of the week (e.g., weekends) had distinct viewing patterns. For instance, view counts were generally higher on Sundays, and drama genres were more frequently watched during colder months.

* Genre Shift: Preference for emotional or comforting genres (e.g., drama, comedy) during menstruation.

* View Count: No statistically significant difference in overall view count.

* Statistical Confirmation: Genre preferences are measurably associated with menstruation status.

* Best Performing Model: Random Forest Classifier (accuracy: 86%).

## Tools and Libraries

* Python, pandas, matplotlib, seaborn for data cleaning and visualization
* scikit-learn, imblearn for modeling and oversampling
* scipy.stats for statistical testing

## Limitations & Future Work

* Single-subject dataset: Limits generalizability.
* Manual genre mapping: Could introduce bias.
* Future steps:

  * Collect more data over longer periods
  * Include physiological or mood ratings for better label accuracy
  * Automate genre classification using NLP or APIs

## Conclusion

This project demonstrates that personal behavioral data—when properly structured and analyzed—can reveal meaningful insights. The menstrual cycle shows a statistically significant association with Netflix genre preferences but not with total viewing time.

Machine learning models showed moderate success in predicting menstruation periods from viewing behavior, with Random Forest offering the best balance of accuracy and robustness.

The results suggest that mood-linked behavioral shifts can be captured via digital footprints, opening the door to more personalized recommendation systems and self-awareness through data.


