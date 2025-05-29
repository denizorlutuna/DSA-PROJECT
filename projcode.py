# ----------------------------------------------
# Netflix & Menstruation Project - Final Version
# ----------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# ------------------------------
# Step 1: Load and clean datasets
# ------------------------------

# Load the feature dataset which includes view counts, genre, dates and menstruation status
feature_df = pd.read_csv("Feature_Table.csv", sep=";")
feature_df.columns = [c.strip().lower().replace(" ", "_") for c in feature_df.columns]
feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
feature_df["is_menstruation"] = feature_df["is_menstruation"].astype(bool)
feature_df["year"] = feature_df["date"].dt.year
feature_df["month"] = feature_df["date"].dt.month
feature_df["day"] = feature_df["date"].dt.day

# Load and clean the raw Netflix viewing history
netflix_df = pd.read_csv("NetflixViewingHistory.csv", sep=";")
netflix_df.columns = [c.strip().lower().replace(" ", "_") for c in netflix_df.columns]
netflix_df["date"] = pd.to_datetime(netflix_df["date"], format="%m/%d/%Y", errors="coerce")

# Load the menstruation start-end date ranges
menstruation_df = pd.read_csv("menstruation_dates.csv", sep=";")
menstruation_df.columns = [c.strip().lower().replace(" ", "_") for c in menstruation_df.columns]
menstruation_df["starting_date"] = pd.to_datetime(menstruation_df["starting_date"], dayfirst=True)
menstruation_df["ending_date"] = pd.to_datetime(menstruation_df["ending_date"], dayfirst=True)

# ------------------------------
# Step 2: Prepare EDA Plot Saving
# ------------------------------

# Create a folder to save generated plots
os.makedirs("eda_plots", exist_ok=True)

def save_fig(fig, name):
    fig.savefig(f"eda_plots/{name}.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# ------------------------------

# These plots help us understand data distribution and potential patterns.

# Plot 1: Frequency of each dominant genre watched
fig = plt.figure(figsize=(12, 6))
sns.countplot(data=feature_df, x="dominant_genre", order=feature_df["dominant_genre"].value_counts().index)
plt.title("Dominant Genre Frequency")
plt.xticks(rotation=45)
save_fig(fig, "01_dominant_genre_count")

# Plot 2: Distribution of daily view counts
fig = plt.figure(figsize=(10, 6))
sns.histplot(feature_df["view_count"], bins=20, kde=True)
plt.title("View Count Distribution")
save_fig(fig, "02_view_count_distribution")

# Plot 3: View counts grouped by genre
fig = plt.figure(figsize=(12, 6))
sns.boxplot(data=feature_df, x="dominant_genre", y="view_count")
plt.xticks(rotation=45)
plt.title("View Count by Genre")
save_fig(fig, "03_view_count_by_genre")

# Plot 4: View counts by weekday
fig = plt.figure(figsize=(10, 5))
sns.boxplot(data=feature_df, x="weekday", y="view_count")
plt.title("View Count by Weekday")
save_fig(fig, "04_view_count_by_weekday")

# Plot 5: View counts by season
fig = plt.figure(figsize=(8, 5))
sns.boxplot(data=feature_df, x="season", y="view_count")
plt.title("View Count by Season")
save_fig(fig, "05_view_count_by_season")

# Plot 6: Genre frequency split by menstruation status
fig = plt.figure(figsize=(10, 5))
sns.countplot(data=feature_df, x="dominant_genre", hue="is_menstruation")
plt.title("Genre by Menstruation Status")
plt.xticks(rotation=45)
save_fig(fig, "06_genre_menstruation")

# Plot 7: Average view count by menstruation status
fig = plt.figure(figsize=(6, 4))
sns.barplot(data=feature_df, x="is_menstruation", y="view_count")
plt.title("Avg View Count by Menstruation")
save_fig(fig, "07_avg_view_menstruation")

# Plot 8: Genre viewing trend over time (smoothed)
genre_time = feature_df.groupby(["date", "dominant_genre"]).size().unstack(fill_value=0)
fig = plt.figure(figsize=(14, 6))
genre_time.rolling(30).mean().plot(ax=plt.gca())
plt.title("Genre Trend Over Time (30-day rolling)")
plt.tight_layout()
save_fig(fig, "08_genre_trend")

# Plot 9: Average view count by month
monthly_avg = feature_df.groupby("month")["view_count"].mean()
fig = plt.figure()
monthly_avg.plot(kind="bar")
plt.title("Monthly Avg View Count")
save_fig(fig, "09_monthly_avg_views")

# Plot 10: Frequency of watching per weekday
fig = plt.figure()
sns.countplot(data=feature_df, x="weekday", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Viewing by Weekday")
save_fig(fig, "10_weekday_distribution")

# Plot 11: Season viewing distribution
fig = plt.figure()
sns.countplot(data=feature_df, x="season")
plt.title("Viewing by Season")
save_fig(fig, "11_season_distribution")

# Plot 12: View count by genre and menstruation status
fig = plt.figure(figsize=(12, 6))
sns.boxplot(data=feature_df, x="dominant_genre", y="view_count", hue="is_menstruation")
plt.title("View Count by Genre & Menstruation")
plt.xticks(rotation=45)
save_fig(fig, "12_genre_view_menstruation")

# Plot 13: Heatmap of viewing genre by weekday
genre_weekday = pd.crosstab(feature_df["weekday"], feature_df["dominant_genre"])
fig = plt.figure(figsize=(12, 6))
sns.heatmap(genre_weekday, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Genre vs Weekday Heatmap")
save_fig(fig, "13_genre_weekday_heatmap")

# Plot 14: Correlation matrix of numeric features
fig = plt.figure(figsize=(6, 4))
sns.heatmap(feature_df[["view_count", "year", "month", "day"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
save_fig(fig, "14_correlation_matrix")

# Plot 15: Pairwise plots for numeric variables
fig = sns.pairplot(feature_df[["view_count", "year", "month", "day"]])
fig.fig.suptitle("Pairplot of Viewing Features", y=1.02)
fig.savefig("eda_plots/15_pairplot.png", bbox_inches="tight")
plt.close()

# Expand menstruation days
menstruation_days = []
for _, row in menstruation_df.iterrows():
    menstruation_days.extend(pd.date_range(start=row["starting_date"], end=row["ending_date"]))
menstruation_days = pd.Series(menstruation_days)

# Plot 16: Monthly comparison
regl_freq = menstruation_days.dt.to_period("M").value_counts().sort_index()
netflix_freq = netflix_df["date"].dt.to_period("M").value_counts().sort_index()
merged_freq = pd.DataFrame({
    "Menstruation Days": regl_freq,
    "Viewing Days": netflix_freq
}).fillna(0)
merged_freq.index = merged_freq.index.to_timestamp()

fig = plt.figure(figsize=(12, 5))
merged_freq.plot(ax=plt.gca(), marker="o")
plt.title("Monthly Comparison of Menstruation Days and Netflix Viewing Days")
plt.ylabel("Days")
plt.xticks(rotation=45)
plt.tight_layout()
save_fig(fig, "16_monthly_regl_vs_netflix")


# Plot 17: 30-day rolling average
daily_men = pd.Series(1, index=menstruation_days).resample("D").sum()
daily_view = netflix_df["date"].value_counts().sort_index().resample("D").sum()
rolling_df = pd.DataFrame({
    "Menstruation": daily_men,
    "Netflix": daily_view
}).fillna(0).rolling(30, min_periods=1).mean()

fig = plt.figure(figsize=(12, 5))
rolling_df.plot(ax=plt.gca())
plt.title("30-Day Rolling Average of Menstruation and Netflix Viewing")
plt.xlabel("Date")
plt.ylabel("Average Daily Count")
plt.tight_layout()
save_fig(fig, "17_rolling_avg_regl_netflix")

# ------------------------------
# Step 4: Hypothesis Testing
# ------------------------------

# Hypothesis 1: Does menstruation affect average daily view count?
# H0: Menstruation status has no effect on view count.
# H1: There is a significant difference in view counts depending on menstruation status.

menstruation_views = feature_df[feature_df["is_menstruation"] == True]["view_count"]
non_menstruation_views = feature_df[feature_df["is_menstruation"] == False]["view_count"]

# Welch's t-test (does not assume equal variance)
t_stat, p_val = ttest_ind(menstruation_views, non_menstruation_views, equal_var=False)
print("\n--- Hypothesis Test 1: View Count vs Menstruation ---")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Reject H0: Menstruation status significantly affects view count.")
else:
    print("Fail to reject H0: No significant difference in view count.")

# Hypothesis 2: Is there an association between dominant genre and menstruation?
# H0: Genre and menstruation are independent.
# H1: Genre and menstruation are related.

contingency_table = pd.crosstab(feature_df["dominant_genre"], feature_df["is_menstruation"])
chi2, p_chi, _, _ = chi2_contingency(contingency_table)
print("\n--- Hypothesis Test 2: Genre vs Menstruation ---")
print(f"Chi-squared: {chi2:.4f}, P-value: {p_chi:.4f}")
if p_chi < 0.05:
    print("Reject H0: Genre is associated with menstruation status.")
else:
    print("Fail to reject H0: No significant association found.")

# ------------------------------
# Step 5: Machine Learning Phase
# ------------------------------

# Prepare features and target
# We'll drop non-predictive features and convert the target to 0/1
X = feature_df.drop(columns=["date", "is_menstruation"])
y = feature_df["is_menstruation"].astype(int)

# One-hot encode categorical columns manually
X_encoded = pd.get_dummies(X, columns=["dominant_genre", "weekday", "season"])

# Use SMOTE to balance the classes
# SMOTE (Synthetic Minority Oversampling Technique) generates synthetic examples for the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred_logreg))

fig = plt.figure()
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, cmap="Blues", ax=plt.gca())
plt.title("Logistic Regression Confusion Matrix")
save_fig(fig, "logistic_confusion_matrix")

# --- Random Forest Classifier ---
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf))

fig = plt.figure()
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap="Greens", ax=plt.gca())
plt.title("Random Forest Confusion Matrix")
save_fig(fig, "random_forest_confusion_matrix")

# --- K-Nearest Neighbors Classifier ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n--- K-Nearest Neighbors Results ---")
print(classification_report(y_test, y_pred_knn))

fig = plt.figure()
ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, cmap="Purples", ax=plt.gca())
plt.title("KNN Confusion Matrix")
save_fig(fig, "knn_confusion_matrix")

