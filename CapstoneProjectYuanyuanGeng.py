# Capstone Project
# Yuan Geng
# yg3130

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

# N-number: N17684158
np.random.seed(17684158)

# Load datasets
data = np.genfromtxt('rmpCapstoneNum.csv', delimiter=',')

# Filter out professors with less than 5 ratings
data = data[data[:, 2] >= 5]
print(data.shape[0])

# Q1
average_rating = data[:, 0]
male = data[:, 6]
female = data[:, 7]

missing_male = np.isnan(data[:, 6]).sum()
missing_female = np.isnan(data[:, 7]).sum()
print(missing_male, missing_female)

male_ratings = average_rating[male == 1]
female_ratings = average_rating[female == 1]

male_median = np.median(male_ratings)
female_median = np.median(female_ratings)
print(male_median, female_median)

u, p_value = stats.mannwhitneyu(male_ratings, female_ratings)
print(u)
print(p_value)

plt.figure(figsize=(8, 6))
plt.boxplot([male_ratings, female_ratings], labels=["Male", "Female"])
plt.title("Comparison of Average Ratings by Gender")
plt.ylabel("Average Rating")
plt.grid(True)
plt.show()

# Q2
average_rating = data[:, 0]
num_of_ratings = data[:, 2]

rho, p_value = spearmanr(num_of_ratings, average_rating)
print(rho)
print(p_value)

plt.figure(figsize=(8, 6))
plt.scatter(num_of_ratings, average_rating, alpha=0.5, s=10)
plt.title("Relationship Between Number of Ratings and Average Rating")
plt.xlabel("Number of Ratings")
plt.ylabel("Average Rating")
plt.grid(True)
plt.show()

# Q3
average_rating = data[:, 0]
average_difficulty = data[:, 1]

r = np.corrcoef(average_rating, average_difficulty)
print(r)

plt.figure(figsize=(8, 6))
plt.scatter(average_rating, average_difficulty, alpha=0.5, s=10)
plt.title("Relationship Between Average Difficulty and Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Average Difficulty")
plt.grid(True)
plt.show()

# Q4
average_rating = data[:, 0]
online_count = data[:, 5]

online_count_missing = np.isnan(data[:, 5]).sum()
print(online_count_missing)

bottom = np.percentile(online_count, 20)
top = np.percentile(online_count, 80)

low_group = average_rating[online_count <= bottom]
high_group = average_rating[online_count >= top]

u, p_value = stats.mannwhitneyu(low_group, high_group)
print(u)
print(p_value)

low_median = np.median(low_group)
high_median = np.median(high_group)
print(low_median, high_median)

plt.figure(figsize=(8, 6))
plt.boxplot([low_group, high_group], labels=["Low Online Class Count", "High Online Class Count"])
plt.title("Average Ratings by Number of Online Teaching")
plt.ylabel("Average Rating")
plt.grid(True)
plt.show()

# Q5
average_rating = data[:, 0]
retake = data[:, 4]
average_rating = average_rating[~np.isnan(retake)]
retake = retake[~np.isnan(retake)]

r = np.corrcoef(average_rating, retake)
print(r)

plt.figure(figsize=(8, 6))
plt.scatter(average_rating, retake, alpha=0.5, s=10)
plt.title("Relationship Between Average Rating and Retake Proportion")
plt.xlabel("Average Rating")
plt.ylabel("Proportion of People Who Would Retake Class")
plt.grid(True)
plt.show()

# Q6
average_rating = data[:, 0]
pepper = data[:, 3]

pepper_missing = np.isnan(data[:, 3]).sum()
print(pepper_missing)

hot_ratings = average_rating[pepper == 1]
not_hot_ratings = average_rating[pepper == 0]

hot_median = np.median(hot_ratings)
not_hot_median = np.median(not_hot_ratings)
print(hot_median, not_hot_median)

u, p_value = stats.mannwhitneyu(hot_ratings, not_hot_ratings)
print(u)
print(p_value)

plt.figure(figsize=(8, 6))
plt.boxplot([hot_ratings, not_hot_ratings], labels=["Hot", "Not Hot"])
plt.title("Comparison of Average Ratings: Hot vs. Not Hot Professors")
plt.ylabel("Average Rating")
plt.grid(True)
plt.show()

# Q7
X = data[:, 1].reshape(-1, 1)
y = data[:, 0]

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

rSq = model.score(X, y)
residuals = y - y_pred
rmse = np.sqrt(np.mean(residuals ** 2))
print(rSq)
print(rmse)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.5, s=10)
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.title("Average Rating vs. Average Difficulty")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.legend()
plt.grid(True)
plt.show()

# Q8
predictors = data[:, 1:6]

predictors = predictors[~np.isnan(predictors).any(axis=1)]

columns = [
    "Difficulty",
    "NumRatings",
    "Pepper",
    "Retake",
    "OnlineCount",
]
df = pd.DataFrame(predictors, columns=columns)

corr_matrix = df.corr()
corr_matrix

X = data[:, [1, 2, 3, 5]]
y = data[:, 0]

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

rSq = model.score(X, y)
residuals = y - y_pred
rmse = np.sqrt(np.mean(residuals ** 2))
print(rSq)
print(rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.3, s=10)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')
plt.title("Predicted vs. Actual Ratings")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.grid(True)
plt.show()

# Q9
X = data[:, [0]]
y = data[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

accuracy, precision, recall, roc_auc

fp, tp, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fp, tp, label="Logistic Regression (AUROC = 0.769)")
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Pepper by Average Rating")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Q10
X = data[:, [0, 1, 2, 5]]
y = data[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

accuracy, precision, recall, roc_auc

fp, tp, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fp, tp, label="Logistic Regression (AUROC = 0.779)")
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Pepper with Multiple Predictors")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Extra Credit
qual_data = pd.read_csv('rmpCapstoneQual.csv') # Used pandas because some of the univerisity names contain a comma

indices = np.where(data[:, 2] >= 5)[0]
qual_data = qual_data.iloc[indices].reset_index(drop=True)
majors = qual_data.iloc[:, 0]

missing_majors = majors.isna().sum()
missing_majors

data = data[~qual_data.iloc[:, 0].isna().to_numpy()]
qual_data = qual_data[~qual_data.iloc[:, 0].isna()].reset_index(drop=True)

ratings = data[:, 0]
majors = qual_data.iloc[:, 0]

stem_keywords = ["computer", "engineering", "math", "physics", "chemistry", "biology", "geology", "astronomy", "statistics"]
stem_flag = np.array([
    1 if any(word in major.lower() for word in stem_keywords) else 0
    for major in majors
])

X = stem_flag.reshape(-1, 1)
y = ratings

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

coef = model.coef_[0]
rSq = model.score(X, y)
residuals = y - y_pred
rmse = np.sqrt(np.mean(residuals ** 2))
print(coef)
print(rSq)
print(rmse)

stem_ratings = ratings[stem_flag == 1]
nonstem_ratings = ratings[stem_flag == 0]

u, p_value = stats.mannwhitneyu(stem_ratings, nonstem_ratings)
print(u)
print(p_value)

plt.figure(figsize=(8, 6))
plt.boxplot([y[stem_flag == 0], y[stem_flag == 1]], labels=["Non-STEM", "STEM"])
plt.title("Average Rating by STEM vs. Non-STEM")
plt.ylabel("Average Rating")
plt.grid(True)
plt.show()