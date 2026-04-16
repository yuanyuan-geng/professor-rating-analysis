# Professor Rating Analysis
A data science capstone project exploring factors associated with professor ratings and building predictive models using statistical testing and machine learning, based on Rate My Professors data.

## Research Questions
### Bias & Group Differences
- Is there evidence of gender bias in student evaluations of professors?
- Do professors rated as "hot" receive significantly higher ratings?
### Correlational Analysis
- Is teaching experience associated with teaching quality?
- How do course difficulty and student retake intention relate to professor ratings?
- Do professors teaching more online classes receive different ratings?
### Predictive Modeling
- Can course difficulty alone and other factors predict a professor's average rating? (Linear Regression)
- Can average rating alone and other factors predict whether a professor receives a "pepper" (i.e., rated as "hot" by students)? (Logistic Regression)
### Exploratory Analysis
- Do professors in STEM fields receive lower ratings than those in non-STEM fields?

## Methods & Tools
- Language: Python (NumPy, Pandas, SciPy, scikit-learn, Matplotlib)
- Statistical Tests: Mann-Whitney U, Pearson Correlation, Spearman Correlation
- Models: Linear Regression, Logistic Regression
- Other: Feature selection, collinearity analysis, class imbalance handling

## Key Findings
### Bias & Group Differences
- Male professors received higher ratings than female professors (p = 0.00084)
- Professors who received a "pepper" had higher ratings (median 4.5 vs. 3.6, p ≈ 0)
### Correlational Analysis
- Teaching experience showed a statistically significant but practically negligible positive correlation with ratings (ρ = 0.028)
- Course difficulty was moderately negatively correlated with ratings (r = -0.619)
- Student retake intention was strongly positively correlated with ratings (r = 0.88)
- Online teaching modality showed no significant effect on ratings (p = 0.331)
### Predictive Modeling
- Difficulty-only linear regression explained 38.3% of variance in ratings (R² = 0.383, RMSE = 0.744)
- Multivariate linear regression improved performance (R² = 0.485, RMSE = 0.679)
- Logistic regression predicted "pepper" from average rating alone with AUROC = 0.769
- Adding more predictors yielded marginal improvement (AUROC = 0.779)
### Exploratory Analysis
- STEM vs. non-STEM field showed no significant difference in professor ratings (p = 0.90)

## Files
- `Professor Rating Analysis Project Report.pdf` — project report with visualizations
- `professor-rating-analysis-project.py` — full code
