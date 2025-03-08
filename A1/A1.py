import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from tabulate import tabulate
import seaborn as sns

df = pd.read_csv("morg2014.csv", quotechar='"', delimiter=",", encoding="utf-8", dtype="unicode")

columns_to_keep = [
    "lfsr94", "hhid", "intmonth", "stfips", "weight", "earnwke", "uhourse", "grade92",
    "race", "age", "sex", "marital", "ownchild", "chldpres", "prcitshp", "state", 
    "ind02", "occ2012", "class94", "unionmme", "unioncov"
]
df = df[columns_to_keep].copy()
df.rename(columns={"class94": "class", "uhourse": "uhours"}, inplace=True)

numeric_cols = ["hhid", "weight", "earnwke", "uhours", "grade92", "race", "age", 
                "sex", "marital", "ownchild", "chldpres", "occ2012"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

df["unioncov"].fillna(df["unioncov"].mode()[0], inplace=True)

df = df[(df.age.between(16, 64)) & (df.lfsr94.isin(["Employed-At Work", "Employed-Absent"]))]
df = df[(df.earnwke > 0) & (df.uhours > 0)]
df["earnings_per_hour"] = df["earnwke"] / df["uhours"]# Create earnings per hour variable

# ============================
# 2. occupation choosing
# ============================
finance_occupations = [800, 810, 820, 830, 840, 850, 860, 900, 910, 930, 940, 950]
df_finance = df[df["occ2012"].isin(finance_occupations)].copy()
# ============================
# 3. DESCRIPTIVE STATISTICS
# ============================
numeric_cols = df_finance.select_dtypes(include=['number']).columns
desc_stats = df_finance[numeric_cols].describe().T[['count', 'mean', 'std', 'min', 'max']]

desc_stats['median'] = df_finance[numeric_cols].median()

desc_stats = desc_stats.round(3)

desc_stats.rename(columns={
    'count': 'Obs',
    'mean': 'Mean',
    'std': 'Std.Dev.',
    'min': 'Min',
    'max': 'Max',
    'median': 'Median'
}, inplace=True)

# Drop categorical variables that don’t belong in summary stats
drop_vars = ['hhid', 'intmonth', 'race', 'ownchild', 'chldpres', 'prcitshp', 'state', 'sex', 'marital', 'lfsr94', 'occ2012']
desc_stats = desc_stats.drop(index=[col for col in drop_vars if col in desc_stats.index], errors='ignore')

# Rename variables for better presentation
variable_names = {
    'weight': 'Survey Weight',
    'earnwke': 'Earnings per Week',
    'uhours': 'Hours Worked per Week',
    'grade92': 'Education Level',
    'age': 'Age',
    'earnings_per_hour': 'Earnings per Hour'
}

desc_stats.rename(index={k: v for k, v in variable_names.items() if k in desc_stats.index}, inplace=True)

desc_stats.reset_index(inplace=True)
desc_stats.rename(columns={'index': 'Variable'}, inplace=True)

stata_table = tabulate(desc_stats, headers='keys', tablefmt='grid', showindex=False, numalign="right")

print(stata_table)

with open("stata_style_descriptive_statistics.txt", "w") as f:
    f.write(stata_table)
fig, ax = plt.subplots(figsize=(12, 6))
ax.text(0.01, 0.99, stata_table, fontsize=12, fontfamily="monospace", va='top', ha='left')

ax.axis('off')
plt.savefig("stata_style_descriptive_statistics.png", bbox_inches='tight', dpi=300)
plt.show()
# ============================
# 4. CHOOSE PREDICTOR & Model
# ============================
df_finance["white"] = (df_finance["race"] == 1).astype(int)
df_finance["afram"] = (df_finance["race"] == 2).astype(int)
df_finance["asian"] = (df_finance["race"] == 4).astype(int)
df_finance["hisp"] = df_finance["race"].isin([3, 5, 6, 7, 8, 11, 19, 21]).astype(int)

df_finance["nonUSborn"] = df_finance["prcitshp"].isin(
    ["Foreign Born, US Cit By Naturalization", "Foreign Born, Not a US Citizen"]
).astype(int)

df_finance["married"] = df_finance["marital"].isin([1, 2]).astype(int)
df_finance["divorced"] = (df_finance["marital"] == 3).astype(int)
df_finance["widowed"] = (df_finance["marital"] == 4).astype(int)
df_finance["nevermar"] = (df_finance["marital"] == 7).astype(int)
df_finance["union"] = ((df_finance["unionmme"] == "Yes") | (df_finance["unioncov"] == "Yes")).astype(int)


df_finance["education_low"] = (df_finance["grade92"] <= 39).astype(int) 
df_finance["education_middle"] = (df_finance["grade92"] <= 43).astype(int) 
df_finance["education_high"] = (df_finance["grade92"] > 43).astype(int)  

df_finance["nonUS_education_high"] = df_finance["nonUSborn"] * df_finance["education_high"]

X = df_finance.drop(columns=["earnings_per_hour", "prcitshp", "marital", "unionmme", "unioncov", "lfsr94"])
y = df_finance["earnings_per_hour"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_ols_model(X_train, y_train, features):
    X_model = X_train[features]
    X_model = sm.add_constant(X_model)  
    model = sm.OLS(y_train, X_model).fit()
    return model

features_1 = ["sex","age","education_low", "education_middle", "education_high"]
features_2 = features_1 + ["white", "afram", "asian", "hisp" ]
features_3 = features_2 + [ "nonUSborn", "married", "divorced", "widowed", "union"]
features_4 = features_3 + [ "nonUS_education_high"]  

model_1 = train_ols_model(X_train, y_train, features_1)
model_2 = train_ols_model(X_train, y_train, features_2)
model_3 = train_ols_model(X_train, y_train, features_3)
model_4 = train_ols_model(X_train, y_train, features_4)


def extract_model_results(models, feature_names):
    results = []
    for i, model in enumerate(models, start=1):
        coef = model.params.round(3)
        std_err = model.bse.round(3)
        p_values = model.pvalues.round(3)
        
        df_model = pd.DataFrame({
            "Variable": coef.index,
            f"Model {i} Coef.": coef.values,
            f"Model {i} Std. Err.": std_err.values,
            f"Model {i} p-value": p_values.values
        })
        results.append(df_model)
    
    
    final_results = results[0]
    for df in results[1:]:
        final_results = final_results.merge(df, on="Variable", how="outer")
    
    return final_results

models = [model_1, model_2, model_3, model_4]
feature_names = [features_1, features_2, features_3, features_4]
regression_summary = extract_model_results(models, feature_names)

# ============================
# 5. CALCULATING RMSE & BIC
# ============================

def calculate_rmse_fixed(model, df, features):
    X = df[features]  
    X = sm.add_constant(X)  
    y = df["earnings_per_hour"]
    y_pred = model.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred))

rmse_values_fixed = [
    calculate_rmse_fixed(model_1, df_finance, features_1),
    calculate_rmse_fixed(model_2, df_finance, features_2),
    calculate_rmse_fixed(model_3, df_finance, features_3),
    calculate_rmse_fixed(model_4, df_finance, features_4)
]

class OLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        X = sm.add_constant(X)  
        self.model = sm.OLS(y, X).fit()
        return self

    def predict(self, X):
        X = sm.add_constant(X)  
        return self.model.predict(X)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_values_fixed = []
for features in [features_1, features_2, features_3, features_4]:
    ols_regressor = OLSRegressor()
    cv_rmse = -cross_val_score(ols_regressor, X_train[features], y_train, 
                               scoring='neg_root_mean_squared_error', cv=cv).mean()
    cv_rmse_values_fixed.append(cv_rmse)

bic_values_fixed = [model_1.bic, model_2.bic, model_3.bic, model_4.bic]

regression_results_fixed = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3", "Model 4"],
    "RMSE Full Sample": rmse_values_fixed,
    "Cross-Validated RMSE": cv_rmse_values_fixed,
    "BIC": bic_values_fixed
})

print(regression_results_fixed)

bic_min, bic_max = regression_results_fixed["BIC"].min(), regression_results_fixed["BIC"].max()
regression_results_fixed["BIC Scaled"] = (regression_results_fixed["BIC"] - bic_min) / (bic_max - bic_min) * 10 

# ============================
# 6. Visulization
# ============================

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(regression_results_fixed["Model"], regression_results_fixed["RMSE Full Sample"], marker="o", label="Full Sample RMSE", color="blue")
ax1.plot(regression_results_fixed["Model"], regression_results_fixed["Cross-Validated RMSE"], marker="s", label="Cross-Validated RMSE", color="orange")
ax1.set_xlabel("Model Complexity")
ax1.set_ylabel("RMSE Values")
ax1.legend(loc="upper left")
ax1.set_title("Model Complexity vs Performance ")

ax2 = ax1.twinx()
ax2.plot(regression_results_fixed["Model"], regression_results_fixed["BIC Scaled"], marker="^", label="BIC (Scaled)", color="red")
ax2.set_ylabel("BIC (Scaled)")
ax2.legend(loc="upper right")

plt.savefig("performance_graph.png", bbox_inches="tight", dpi=300)
plt.show()


plt.figure(figsize=(7,5))
sns.kdeplot(df_finance[df_finance["sex"] == 1]["age"], label="Male", color="green")
sns.kdeplot(df_finance[df_finance["sex"] == 2]["age"], label="Female", color="blue")
plt.xlabel("Age (years)")
plt.ylabel("Density")
plt.legend()
plt.title("Age Distribution by Gender")
plt.savefig("age_distribution_gender.png", bbox_inches="tight", dpi=300)
plt.show()

# Visualization 2: Density plot by education level
plt.figure(figsize=(7,5))
sns.kdeplot(df_finance[df_finance["education_low"]== 1]["earnings_per_hour"], label="Low Education", color="red")
sns.kdeplot(df_finance[df_finance["education_middle"]== 1]["earnings_per_hour"], label= "Middle Education", color="green")
sns.kdeplot(df_finance[df_finance["education_high"] == 1]["earnings_per_hour"], label="High Education", color="blue")
plt.xlabel("Earning per Hours")
plt.ylabel("Density")
plt.legend()
plt.title("Education Distribution by Earning per Hours")
plt.savefig("distribution_education.png", bbox_inches="tight", dpi=300)
plt.show()

# Visualization 3: Histogram of earnings per hour
plt.figure(figsize=(7,5))
sns.histplot(df_finance["earnings_per_hour"], bins=30, kde=True, color="purple")
plt.xlabel("Earnings per Hour")
plt.ylabel("Frequency")
plt.title("Histogram of Earnings per Hour")
plt.savefig("earnings_histogram.png", bbox_inches="tight", dpi=300)
plt.show()
desc_stats_latex = desc_stats.to_latex(index=True, float_format="%.2f")
model_performance_latex = regression_results_fixed.to_latex(index=False, float_format="%.3f")
regression_summary_latex = regression_summary.to_latex(index=False, float_format="%.3f")
regression_summary_latex,desc_stats_latex, model_performance_latex
