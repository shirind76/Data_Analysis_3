import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("morg-2014-emp.csv", low_memory=False)     
df.info()
print("Data types before conversion:")
print(df.dtypes)

expected_numeric_cols = [
    'hhid', 'weight', 'earnwke', 'uhours', 'grade92', 
    'race', 'age', 'sex', 'marital', 'ownchild', 
    'chldpres', 'occ2012'
]

for col in expected_numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['unioncov'] = df['unioncov'].fillna(df['unioncov'].mode()[0])
df = df.drop(columns=["ethnic"])
print("Missing values by column:\n", df.isnull().sum())

df = df[df['uhours'] > 0]
df['earnings_per_hour'] = df['earnwke'] / df['uhours']

columns_for_model = ['hhid', 'intmonth', 'stfips', 'weight', 'earnwke', 
                     'uhours', 'grade92', 'race', 'age', 'sex', 'marital', 
                     'ownchild', 'chldpres', 'prcitshp', 'state', 'ind02', 
                     'occ2012', 'class', 'unionmme', 'unioncov', 'lfsr94', 
                     'earnings_per_hour']

df_model = df[columns_for_model]

print(df_model.info())
df_model=df_model.dropna()
#handling outliars

print('max:', df_model['earnings_per_hour'].max())
print('min', df_model['earnings_per_hour'].min())
print(sorted(df_model['earnings_per_hour'].unique()))
lower_bound = df['earnings_per_hour'].quantile(0.01)
upper_bound = df['earnings_per_hour'].quantile(0.99)

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

df_clean = df_model[(df_model['earnings_per_hour'] >= lower_bound) & (df_model['earnings_per_hour'] <= upper_bound)]

print("New min earnings per hour:", df_clean['earnings_per_hour'].min())
print("New max earnings per hour:", df_clean['earnings_per_hour'].max())
#descriptive statistice without outliar



#  descriptive statistics  for all data
import pandas as pd

# Compute descriptive statistics
desc_stats = df_clean.describe().T.round(2)  # Transpose for readability

# Drop 'hhid' since it's an ID column
desc_stats = desc_stats.drop(index=['hhid','intmonth','race''ownchild', 'chldpres', 'prcitshp', 'state' , 'sex', 'marital','ownchild','lfsr94'], errors='ignore')

# Add median separately
desc_stats['Median'] = df_clean.median()

# Rename variables for better presentation
variable_names = {
    'weight': 'Survey Weight',
    'earnwke': 'Earnings per Week',
    'uhours': 'Hours Worked per Week',
    'grade92': 'Education Level',
    'age': 'Age',
    'lfsr94': 'Labor Force Status',
    'earnings_per_hour': 'Earnings per Hour',
    'experience': 'Work Experience'
}

desc_stats.rename(index=variable_names, inplace=True)

# Save as CSV for easy access
desc_stats.to_csv("descriptive_statistics.csv")


# Create a figure
fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
ax.axis('tight')
ax.axis('off')

# Define colors for styling
row_colors = ['#f0f8ff' if i % 2 == 0 else '#d3d3d3' for i in range(len(desc_stats))]  # Alternating row colors
col_colors = ['#4682b4'] * len(desc_stats.columns)  # Blue header

# Create the table with formatted colors
table = ax.table(cellText=desc_stats.values,
                 colLabels=desc_stats.columns,
                 rowLabels=desc_stats.index,
                 cellLoc='center',
                 loc='center',
                 colColours=col_colors)

# Apply row colors
for i, key in enumerate(table._cells):
    cell = table._cells[key]
    if key[0] > 0:  # Ignore header row
        cell.set_facecolor(row_colors[key[0] % len(row_colors)])

# Save as an image
plt.savefig("descriptive_statistics_colored.png", bbox_inches='tight', dpi=300)
plt.show()





plt.scatter(df_clean['age'], df_clean['earnings_per_hour'], alpha=0.5)
plt.title("Age vs. Earnings per Hour")
plt.xlabel("Age")
plt.ylabel("Earnings per Hour")
plt.show()

df_clean['earnings_per_hour'].hist(bins=100)
plt.title("Distribution of Earnings per Hour")
plt.xlabel("Earnings per Hour")
plt.ylabel("Frequency")
plt.show()


