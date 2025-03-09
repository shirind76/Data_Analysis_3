import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = "Paris.csv"
df = pd.read_csv(file_path)

df.head()

columns_to_drop = ['name', 'host_id', 'host_name', 'last_review', 'license','neighbourhood_group']
df_cleaned = df.drop(columns=columns_to_drop)

df_cleaned = df_cleaned.dropna(subset=['price'])  
df_cleaned['reviews_per_month'].fillna(0, inplace=True) 
df_cleaned = pd.get_dummies(df_cleaned, columns=['room_type', 'neighbourhood'], drop_first=True)
df_cleaned.describe()


plt.figure(figsize=(8, 5))
sns.histplot(df_cleaned['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.show()


plt.figure(figsize=(14, 10))
corr_matrix = df_cleaned.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
plt.xticks(rotation=45, ha="right")  
plt.yticks(rotation=0)  
plt.title("Correlation Heatmap")
plt.show()
