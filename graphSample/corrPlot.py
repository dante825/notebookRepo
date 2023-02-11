import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./graphSample/input/attritionTrain.csv")

# Explore the distribution of the dataset
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='Attrition', data=df, ax=ax)
ax.set_title('Attrition Distribution')
ax.set_xlabel('Attrition')
ax.set_ylabel('Count')
plt.show()

# Explore the relationship between the target variable and a few key features
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='Attrition', y='Age', data=df, ax=axs[0])
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=axs[1])
sns.boxplot(x='Attrition', y='TotalWorkingYears', data=df, ax=axs[2])

axs[0].set_title('Age vs Attrition')
axs[0].set_xlabel('Attrition')
axs[0].set_ylabel('Age')

axs[1].set_title('Monthly Income vs Attrition')
axs[1].set_xlabel('Attrition')
axs[1].set_ylabel('Monthly Income')

axs[2].set_title('Total Working Years vs Attrition')
axs[2].set_xlabel('Attrition')
axs[2].set_ylabel('Total Working Years')

plt.show()


# Explore the relationship between the target variable and a categorical feature
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='Attrition', hue='Department', data=df, ax=ax)
ax.set_title('Attrition Distribution by Department')
ax.set_xlabel('Attrition')
ax.set_ylabel('Count')
plt.show()

# Explore the correlation between all numerical features
corr = df.corr()

# Keep only correlation higher than a threshold
threshold = 0.3
corr_threshold = corr[(corr > threshold) | (corr < -threshold)]

# Plot the heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_threshold, annot=True, cmap='coolwarm', fmt=".1f",
linewidths=.5, cbar_kws={'shrink': .5}, annot_kws={'size': 8})
plt.title('Correlations among features')
plt.show()

