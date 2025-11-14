import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load Data

url = "../data/benin.csv"
df = pd.read_csv(url)
print(df.shape)
df.head()



# Summary statistatistics and missing value report
print("Checking the data types of all columns")
print(df.info())
print("Change Timestamp column data type from object to DateTime and cleaning as categorical")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Ensure the Cleaning flag is categorical
df['Cleaning'] = df['Cleaning'].astype('category')
print("Since all columns except the timestamp and cleaning are numeric, here are their description")
df.describe()
print("Check for missing values:")
print(df.isna().sum())

print("Columns with >5% nulls")
percent_null = df.isna().mean() * 100
cols_with_5per_null = percent_null[percent_null>5].index.tolist()
print(cols_with_5per_null)

##  2. Outlier Detection & Basic Cleaning
###  2.1. Look for missing values, outliers, or incorrect entries
print("Visualize Outliers using Box Plot")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    plt.figure()
    plt.boxplot(df[c].dropna())
    plt.title(f"Box plot: {c}")
    plt.ylabel(c)
    plt.show()
### 2.2. Compute Z-scores for GHI, DNI, DHI, ModA, ModB, WS, WSgust; flag rows with |Z|>3.
print("Using z-score")
key_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
z = np.abs(stats.zscore(df[key_cols], nan_policy="omit"))
z_outliers_mask = (z > 3).any(axis=1)
print("Z-score outlier rows:", int(z_outliers_mask.sum()))
### 2.3. Drop or impute (median) missing values in key columns.
# There are no missing values in key columns.
df_cleaned = df.dropna(axis=1, how='all')
# 5️⃣ Save cleaned dataset
df_cleaned.to_csv("../data/benin_clean.csv", index=False)

print("Cleaned dataset saved to: data/benin_clean.csv")
## 3. Time Series Analysis

### 3.1. Line or bar charts of GHI, DNI, DHI, Tamb vs. Timestamp.
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Timestamp', y='GHI', label='GHI',errorbar = None)
sns.lineplot(data=df, x='Timestamp', y='DNI', label='DNI',errorbar = None)
sns.lineplot(data=df, x='Timestamp', y='DHI', label='DHI',errorbar = None)
sns.lineplot(data=df, x='Timestamp', y='Tamb', label='Tamb', linestyle='--',errorbar = None)
plt.title('GHI, DNI, DHI, and Tamb vs. Timestamp')
plt.xlabel('Timestamp')
plt.ylabel('Measurement')
plt.legend()
plt.show()

### 3.2. Patterns by month, trends throughout day, or anomalies, such as peaks in solar irradiance or temperature fluctuations. 
# Add month col
df['Month'] = df['Timestamp'].dt.month
monthly_avg = df.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

monthly_avg.plot(kind='bar', figsize=(10,5))
plt.title('Monthly Average of GHI, DNI, DHI, and Tamb')
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.tight_layout()
plt.show()
# Add hour col
df['Hour'] = df['Timestamp'].dt.hour
hourly_avg = df.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()
hourly_avg.plot(figsize=(10,5))
plt.title('Average Daily Pattern of GHI, DNI, DHI, and Tamb')
plt.xlabel('Hour of Day')
plt.ylabel('Average Value')
plt.tight_layout()
plt.show()

## 4. Cleaning Impact
print(df['Cleaning'].value_counts())
cleaning_avg = df.groupby('Cleaning')[['ModA', 'ModB']].mean().reset_index()
cleaning_melted = cleaning_avg.melt(id_vars='Cleaning', var_name='Module', value_name='Average_Value')
plt.figure(figsize=(8,5))
sns.barplot(data=cleaning_melted, x='Cleaning', y='Average_Value', hue='Module', palette='crest')
plt.title('Average Module Output Before and After Cleaning')
plt.xlabel('Cleaning Status')
plt.ylabel('Average Module Output')
plt.legend(title='Module')
plt.tight_layout()
plt.show()

## 5. Correlation and Relationship Analysis
### 5.1. Heatmap of correlations (GHI, DNI, DHI, TModA, TModB).
# Select relevant columns
corr_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']

# Compute correlation matrix
corr_matrix = df[corr_cols].corr()

# Plot heatmap
plt.figure(figsize=(7,5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='crest', linewidths=0.5, square=True)
plt.title('Correlation Heatmap: Solar Irradiance & Module Temperatures')
plt.tight_layout()
plt.show()
### 5.2. Scatter plots: WS, WSgust, WD vs. GHI; RH vs. Tamb or RH vs. GHI.
fig, axes = plt.subplots(2, 3, figsize=(16,10))
sns.scatterplot(data=df, x='WS', y='GHI', ax=axes[0,0])
sns.scatterplot(data=df, x='WSgust', y='GHI', ax=axes[0,1])
sns.scatterplot(data=df, x='WD', y='GHI', ax=axes[0,2])
sns.scatterplot(data=df, x='RH', y='Tamb', ax=axes[1,0])
sns.scatterplot(data=df, x='RH', y='GHI', ax=axes[1,1])

axes[1,2].axis('off')  # empty last subplot
fig.suptitle('Environmental Variables vs. Solar Irradiance & Temperature', fontsize=14)
plt.tight_layout()
plt.show()

## 6. Wind & Distribution Analysis
### 6.1. Wind rose or radial bar plot of WS/WD.
# Make sure direction and speed columns are valid
df = df.dropna(subset=['WD', 'WS'])
df['WD'] = df['WD'] % 360  # normalize directions
# Define compass sectors (16 bins)
bins = np.arange(-11.25, 371.25, 22.5)
labels = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
]
df['WD_bin'] = pd.cut(df['WD'], bins=bins, labels=labels, right=False)

# Compute average WS per direction
wind_summary = df.groupby('WD_bin')['WS'].mean().reindex(labels)
# Convert to radians
angles = np.deg2rad(np.linspace(0, 360, len(wind_summary), endpoint=False))

# Setup polar plot
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

# Bars
bars = ax.bar(angles, wind_summary, width=np.deg2rad(22.5), color=sns.color_palette('crest', len(wind_summary)), edgecolor='black')

# Format
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title('Wind Rose: Average Wind Speed by Direction', va='bottom', fontsize=13)

# Add direction labels
ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax.set_yticklabels([])

plt.tight_layout()
plt.show()
### 6.2. Histograms for GHI and one other variable (e.g. WS).
fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.histplot(df['GHI'], bins=30, kde=True, color='orange', ax=axes[0])
axes[0].set_title('Global Horizontal Irradiance (GHI)')
axes[0].set_xlabel('GHI (W/m²)')
axes[0].set_ylabel('Frequency')

sns.histplot(df['WS'], bins=30, kde=True, color='teal', ax=axes[1])
axes[1].set_title('Wind Speed (WS)')
axes[1].set_xlabel('Wind Speed (m/s)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

## 7. Temprature Analysis
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='RH', y='Tamb', alpha=0.6, color='royalblue')
plt.title('Relationship between Relative Humidity (RH) and Ambient Temperature (Tamb)')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Ambient Temperature (°C)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='RH', y='GHI', alpha=0.6, color='darkorange')
plt.title('Effect of Relative Humidity (RH) on Solar Irradiance (GHI)')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Global Horizontal Irradiance (W/m²)')
plt.tight_layout()
plt.show()

## 8.Bubble Chart
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='Tamb',
    y='GHI',
    size='RH',         # bubble size based on Relative Humidity
    sizes=(20, 300),   # min and max bubble sizes
    hue='RH',          # optional: color by humidity too
    palette='coolwarm',
    alpha=0.6,
    edgecolor='black'
)

plt.title('Bubble Chart: GHI vs. Tamb (Bubble size = RH)')
plt.xlabel('Ambient Temperature (°C)')
plt.ylabel('Global Horizontal Irradiance (W/m²)')
plt.legend(title='Relative Humidity (%)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()