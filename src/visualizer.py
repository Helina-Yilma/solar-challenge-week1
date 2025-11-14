import seaborn as sns
import matplotlib.pyplot as plt

def plot_time_series(df):
    sns.lineplot(data=df, x='Timestamp', y='GHI', label='GHI', errorbar=None)
    sns.lineplot(data=df, x='Timestamp', y='DNI', label='DNI', errorbar=None)
    sns.lineplot(data=df, x='Timestamp', y='DHI', label='DHI', errorbar=None)
    sns.lineplot(data=df, x='Timestamp', y='Tamb', label='Tamb', linestyle='--', errorbar=None)
    plt.legend()
    plt.title('Solar Irradiance and Temperature Over Time')
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(df, cols):
    sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='crest')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()