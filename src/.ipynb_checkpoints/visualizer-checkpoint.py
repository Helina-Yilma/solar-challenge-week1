import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

def plot_monthly_metrics(df, metrics=["GHI","DNI","DHI","Tamb"]):
    """Plot monthly mean values for multiple metrics for a single country."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["MonthName"] = df["Timestamp"].dt.strftime("%b")

    monthly_avg = (
        df.groupby("MonthName")[metrics]
        .mean()
        .reindex(MONTH_ORDER)
    )

    monthly_avg.plot(kind="bar", figsize=(12, 6))
    plt.title("Monthly Average Metrics")
    plt.xlabel("Month")
    plt.ylabel("Average Value")
    plt.tight_layout()
    plt.show()
def plot_daily_trends(df, date_col='Date'):
    """
    Plots daily average GHI, DNI, DHI, and Tamb.

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset containing date + metrics.
    date_col : str
        Name of the date column. Must be convertible to datetime.
    """

    df[date_col] = pd.to_datetime(df[date_col])

    # Group by date
    daily_avg = df.groupby(df[date_col].dt.date)[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

    plt.figure(figsize=(12, 5))
    daily_avg.plot(figsize=(12, 5))
    plt.title('Daily Average of GHI, DNI, DHI, and Tamb')
    plt.xlabel('Date')
    plt.ylabel('Average Value')
    plt.tight_layout()
    plt.show()