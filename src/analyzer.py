def monthly_trends(df):
    df['Month'] = df['Timestamp'].dt.month
    return df.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

def hourly_trends(df):
    df['Hour'] = df['Timestamp'].dt.hour
    return df.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()