import pandas as pd
import numpy as np
from scipy import stats

class Cleaner:
    def __init__(self, df):
        self.df = df.copy()

    def convert_types(self):
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        if 'Cleaning' in self.df.columns:
            self.df['Cleaning'] = self.df['Cleaning'].astype('category')
        return self.df
    def remove_outliers(self, cols, threshold=3):
        z = np.abs(stats.zscore(self.df[cols], nan_policy="omit"))
        mask = (z < threshold).all(axis=1)
        return self.df.loc[mask]