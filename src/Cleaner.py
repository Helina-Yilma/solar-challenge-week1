import pandas as pd
from scipy import stats

class Cleaner:
    def __init__(self, df):
        self.df = df.copy()

    def convert_types(self):
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        if 'Cleaning' in self.df.columns:
            self.df['Cleaning'] = self.df['Cleaning'].astype('category')
        return self.df