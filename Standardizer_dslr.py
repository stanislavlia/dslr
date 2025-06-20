import pandas as pd
from custom import mean, std


class StandardizeDf:
    '''
    Initialize the standardizer with the target column name:
    standardizer = StandardizeDf(target)

    Fit the standardizer on the training data:
    standardizer.fit(df_train)

    Transform (standardize) any dataset using the fitted parameters:
    df_train_std = standardizer.transform(df_train)
    df_test_std = standardizer.transform(df_test)

    Note: Will return None if NaN/None detected in features during transform
    '''
    def __init__(self, target):
        self.target = target
        self.means = {}
        self.stds = {}


    def fit(self, df):
        for col in df.columns:
            if col != self.target:
                values = df[col].values
                self.means[col] = mean(values)
                self.stds[col] = std(values)


    def transform(self, df):
        df_standardized = df.copy()
        # Copy df to avoid modifying original
        for col in df.columns:
            if col != self.target:
                if col not in self.means or col not in self.stds:
                    raise ValueError(f'Standardizer was not fitted')
                m = self.means[col]
                s = self.stds[col]
                values = df[col].values
                standardized = []
                for v in values:
                    if v is not None and not v != v:
                        standardized.append((v - m) / s if s != 0 else 0)
                    else:
                        print('None/Nan detected -> return None')
                        return None
                df_standardized[col] = standardized
        df_standardized[self.target] = df[self.target]
        return df_standardized
