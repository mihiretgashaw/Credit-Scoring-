import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, amount_col):
        self.group_col = group_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.group_col)[self.amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_amount', 'std')
        ]).reset_index()
        X_agg = pd.merge(X, agg, on=self.group_col, how='left')
        X_agg['std_amount'] = X_agg['std_amount'].fillna(0)
        return X_agg

class TemporalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.datetime_col] = pd.to_datetime(X_[self.datetime_col])
        X_['trans_hour'] = X_[self.datetime_col].dt.hour
        X_['trans_day'] = X_[self.datetime_col].dt.day
        X_['trans_month'] = X_[self.datetime_col].dt.month
        X_['trans_year'] = X_[self.datetime_col].dt.year
        return X_

def build_feature_engineering_pipeline(
    group_col,
    amount_col,
    datetime_col,
    numeric_cols,
    categorical_cols
):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # remove sparse param
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    full_pipeline = Pipeline([
        ('agg', AggregateFeatures(group_col=group_col, amount_col=amount_col)),
        ('temporal', TemporalExtractor(datetime_col=datetime_col)),
        ('preproc', preprocessor)
    ])

    return full_pipeline

if __name__ == "__main__":
    df = pd.read_csv('data/raw/data.csv', low_memory=False)
    df.drop(columns=['Unnamed: 16', 'Unnamed: 17'], inplace=True, errors='ignore')

    # Use actual column names
    group_col = 'CustomerId'
    amount_col = 'Amount'
    datetime_col = 'TransactionStartTime'

    numeric_cols = [
        'total_amount', 'avg_amount', 'transaction_count', 'std_amount',
        'trans_hour', 'trans_day', 'trans_month', 'trans_year'
    ]

    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 'ChannelId']

    pipeline = build_feature_engineering_pipeline(
        group_col=group_col,
        amount_col=amount_col,
        datetime_col=datetime_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    processed = pipeline.fit_transform(df)

    # Save processed output
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/processed.npy', processed)
    print("✅ Processed feature matrix saved to data/processed/processed.npy")
    print("📐 Shape:", processed.shape)
