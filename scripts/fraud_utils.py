# fraud_utils.py
import pandas as pd
import numpy as np
import ast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.high_txn_threshold = None
        self.high_amt_threshold = None
        self.long_dist_threshold = None

    def extract_velocity(self, value):
        try:
            val = ast.literal_eval(value) if isinstance(value, str) else value
            return [val.get('num_transactions', 0), val.get('total_amount', 0.0)]
        except:
            return [0, 0.0]

    def fit(self, X, y=None):
        df = X.copy()

        # Drop unnecessary columns
        drop_cols = ['transaction_id', 'customer_id', 'card_number', 'device_fingerprint', 'ip_address',
                     'currency', 'city', 'city_size', 'high_risk_merchant', 'timestamp']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

        def time_of_day(hour):
            if 5 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 17:
                return "Afternoon"
            elif 17 <= hour < 21:
                return "Evening"
            else:
                return "Night"

        df['time_of_day'] = df['transaction_hour'].apply(time_of_day)

        # Parse velocity features
        df[['num_transactions_last_hour', 'total_amount_last_hour']] = df['velocity_last_hour'].apply(
            self.extract_velocity).tolist()
        df.drop(columns=['velocity_last_hour'], inplace=True)

        # Log amount
        df['log_amount'] = np.log1p(df['amount'])

        # Drop weak features
        df.drop(columns=['transaction_hour', 'amount'], inplace=True, errors='ignore')

        # Set thresholds
        self.high_txn_threshold = df['num_transactions_last_hour'].quantile(0.95)
        self.high_amt_threshold = df['total_amount_last_hour'].quantile(0.95)
        self.long_dist_threshold = df['distance_from_home'].quantile(0.95)

        # Fit label encoders
        categorical_columns = ['merchant_category', 'merchant_type', 'merchant', 'country', 'card_type',
                               'card_present', 'device', 'channel', 'time_of_day']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                le.fit(df[col])
                self.label_encoders[col] = le

        # Create features
        df['velocity_risk_score'] = (
            df['num_transactions_last_hour'] * 0.5 + df['total_amount_last_hour'] * 0.5
        )

        # Scale features
        scale_cols = ['log_amount', 'num_transactions_last_hour', 'total_amount_last_hour', 'velocity_risk_score']
        self.scaler.fit(df[scale_cols])
        return self

    def transform(self, X):
        df = X.copy()

        drop_cols = ['transaction_id', 'customer_id', 'card_number', 'device_fingerprint', 'ip_address',
                     'currency', 'city', 'city_size', 'high_risk_merchant', 'timestamp']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

        df[['num_transactions_last_hour', 'total_amount_last_hour']] = df['velocity_last_hour'].apply(
            self.extract_velocity).tolist()
        df.drop(columns=['velocity_last_hour'], inplace=True)

        df['log_amount'] = np.log1p(df['amount'])
        df.drop(columns=['transaction_hour', 'amount'], inplace=True, errors='ignore')

        df['high_transaction_flag'] = (df['num_transactions_last_hour'] > self.high_txn_threshold).astype(int)
        df['high_spending_flag'] = (df['total_amount_last_hour'] > self.high_amt_threshold).astype(int)
        df['long_distance_spender'] = (
            (df['distance_from_home'] > self.long_dist_threshold) &
            (df['log_amount'] > self.high_amt_threshold)
        ).astype(int)

        df['velocity_risk_score'] = (
            df['num_transactions_last_hour'] * 0.5 + df['total_amount_last_hour'] * 0.5
        )

        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        scale_cols = ['log_amount', 'num_transactions_last_hour', 'total_amount_last_hour', 'velocity_risk_score']
        df[scale_cols] = self.scaler.transform(df[scale_cols])

        return df


