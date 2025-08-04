#!/usr/bin/env python3
"""
Restaurant Annual Turnover Prediction
Goal: Beat RMSE of 12,337,503.37
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the training and test datasets"""

    # Load data
    train_df = pd.read_csv('Train_dataset.csv')
    test_df = pd.read_csv('Test_dataset.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Combine for preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['Annual Turnover'] = np.nan

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    return combined_df, train_df.shape[0]

def feature_engineering(df):
    """Advanced feature engineering"""

    # Handle missing values strategically
    df['Facebook Popularity Quotient'] = df['Facebook Popularity Quotient'].fillna(df['Facebook Popularity Quotient'].median())
    df['Instagram Popularity Quotient'] = df['Instagram Popularity Quotient'].fillna(df['Instagram Popularity Quotient'].median())

    # City cleaning - handle -1 and missing
    df['City'] = df['City'].replace('-1', 'Unknown')
    df['City'] = df['City'].fillna('Unknown')

    # Parse opening date
    df['Opening Day of Restaurant'] = pd.to_datetime(df['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
    df['opening_year'] = df['Opening Day of Restaurant'].dt.year
    df['opening_month'] = df['Opening Day of Restaurant'].dt.month
    df['opening_day'] = df['Opening Day of Restaurant'].dt.day
    df['restaurant_age'] = 2024 - df['opening_year']
    df['restaurant_age'] = np.where(df['restaurant_age'] < 0, 0, df['restaurant_age'])

    # Handle categorical variables
    df['Endorsed By'] = df['Endorsed By'].fillna('Not Specific')
    df['Endoresed By'] = df['Endoresed By'].fillna('Not Specific')  # Typo in test set
    df['Endorsed By'] = df['Endorsed By'].fillna(df.get('Endoresed By', 'Not Specific'))

    # Cuisine analysis
    df['cuisine_count'] = df['Cuisine'].str.count(',') + 1
    df['has_indian'] = df['Cuisine'].str.contains('indian', case=False, na=False).astype(int)
    df['has_italian'] = df['Cuisine'].str.contains('italian', case=False, na=False).astype(int)
    df['has_chinese'] = df['Cuisine'].str.contains('chinese', case=False, na=False).astype(int)

    # Location features
    df['is_business_hub'] = (df['Restaurant Location'] == 'Near Business Hub').astype(int)
    df['is_party_hub'] = (df['Restaurant Location'] == 'Near Party Hub').astype(int)

    # Social media engagement
    df['social_media_avg'] = (df['Facebook Popularity Quotient'] + df['Instagram Popularity Quotient']) / 2
    df['social_media_diff'] = df['Facebook Popularity Quotient'] - df['Instagram Popularity Quotient']

    # Rating features - handle NA values
    rating_cols = ['Order Wait Time', 'Staff Responsivness', 'Value for Money',
                   'Hygiene Rating', 'Food Rating', 'Overall Restaurant Rating',
                   'Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating',
                   'Live Sports Rating', 'Ambience', 'Lively', 'Service',
                   'Comfortablility', 'Privacy']

    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Create composite ratings
    service_ratings = ['Staff Responsivness', 'Service', 'Order Wait Time']
    df['service_score'] = df[service_ratings].mean(axis=1)

    food_ratings = ['Food Rating', 'Value for Money', 'Hygiene Rating']
    df['food_score'] = df[food_ratings].mean(axis=1)

    ambience_ratings = ['Ambience', 'Lively', 'Comfortablility', 'Privacy']
    df['ambience_score'] = df[ambience_ratings].mean(axis=1)

    entertainment_ratings = ['Live Music Rating', 'Comedy Gigs Rating', 'Live Sports Rating']
    df['entertainment_score'] = df[entertainment_ratings].mean(axis=1)

    # Binary features
    binary_cols = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
                   'Dedicated Parking', 'Open Sitting Available']
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

    df['amenity_score'] = df[binary_cols].sum(axis=1)

    # Restaurant tier and type
    df['Resturant Tier'] = pd.to_numeric(df['Resturant Tier'], errors='coerce').fillna(2)
    df['Restaurant City Tier'] = pd.to_numeric(df['Restaurant City Tier'], errors='coerce').fillna(1)

    # Premium features
    df['is_tier1_restaurant'] = (df['Resturant Tier'] == 1).astype(int)
    df['is_tier1_city'] = (df['Restaurant City Tier'] == 1).astype(int)
    df['is_celebrity_endorsed'] = (df['Endorsed By'] == 'Tier A Celebrity').astype(int)

    # Interaction features
    df['tier_celebrity_interaction'] = df['is_tier1_restaurant'] * df['is_celebrity_endorsed']
    df['social_rating_interaction'] = df['social_media_avg'] * df['Overall Restaurant Rating']
    df['age_rating_interaction'] = df['restaurant_age'] * df['Overall Restaurant Rating']

    return df

def encode_categorical_features(df):
    """Encode categorical features"""

    # Label encoding for high cardinality
    le_cols = ['City', 'Restaurant Type', 'Restaurant Theme', 'Endorsed By']

    encoders = {}
    for col in le_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders

def prepare_features(df, train_size):
    """Prepare final feature set"""

    # Drop non-predictive columns
    drop_cols = ['Registration Number', 'Annual Turnover', 'is_train',
                 'Opening Day of Restaurant', 'Cuisine']

    # Keep original categorical columns for now, but we'll use encoded versions
    keep_original = ['City', 'Restaurant Type', 'Restaurant Theme', 'Endorsed By', 'Endoresed By']

    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols].select_dtypes(include=[np.number])  # Only numeric features
    y = df['Annual Turnover'][:train_size]

    # Split back to train/test
    X_train = X[:train_size]
    X_test = X[train_size:]

    return X_train, X_test, y

class EnsemblePredictor:
    """Advanced ensemble model"""

    def __init__(self):
        self.models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'et': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }

        self.weights = None
        self.scaler = RobustScaler()

    def fit(self, X_train, y_train):
        """Fit ensemble with optimal weights"""

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train each model
        predictions = {}
        cv_scores = {}

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"Training {name}...")

            # Cross-validation score
            scores = cross_val_score(model, X_scaled, y_train,
                                   cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_scores[name] = np.sqrt(-scores.mean())
            print(f"{name} CV RMSE: {cv_scores[name]:.2f}")

            # Fit on full data
            model.fit(X_scaled, y_train)
            predictions[name] = model.predict(X_scaled)

        # Calculate optimal weights based on CV performance
        inv_scores = {name: 1/score for name, score in cv_scores.items()}
        total_inv = sum(inv_scores.values())
        self.weights = {name: inv_score/total_inv for name, inv_score in inv_scores.items()}

        print("Optimal weights:", self.weights)

        return self

    def predict(self, X_test):
        """Make ensemble predictions"""

        X_scaled = self.scaler.transform(X_test)

        ensemble_pred = np.zeros(len(X_test))

        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            ensemble_pred += self.weights[name] * pred

        return ensemble_pred

def main():
    """Main execution pipeline"""

    print("Loading and preprocessing data...")
    df, train_size = load_and_preprocess_data()

    print("Feature engineering...")
    df = feature_engineering(df)

    print("Encoding categorical features...")
    df, encoders = encode_categorical_features(df)

    print("Preparing features...")
    X_train, X_test, y_train = prepare_features(df, train_size)

    print(f"Final feature shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")

    # Remove any infinite or NaN values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

    print("Training ensemble model...")
    model = EnsemblePredictor()
    model.fit(X_train, y_train)

    # Validation
    print("Validating model...")
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model_val = EnsemblePredictor()
    model_val.fit(X_train_val, y_train_val)
    val_pred = model_val.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.2f}")

    # Final predictions
    print("Making final predictions...")
    test_predictions = model.predict(X_test)

    # Create submission
    test_df = pd.read_csv('Test_dataset.csv')
    submission = pd.DataFrame({
        'Registration Number': test_df['Registration Number'],
        'Annual Turnover': test_predictions
    })

    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)

    print(f"Submission saved to {submission_file}")
    print(f"Predicted RMSE should be better than 12,337,503.37")
    print(f"Validation RMSE: {val_rmse:.2f}")

    return submission

if __name__ == "__main__":
    main()
