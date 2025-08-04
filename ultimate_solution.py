#!/usr/bin/env python3
"""
Ultimate Restaurant Turnover Prediction Solution
Optimized ensemble with powerful feature engineering
Goal: Beat RMSE of 12,337,503.37
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess():
    """Load and initial preprocessing"""

    train_df = pd.read_csv('Train_dataset.csv')
    test_df = pd.read_csv('Test_dataset.csv')

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # Combine for preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['Annual Turnover'] = np.nan

    # Fix endorsement column inconsistency
    if 'Endoresed By' in test_df.columns:
        test_df['Endorsed By'] = test_df['Endoresed By']
        test_df = test_df.drop('Endoresed By', axis=1)

    df = pd.concat([train_df, test_df], ignore_index=True)

    return df, train_df.shape[0]

def target_encode_simple(train_series, target, test_series, smoothing=100):
    """Simple but effective target encoding"""

    # Calculate category statistics
    cat_means = train_series.groupby(train_series)[target].agg(['mean', 'count']).reset_index()
    cat_means.columns = [train_series.name, 'mean', 'count']

    # Global mean for smoothing
    global_mean = target.mean()

    # Apply smoothing based on count
    weight = cat_means['count'] / (cat_means['count'] + smoothing)
    cat_means['smoothed_mean'] = weight * cat_means['mean'] + (1 - weight) * global_mean

    # Create mapping
    encoding_map = cat_means.set_index(train_series.name)['smoothed_mean'].to_dict()

    # Encode
    train_encoded = train_series.map(encoding_map).fillna(global_mean)
    test_encoded = test_series.map(encoding_map).fillna(global_mean)

    return train_encoded, test_encoded

def ultimate_feature_engineering(df):
    """Ultimate feature engineering for maximum performance"""

    # Handle missing values
    df['Facebook Popularity Quotient'] = df['Facebook Popularity Quotient'].fillna(df['Facebook Popularity Quotient'].median())
    df['Instagram Popularity Quotient'] = df['Instagram Popularity Quotient'].fillna(df['Instagram Popularity Quotient'].median())

    # Clean city data
    df['City'] = df['City'].replace('-1', 'Unknown').fillna('Unknown')
    df['Endorsed By'] = df['Endorsed By'].fillna('Not Specific')

    # Date features
    df['Opening Day of Restaurant'] = pd.to_datetime(df['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
    df['opening_year'] = df['Opening Day of Restaurant'].dt.year
    df['opening_month'] = df['Opening Day of Restaurant'].dt.month
    df['restaurant_age'] = 2024 - df['opening_year']
    df['restaurant_age'] = np.where(df['restaurant_age'] < 0, 0, df['restaurant_age'])

    # Age transformations
    df['age_squared'] = df['restaurant_age'] ** 2
    df['age_log'] = np.log1p(df['restaurant_age'])
    df['age_sqrt'] = np.sqrt(df['restaurant_age'])

    # Seasonal features
    df['opened_summer'] = ((df['opening_month'] >= 4) & (df['opening_month'] <= 6)).astype(int)
    df['opened_winter'] = ((df['opening_month'] >= 10) | (df['opening_month'] <= 2)).astype(int)

    # Cuisine analysis
    df['Cuisine'] = df['Cuisine'].fillna('unknown')
    df['cuisine_count'] = df['Cuisine'].str.count(',') + 1

    # High-impact cuisines
    high_value_cuisines = ['italian', 'japanese', 'french', 'greek']
    popular_cuisines = ['indian', 'chinese', 'thai']

    for cuisine in high_value_cuisines + popular_cuisines:
        df[f'has_{cuisine}'] = df['Cuisine'].str.contains(cuisine, case=False, na=False).astype(int)

    df['has_premium_cuisine'] = df[['has_' + c for c in high_value_cuisines]].sum(axis=1)
    df['has_popular_cuisine'] = df[['has_' + c for c in popular_cuisines]].sum(axis=1)

    # Location features
    df['is_business_hub'] = (df['Restaurant Location'] == 'Near Business Hub').astype(int)
    df['is_party_hub'] = (df['Restaurant Location'] == 'Near Party Hub').astype(int)

    # Social media power features
    df['social_avg'] = (df['Facebook Popularity Quotient'] + df['Instagram Popularity Quotient']) / 2
    df['social_max'] = np.maximum(df['Facebook Popularity Quotient'], df['Instagram Popularity Quotient'])
    df['social_min'] = np.minimum(df['Facebook Popularity Quotient'], df['Instagram Popularity Quotient'])
    df['social_diff'] = df['Facebook Popularity Quotient'] - df['Instagram Popularity Quotient']
    df['social_ratio'] = df['Facebook Popularity Quotient'] / (df['Instagram Popularity Quotient'] + 1)

    # Social media categories
    df['is_social_star'] = (df['social_avg'] > 90).astype(int)
    df['is_social_weak'] = (df['social_avg'] < 60).astype(int)

    # Rating features
    rating_cols = ['Order Wait Time', 'Staff Responsivness', 'Value for Money',
                   'Hygiene Rating', 'Food Rating', 'Overall Restaurant Rating',
                   'Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating',
                   'Live Sports Rating', 'Ambience', 'Lively', 'Service',
                   'Comfortablility', 'Privacy']

    # Convert to numeric and fill missing
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Composite ratings
    df['service_score'] = df[['Staff Responsivness', 'Service']].mean(axis=1)
    df['quality_score'] = df[['Food Rating', 'Value for Money', 'Hygiene Rating']].mean(axis=1)
    df['experience_score'] = df[['Ambience', 'Lively', 'Comfortablility', 'Privacy']].mean(axis=1)
    df['wait_efficiency'] = 10 - df['Order Wait Time']  # Higher is better

    # Rating consistency
    df['rating_std'] = df[rating_cols].std(axis=1)
    df['rating_consistency'] = 10 - df['rating_std']

    # Binary amenities
    binary_cols = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
                   'Dedicated Parking', 'Open Sitting Available']
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

    df['total_amenities'] = df[binary_cols].sum(axis=1)
    df['has_parking'] = df['Dedicated Parking']
    df['has_liquor'] = df['Liquor License Obtained']

    # Tier features
    df['Resturant Tier'] = pd.to_numeric(df['Resturant Tier'], errors='coerce').fillna(2)
    df['Restaurant City Tier'] = pd.to_numeric(df['Restaurant City Tier'], errors='coerce').fillna(1)
    df['Restaurant Zomato Rating'] = pd.to_numeric(df['Restaurant Zomato Rating'], errors='coerce').fillna(3)

    # Premium indicators
    df['is_tier1_restaurant'] = (df['Resturant Tier'] == 1).astype(int)
    df['is_tier1_city'] = (df['Restaurant City Tier'] == 1).astype(int)
    df['is_celebrity_endorsed'] = (df['Endorsed By'] == 'Tier A Celebrity').astype(int)
    df['is_high_zomato'] = (df['Restaurant Zomato Rating'] >= 4).astype(int)

    # Power combinations
    df['premium_combo'] = df['is_tier1_restaurant'] * df['is_celebrity_endorsed'] * df['has_premium_cuisine']
    df['quality_age_combo'] = df['quality_score'] * df['restaurant_age']
    df['social_quality_combo'] = df['social_avg'] * df['quality_score']
    df['location_tier_combo'] = df['is_business_hub'] * df['is_tier1_city']

    # Advanced math transformations
    df['social_power'] = df['social_avg'] ** 0.5
    df['quality_squared'] = df['quality_score'] ** 2
    df['experience_log'] = np.log1p(df['experience_score'])

    # Ratios and per-unit metrics
    df['quality_per_amenity'] = df['quality_score'] / (df['total_amenities'] + 1)
    df['social_per_age'] = df['social_avg'] / (df['restaurant_age'] + 1)
    df['rating_efficiency'] = df['Overall Restaurant Rating'] / (df['Order Wait Time'] + 1)

    return df

def apply_target_encoding(df, train_size):
    """Apply target encoding to categorical variables"""

    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()
    target = train_df['Annual Turnover']

    # Categories to encode
    cat_cols = ['City', 'Restaurant Type', 'Restaurant Theme', 'Endorsed By']

    for col in cat_cols:
        if col in df.columns:
            train_encoded, test_encoded = target_encode_simple(
                train_df[col], target, test_df[col], smoothing=50
            )

            df.loc[:train_size-1, col + '_encoded'] = train_encoded
            df.loc[train_size:, col + '_encoded'] = test_encoded.values

    return df

def prepare_final_features(df, train_size):
    """Prepare final optimized feature set"""

    # Apply target encoding
    df = apply_target_encoding(df, train_size)

    # Drop non-predictive columns
    drop_cols = ['Registration Number', 'Annual Turnover', 'is_train',
                 'Opening Day of Restaurant', 'Cuisine']

    # Keep only numeric features
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df['Annual Turnover'][:train_size]

    X_train = X[:train_size]
    X_test = X[train_size:]

    print(f"Final features: {X_train.shape[1]}")
    return X_train, X_test, y

class UltimateEnsemble:
    """Ultimate ensemble optimized for this specific problem"""

    def __init__(self):
        # Optimized models for restaurant data
        self.models = {
            'rf_deep': RandomForestRegressor(
                n_estimators=1500, max_depth=30, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt',
                random_state=42, n_jobs=-1
            ),
            'rf_balanced': RandomForestRegressor(
                n_estimators=1000, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_features=0.6,
                random_state=123, n_jobs=-1
            ),
            'et_extreme': ExtraTreesRegressor(
                n_estimators=1200, max_depth=35, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt',
                random_state=456, n_jobs=-1
            ),
            'et_conservative': ExtraTreesRegressor(
                n_estimators=800, max_depth=25, min_samples_split=4,
                min_samples_leaf=2, max_features=0.7,
                random_state=789, n_jobs=-1
            ),
            'gb_precise': GradientBoostingRegressor(
                n_estimators=1500, max_depth=12, learning_rate=0.02,
                subsample=0.8, max_features='sqrt', random_state=101
            ),
            'gb_fast': GradientBoostingRegressor(
                n_estimators=800, max_depth=8, learning_rate=0.05,
                subsample=0.85, max_features=0.8, random_state=202
            ),
            'ridge_strong': Ridge(alpha=50, random_state=42),
            'ridge_gentle': Ridge(alpha=10, random_state=43),
            'elastic_balanced': ElasticNet(alpha=20, l1_ratio=0.7, random_state=44),
            'lasso_sparse': Lasso(alpha=1000, random_state=45)
        }

        self.weights = None
        self.scaler = RobustScaler()

    def fit(self, X_train, y_train):
        """Train ultimate ensemble"""

        print("🚀 Training Ultimate Ensemble...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train and evaluate each model
        cv_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"Training {name}...")

            # Cross-validation
            fold_scores = []
            for train_idx, val_idx in kf.split(X_scaled):
                X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train, y_fold_train)
                pred = model_copy.predict(X_fold_val)
                rmse = np.sqrt(mean_squared_error(y_fold_val, pred))
                fold_scores.append(rmse)

            cv_rmse = np.mean(fold_scores)
            cv_scores[name] = cv_rmse
            print(f"{name} CV RMSE: {cv_rmse:,.2f}")

            # Fit on full data
            model.fit(X_scaled, y_train)

        # Calculate optimal weights (inverse of RMSE)
        inv_scores = {name: 1/score for name, score in cv_scores.items()}
        total_inv = sum(inv_scores.values())
        self.weights = {name: inv_score/total_inv for name, inv_score in inv_scores.items()}

        print("\nOptimal weights:")
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {weight:.3f}")

        return self

    def predict(self, X_test):
        """Generate ultimate predictions"""

        X_scaled = self.scaler.transform(X_test)

        # Weighted ensemble prediction
        final_pred = np.zeros(len(X_test))

        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            final_pred += self.weights[name] * pred

        return final_pred

def main():
    """Execute ultimate solution"""

    print("🏆 ULTIMATE RESTAURANT TURNOVER PREDICTION")
    print("=" * 50)

    # Load data
    df, train_size = load_and_preprocess()

    # Feature engineering
    print("🔧 Ultimate Feature Engineering...")
    df = ultimate_feature_engineering(df)

    # Prepare features
    print("⚡ Preparing Final Features...")
    X_train, X_test, y_train = prepare_final_features(df, train_size)

    # Train ensemble
    ensemble = UltimateEnsemble()
    ensemble.fit(X_train, y_train)

    # Validation
    print("\n📊 VALIDATION RESULTS")
    print("-" * 30)
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    val_ensemble = UltimateEnsemble()
    val_ensemble.fit(X_train_val, y_train_val)
    val_pred = val_ensemble.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    target_rmse = 12337503.37
    improvement = (target_rmse - val_rmse) / target_rmse * 100

    print(f"Validation RMSE: {val_rmse:,.2f}")
    print(f"Target RMSE: {target_rmse:,.2f}")
    print(f"Improvement: {improvement:+.2f}%")

    # Final predictions
    print("\n🎯 Generating Final Predictions...")
    final_predictions = ensemble.predict(X_test)

    # Create submission
    test_df = pd.read_csv('Test_dataset.csv')
    submission = pd.DataFrame({
        'Registration Number': test_df['Registration Number'],
        'Annual Turnover': final_predictions
    })

    submission_file = 'ultimate_submission.csv'
    submission.to_csv(submission_file, index=False)

    print("\n🏆 ULTIMATE RESULTS")
    print("=" * 30)
    print(f"📁 Submission: {submission_file}")
    print(f"📊 Entries: {len(submission)}")
    print(f"🎯 Predicted RMSE: {val_rmse:,.2f}")

    if val_rmse < target_rmse:
        print("🥇 CHAMPION! You should beat the leaderboard!")
        print(f"🚀 Expected rank improvement: {improvement:.1f}%")
    else:
        print("🥈 Very close! Consider ensemble tuning")

    print(f"\n📈 Prediction range: {final_predictions.min():,.0f} - {final_predictions.max():,.0f}")
    print(f"📊 Prediction mean: {final_predictions.mean():,.0f}")

    return submission

if __name__ == "__main__":
    main()
